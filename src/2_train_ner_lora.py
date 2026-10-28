"""Train a small LoRA pilot and keep the best generated-entity F1 adapter.

Conversational SFT and PEFT: https://huggingface.co/docs/trl/sft_trainer
LoRA configuration: https://huggingface.co/docs/peft/package_reference/lora
Callback and stop control: https://huggingface.co/docs/transformers/trainer_callbacks
"""

import json
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from ner_evaluation import OUTPUT_DIR, evaluate, read_split, sample_records
from ner_prompt import LABELS, MODEL_ID, SYSTEM_PROMPT


SEED = 42
BEST_DIR = OUTPUT_DIR / "best_adapter"


def training_example(row):
    entities = [
        {"text": entity["text"], "label": entity["label"]}
        for entity in row["entityMentions"]
        if entity["label"] in LABELS
    ]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["sentText"]},
            {"role": "assistant", "content": json.dumps({"entities": entities}, ensure_ascii=False)},
        ]
    }


class BestEntityF1(TrainerCallback):
    """Evaluate generated JSON at each Trainer evaluation step."""

    def __init__(self, validation, tokenizer):
        self.validation = validation
        self.tokenizer = tokenizer
        self.best_f1 = -1.0
        self.no_improvement = 0

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # https://huggingface.co/docs/transformers/trainer_callbacks
        was_training = model.training
        use_cache = model.config.use_cache
        model.eval()
        model.config.use_cache = True
        try:
            metrics = evaluate(
                self.validation,
                model,
                self.tokenizer,
                description=f"Validation step {state.global_step}",
            )
        finally:
            model.config.use_cache = use_cache
            if was_training:
                model.train()

        f1 = metrics["entity_f1"]
        print(f"Validation entity F1: {f1:.3f}")
        if f1 > self.best_f1 + 0.005:
            self.best_f1 = f1
            self.no_improvement = 0
            BEST_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_DIR)
            self.tokenizer.save_pretrained(BEST_DIR)
            print(f"Best adapter saved: {BEST_DIR}")
        else:
            self.no_improvement += 1
            if self.no_improvement >= 2:
                control.should_training_stop = True
                print("Stopped after two F1 checks without improvement")
        return control


def main():
    # differenciate between the training and the testing data
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_records = read_split("valid")
    valid_texts = {row["sentText"] for row in valid_records}
    seen = set()
    train_records = []
    for row in read_split("train"):
        text = row["sentText"]
        if text not in seen and text not in valid_texts:
            train_records.append(row)
            seen.add(text)
    train_records = random.Random(SEED).sample(
        train_records, min(2000, len(train_records))
    )
    validation = sample_records(valid_records, 200, SEED)
    train_dataset = Dataset.from_list([training_example(row) for row in train_records])
    eval_dataset = Dataset.from_list(
        [training_example(row) for _, row in validation]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    print(f"Loading {MODEL_ID}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=0, disable_mmap=True
    )
    model.config.use_cache = False
    # https://huggingface.co/docs/trl/sft_trainer
    peft_config = LoraConfig(
        r=8, #Rank: no. of trainable parameters
        lora_alpha=16, # scaling factor, 2 x r
        lora_dropout=0.05, # drops 5% of the activations, to avoid overfitting
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # specifies which layer is the multi head attention
        task_type="CAUSAL_LM", # casual language modeling, so that the net token is left to right
    )
    args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "training"),
        max_steps=75, # max optimization steps for trainning
        per_device_train_batch_size=4, # simultaneously on a single GPU
        gradient_accumulation_steps=4, # batch size is 4 * 4 = 16
        learning_rate=1e-4,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="no", # saving space/time and prevvents writing model check points.
        logging_steps=5, # print loss and performance metrics to the console
        bf16=True, # Brain floaing
        gradient_checkpointing=True,
        assistant_only_loss=True,
        max_length=512,
        report_to="none",
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[BestEntityF1(validation, tokenizer)],
    )
    print(f"Training rows: {len(train_records)}; validation rows: {len(validation)}")
    trainer.train()
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
