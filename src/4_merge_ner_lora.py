"""Merge the demo's best LoRA adapter into a standalone model.

PEFT merge example: https://huggingface.co/docs/peft/developer_guides/checkpoint#merge-the-weights
Model serialization: https://huggingface.co/docs/transformers/main_classes/model
"""

import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ner_evaluation import OUTPUT_DIR
from ner_prompt import MODEL_ID


def main():
    started = time.perf_counter()
    adapter_dir = OUTPUT_DIR / "best_adapter"
    output_dir = OUTPUT_DIR / "M2_merged"
    if output_dir.exists():
        raise FileExistsError(output_dir)

    print(
        f"Loading base model and adapter; elapsed {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=0, disable_mmap=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    # https://huggingface.co/docs/peft/developer_guides/checkpoint#merge-the-weights
    print(f"Merging adapter; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    merged_model = model.merge_and_unload(safe_merge=True, progressbar=True)
    print(f"Saving merged model; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved: {output_dir}")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
