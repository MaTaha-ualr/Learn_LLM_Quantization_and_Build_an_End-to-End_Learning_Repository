"""Shared Qwen generation and exact entity scoring for this dataset.

Chat and batched generation: https://huggingface.co/docs/transformers/chat_templating
Qwen3 non-thinking mode: https://huggingface.co/Qwen/Qwen3-0.6B
Model loading: https://huggingface.co/docs/transformers/main_classes/model
Counting duplicate labels: https://docs.python.org/3/library/collections.html#collections.Counter
NER precision, recall, and F1 background: https://huggingface.co/docs/transformers/tasks/token_classification
Progress bars: https://tqdm.github.io/
"""

import json
import random
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ner_prompt import LABELS, MODEL_ID, SYSTEM_PROMPT


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "curated"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS = {
    "M0": MODEL_ID,
    "M1": OUTPUT_DIR / "M1_int8",
    "M2": OUTPUT_DIR / "M2_merged",
    "M3": OUTPUT_DIR / "M3_int8",
}


def read_split(split):
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown data split: {split}")
    path = DATA_DIR / f"{split}.json"
    # JSON format documentation: https://docs.python.org/3/library/json.html
    with path.open(encoding="utf-8-sig") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict) or not isinstance(row.get("sentText"), str):
            raise ValueError(f"Invalid record {index} in {path}: expected sentText")
        if not isinstance(row.get("entityMentions"), list):
            raise ValueError(f"Invalid record {index} in {path}: expected entityMentions")
    return records


def sample_records(records, size, seed):
    """Keep original one-based indices so variants can be compared."""
    indexed = list(enumerate(records, start=1))
    return sorted(random.Random(seed).sample(indexed, min(size, len(indexed))))


def load_model(source):
    # https://huggingface.co/docs/transformers/main_classes/model
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype="auto", device_map=0, disable_mmap=True
    )
    model.eval()
    return model, tokenizer


def generate(sentences, model, tokenizer):
    # https://huggingface.co/docs/transformers/chat_templating
    # https://huggingface.co/Qwen/Qwen3-0.6B
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ]
        for sentence in sentences
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=192, do_sample=False)
    return tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

# validate if the data is in json format - if it fails then it becomes invalid
def parse_entities(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], False
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        return [], False
    valid = all(
        isinstance(entity, dict)
        and isinstance(entity.get("text"), str)
        and isinstance(entity.get("label"), str)
        for entity in data["entities"]
    )
    if not valid:
        return [], False
    return [entity for entity in data["entities"] if entity["label"] in LABELS], True


def evaluate(
    indexed_records,
    model,
    tokenizer,
    output_path=None,
    description="Evaluating",
    ignore_extra=False,
):
    """Count exact (text, label) matches; optionally save each prediction."""
    # This project's generated JSON uses (text, label) pairs instead of token tags.
    tp = fp = fn = exact = valid_json = 0
    gold_entity_count = captured_gold_count = fully_covered_examples = 0
    rows = []
    for start in tqdm(range(0, len(indexed_records), 2), desc=description, unit="batch"):
        batch = indexed_records[start : start + 2]
        predictions = generate([row["sentText"] for _, row in batch], model, tokenizer)
        for (index, row), raw in zip(batch, predictions):
            predicted, valid = parse_entities(raw.strip())
            expected = [
                {"text": entity["text"], "label": entity["label"]}
                for entity in row["entityMentions"]
                if entity["label"] in LABELS
            ]
            gold = Counter((entity["text"], entity["label"]) for entity in expected)
            guess = Counter((entity["text"], entity["label"]) for entity in predicted)
            matched = gold & guess
            missing = gold - guess
            matched_count = sum(matched.values())
            gold_count = sum(gold.values())
            tp += matched_count
            fp += sum((guess - gold).values())
            fn += sum(missing.values())
            exact += gold == guess
            valid_json += valid
            gold_entity_count += gold_count
            captured_gold_count += matched_count
            fully_covered_examples += not missing
            if output_path is not None:
                result = {
                    "exampleIndex": index,
                    "sentId": row.get("sentId"),
                    "sentText": row["sentText"],
                    "expected": expected,
                    "prediction": {"entities": predicted},
                }
                if ignore_extra:
                    result["matchedGold"] = [
                        {"text": text, "label": label, "count": count}
                        for (text, label), count in matched.items()
                    ]
                    result["missingGold"] = [
                        {"text": text, "label": label, "count": count}
                        for (text, label), count in missing.items()
                    ]
                    result["allGoldCaptured"] = not missing
                rows.append(result)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if ignore_extra:
        metrics = {
            "examples": len(indexed_records),
            "gold_entities": gold_entity_count,
            "captured_gold_entities": captured_gold_count,
            "gold_entity_recall": (
                captured_gold_count / gold_entity_count if gold_entity_count else 1.0
            ),
            "examples_with_all_gold_entities_captured": fully_covered_examples,
            "complete_example_rate": (
                fully_covered_examples / len(indexed_records) if indexed_records else 0.0
            ),
            "valid_json_rate": valid_json / len(indexed_records) if indexed_records else 0.0,
            "scoring": "Exact entity text and label recall; extra predictions are ignored.",
        }
    else:
        metrics = {
            "examples": len(indexed_records),
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "exact_matches": exact,
            "valid_json_rate": valid_json / len(indexed_records) if indexed_records else 0.0,
        }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics
