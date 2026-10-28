"""Evaluate the Qwen3.5 base model on the curated test split."""

import json
import os
from collections import Counter

# Avoid a Windows access violation in Transformers' asynchronous weight loader.
# Official setting: https://huggingface.co/docs/transformers/main/reference/environment_variables
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from ner_evaluation import DATA_DIR, OUTPUT_DIR, parse_entities, read_split


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"

TASK_PROMPT = (
    "Extract named entities from the sentence. Use only these labels: PERSON, "
    "ORGANIZATION, LOCATION. Copy each entity exactly as it appears in the "
    "sentence. Return only JSON in this format: "
    '{"entities": [{"text": "name", "label": "PERSON"}]}. '
    'If there are no entities, return {"entities": []}.'
)


def main():
    source_path = DATA_DIR / "test.json"
    records = read_split("test")
    print(f"Reading {len(records)} examples from {source_path}", flush=True)

    # Official model example: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
    print(f"Loading processor for {MODEL_ID}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Processor loaded. Loading model...", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, device_map="auto")
    print("Model loaded. Generating predictions...", flush=True)

    results = []
    matched_gold_count = 0
    total_gold_count = 0
    fully_covered_examples = 0
    valid_json_count = 0
    for index, row in enumerate(tqdm(records, desc="Baseline", unit="example"), start=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{TASK_PROMPT}\n\nSentence: {row['sentText']}",
                    }
                ],
            },
        ]
        # The base model's tokenizer has the chat template; AutoProcessor does not expose it.
        inputs = processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # isolate the json string values, to capture details for evaluation
        outputs = model.generate(**inputs, max_new_tokens=256)
        prediction = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        json_start = prediction.find("{")
        json_end = prediction.rfind("}")
        if json_start >= 0 and json_end >= json_start:
            json_text = prediction[json_start : json_end + 1]
        else:
            json_text = prediction
        predicted_entities, valid_json = parse_entities(json_text)
        expected = [
            {"text": entity["text"], "label": entity["label"]}
            for entity in row["entityMentions"]
            if entity["label"] in {"PERSON", "ORGANIZATION", "LOCATION"}
        ]
        gold_counts = Counter((entity["text"], entity["label"]) for entity in expected)
        predicted_counts = Counter(
            (entity["text"], entity["label"]) for entity in predicted_entities
        )
        matched_counts = gold_counts & predicted_counts
        missing_counts = gold_counts - predicted_counts
        matched_gold_count += sum(matched_counts.values())
        total_gold_count += sum(gold_counts.values())
        fully_covered_examples += not missing_counts
        valid_json_count += valid_json
        results.append(
            {
                "exampleIndex": index,
                "sentId": row["sentId"],
                "sentText": row["sentText"],
                "expected": expected,
                "predicted": predicted_entities,
                "matchedGold": [
                    {"text": text, "label": label, "count": count}
                    for (text, label), count in matched_counts.items()
                ],
                "missingGold": [
                    {"text": text, "label": label, "count": count}
                    for (text, label), count in missing_counts.items()
                ],
                "allGoldCaptured": not missing_counts,
                "validJson": valid_json,
                "modelOutput": prediction,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline.json"
    with output_path.open("w", encoding="utf-8") as target:
        json.dump(results, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"Saved {len(results)} predictions to {output_path}")

    metrics = {
        "examples": len(records),
        "gold_entities": total_gold_count,
        "captured_gold_entities": matched_gold_count,
        "gold_entity_recall": (
            matched_gold_count / total_gold_count if total_gold_count else 1.0
        ),
        "examples_with_all_gold_entities_captured": fully_covered_examples,
        "complete_example_rate": fully_covered_examples / len(records) if records else 0.0,
        "valid_json_examples": valid_json_count,
        "valid_json_rate": valid_json_count / len(records) if records else 0.0,
        "scoring": "Exact entity text and label recall; extra predictions are ignored.",
    }
    metrics_path = OUTPUT_DIR / "baseline.metrics.json"
    with metrics_path.open("w", encoding="utf-8") as target:
        json.dump(metrics, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"Gold entity recall: {metrics['gold_entity_recall']:.1%}")
    print(f"Complete example rate: {metrics['complete_example_rate']:.1%}")
    print(f"Valid JSON rate: {metrics['valid_json_rate']:.1%}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
