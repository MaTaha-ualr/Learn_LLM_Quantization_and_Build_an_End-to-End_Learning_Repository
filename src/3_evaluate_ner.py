"""Evaluate one M0-M3 model on all curated test rows; extra predictions are ignored.

Model loading: https://huggingface.co/docs/transformers/main_classes/model
Qwen generation: https://huggingface.co/Qwen/Qwen3-0.6B
"""

import argparse
import json
import time

from ner_evaluation import MODELS, OUTPUT_DIR, evaluate, load_model, read_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODELS, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    records = list(enumerate(read_split("test"), start=1))
    print(f"Loading {args.variant} for {len(records)} held-out rows", flush=True)
    model, tokenizer = load_model(MODELS[args.variant])
    path = OUTPUT_DIR / f"test_{args.variant}.jsonl"
    metrics = evaluate(
        records,
        model,
        tokenizer,
        path,
        f"Test {args.variant}",
        ignore_extra=True,
    )
    with path.with_suffix(".metrics.json").open("w", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)
    print(
        f"{args.variant}: {metrics['examples']} rows, "
        f"gold entity recall={metrics['gold_entity_recall']:.1%}, "
        f"complete example rate={metrics['complete_example_rate']:.1%}"
    )
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
