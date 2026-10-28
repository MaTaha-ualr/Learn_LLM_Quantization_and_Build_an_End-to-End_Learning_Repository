"""Compare the four saved held-out evaluations using gold-entity recall.

JSON files: https://docs.python.org/3/library/json.html
CSV writer: https://docs.python.org/3/library/csv.html#csv.DictWriter
"""

import csv
import json

from ner_evaluation import OUTPUT_DIR, RESULTS_DIR


def main():
    summary = []
    reference = None
    for variant in ("M0", "M1", "M2", "M3"):
        path = OUTPUT_DIR / f"test_{variant}.jsonl"
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        keys = [(row["exampleIndex"], row["sentText"]) for row in rows]
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError(f"{variant} used different test rows")
        with path.with_suffix(".metrics.json").open(encoding="utf-8") as source:
            metrics = json.load(source)
        if metrics["examples"] != len(rows):
            raise ValueError(f"{variant} prediction and metric counts differ")
        summary.append({"variant": variant, **metrics})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    recall = {row["variant"]: row["gold_entity_recall"] for row in summary}
    for after, before in (("M1", "M0"), ("M2", "M0"), ("M3", "M2"), ("M3", "M0")):
        print(
            f"{after} - {before} gold entity recall: "
            f"{recall[after] - recall[before]:+.4f}"
        )
    print(f"Summary saved: {path}")


if __name__ == "__main__":
    main()
