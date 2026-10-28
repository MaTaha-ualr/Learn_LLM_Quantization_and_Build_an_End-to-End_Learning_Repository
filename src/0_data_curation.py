"""Convert the original JSONL splits to the JSON arrays used by this demo.

JSON reading and writing: https://docs.python.org/3/library/json.html
Path operations: https://docs.python.org/3/library/pathlib.html
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "Data" / "dataset"
DATA_DIR = PROJECT_DIR / "Data" / "curated"
SPLIT_SIZES = {"train": 300, "valid": 50, "test": 50}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, limit in SPLIT_SIZES.items():
        source_path = RAW_DIR / f"{split}.json"
        with source_path.open(encoding="utf-8-sig") as source:
            records = [json.loads(line) for line in source if line.strip()][:limit]

        # Raw sentId values restart within each article, so use a split-prefixed ID.
        records = [
            {
                "sentId": f"{split}-{index}",
                "sentText": row["sentText"],
                "entityMentions": row["entityMentions"],
            }
            for index, row in enumerate(records, start=1)
        ]
        with (DATA_DIR / f"{split}.json").open("w", encoding="utf-8") as target:
            json.dump(records, target, ensure_ascii=False, indent=2)
            target.write("\n")
        print(f"{split}: saved {len(records)} records")


if __name__ == "__main__":
    main()
