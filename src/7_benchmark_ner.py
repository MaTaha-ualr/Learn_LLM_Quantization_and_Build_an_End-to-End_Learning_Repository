"""Time one model on the same 50 validation examples.

CUDA timing and memory: https://docs.pytorch.org/docs/stable/notes/cuda.html
Peak allocated memory: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html
Progress bars: https://tqdm.github.io/
"""

import argparse
import json
import statistics
import time

import torch
from tqdm import tqdm

from ner_evaluation import (
    MODELS,
    RESULTS_DIR,
    generate,
    load_model,
    read_split,
    sample_records,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODELS, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    records = sample_records(read_split("valid"), 50, 2026)
    print(f"Loading {args.variant} for {len(records)} validation rows", flush=True)
    model, tokenizer = load_model(MODELS[args.variant])
    batches = [
        [row["sentText"] for _, row in records[start : start + 2]]
        for start in range(0, len(records), 2)
    ]
    print(f"Warm-up; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    generate(batches[0], model, tokenizer)
    # https://docs.pytorch.org/docs/stable/notes/cuda.html
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    seconds = []
    with tqdm(total=len(batches) * 3, desc=args.variant, unit="batch") as progress:
        for _ in range(3):
            torch.cuda.synchronize()
            pass_started = time.perf_counter()
            for batch in batches:
                generate(batch, model, tokenizer)
                progress.update()
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - pass_started)

    result = {
        "variant": args.variant,
        "examples": len(records),
        "pass_seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "examples_per_second": len(records) / statistics.median(seconds),
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    path = RESULTS_DIR / f"benchmark_{args.variant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
    print(f"{args.variant}: {result['examples_per_second']:.3f} examples/s")
    print(f"Peak allocated CUDA: {result['peak_allocated_cuda_bytes'] / 1024**2:.1f} MiB")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s; saved: {path}")


if __name__ == "__main__":
    main()
