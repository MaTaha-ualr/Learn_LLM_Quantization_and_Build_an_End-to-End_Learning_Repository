"""Save M1 or M3 with INT8 weight-only TorchAO quantization.

TorchAO integration: https://huggingface.co/docs/transformers/quantization/torchao
TorchAoConfig: https://huggingface.co/docs/transformers/main_classes/quantization
"""

import argparse
import time

from torchao.quantization import Int8WeightOnlyConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from ner_evaluation import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("M1", "M3"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    #MO base to Quantized M1
    #M2 finetuned merged to Quantized M3
    source = MODELS["M0"] if args.variant == "M1" else MODELS["M2"]
    output_dir = MODELS[args.variant]
    if output_dir.exists():
        raise FileExistsError(output_dir)
    print(
        f"Loading and quantizing {source}; elapsed {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    # https://huggingface.co/docs/transformers/quantization/torchao
    quantization_config = TorchAoConfig(quant_type=Int8WeightOnlyConfig())
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype="auto",
        device_map=0,
        disable_mmap=True,
        quantization_config=quantization_config,
    )
    print(
        f"Saving quantized model; elapsed {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized {args.variant} saved: {output_dir}")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
