# PEFT and LoRA notes

Parameter-Efficient Fine-Tuning (PEFT) adapts a pretrained model by training a small set of additional parameters while keeping the base weights frozen. This can reduce training memory and storage compared with updating and saving every model weight.

## LoRA

Low-Rank Adaptation (LoRA) adds trainable low-rank matrices to selected model layers. The original layer weights remain frozen while the adapter learns an update. The adapter can be saved separately or merged into the base model for standalone inference.

The main LoRA settings used in this repository are:

| Setting | Project value | Meaning |
| --- | ---: | --- |
| `r` | 8 | Rank of the low-rank update. Larger values add trainable parameters. |
| `lora_alpha` | 16 | Scaling factor for the LoRA update. |
| `lora_dropout` | 0.05 | Dropout applied within the adapter during training. |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention projections receiving LoRA adapters. |
| `task_type` | `CAUSAL_LM` | Configures PEFT for causal language modeling. |

These settings are defined in [`src/2_train_ner_lora.py`](src/2_train_ner_lora.py). They are experiment choices, not universal defaults.

## QLoRA

QLoRA combines a frozen 4-bit quantized base model with trainable LoRA adapters. The adapter weights are trained at higher precision while gradients pass through the quantized base model. It can reduce memory needs for fine-tuning large models, though the exact savings depend on the model, quantization setup, hardware, and software versions.

This repository does **not** implement QLoRA. Its training run loads the base model in BF16 and trains LoRA adapters. INT8 weight-only quantization is applied to saved models afterward for inference comparisons.

## When adapters are merged

The merge step combines the learned LoRA update with the frozen base weights to create a standalone checkpoint. This is useful for inference and for applying the repository's post-training INT8 quantization step. The project performs this in [`src/4_merge_ner_lora.py`](src/4_merge_ner_lora.py).

## References

- [PEFT LoRA configuration reference](https://huggingface.co/docs/peft/package_reference/lora)
- [PEFT guide to merging adapter weights](https://huggingface.co/docs/peft/developer_guides/checkpoint#merge-the-weights)
- [TRL supervised fine-tuning trainer](https://huggingface.co/docs/trl/sft_trainer)
- [QLoRA paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
