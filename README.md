# Qwen3-8B AIMO3 Fine-Tuning on GPT-OSS-120B-Synthesized Harmony dataset

Full-precision LoRA fine-tuning of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on the [AIMO3 Tool-Integrated Reasoning](https://www.kaggle.com/datasets/jeannkouagou/aimo3-tool-integrated-reasoning) dataset in Harmony format which is synthesized by GPT-OSS-120B, optimized for a single H100 GPU on Kaggle.

## Overview

This notebook fine-tunes Qwen3-8B using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA). It targets math reasoning tasks and uses the ChatML instruction format with Flash Attention 2 for efficient training.

Key design choices:
- **No quantization** — full bfloat16 precision, taking advantage of H100 native bf16 support
- **LoRA adapters only** — ~0.3% of parameters are trained, keeping memory usage manageable
- **Gradient checkpointing** — trades compute for memory to fit longer sequences
- **Chunked evaluation** — prevents OOM during validation via `eval_accumulation_steps`

## Requirements

| Dependency | Notes |
|---|---|
| `torch` | CUDA-enabled build |
| `transformers` | HuggingFace Transformers |
| `peft` | LoRA / PEFT library |
| `datasets` | HuggingFace Datasets |
| `flash-attn` | Must be built from source (`--no-build-isolation`) |
| `pandas` | CSV data loading |

Install Flash Attention first (required before other imports):

```bash
pip install flash-attn --no-build-isolation
```

## Hardware

Designed and tested on a **single NVIDIA H100 80GB** (Kaggle). The configuration may need adjustment for other GPUs:

- **A100 40GB** — reduce `per_device_train_batch_size` to 1, reduce `max_seq_length`
- **A100 80GB** — should work as-is; `tf32=True` will still apply
- **Multi-GPU** — `device_map="auto"` handles this, but `deepspeed` config can be passed via `Config.deepspeed`

## Dataset Format

The notebook reads a CSV file with flexible column naming. Accepted column names:

| Role | Accepted column names |
|---|---|
| Input / Prompt | `problem`, `question`, `input`, `prompt` |
| Output / Response | `solution`, `answer`, `output`, `response`, `completion` |

Columns are automatically detected and renamed. Empty rows are dropped. The data is split into train/validation sets (default: 97.5% / 2.5%).

The AIMO3 dataset used in this notebook is available on Kaggle:
`/kaggle/input/datasets/jeannkouagou/aimo3-tool-integrated-reasoning/data.csv`

## Model

Base model path (Kaggle):
```
/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1
```

For HuggingFace Hub, replace with `Qwen/Qwen3-8B`.

## Configuration

All hyperparameters are controlled via the `Config` dataclass. Key settings:

```python
config = Config(
    model_name        = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1",
    dataset_file      = "data.csv",

    # LoRA
    lora_r            = 16,
    lora_alpha        = 32,
    lora_dropout      = 0.05,
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],

    # Training
    num_train_epochs              = 2,
    per_device_train_batch_size   = 2,
    gradient_accumulation_steps   = 8,   # Effective batch size = 16
    learning_rate                 = 2e-4,
    max_seq_length                = 8192,

    # Evaluation
    eval_steps    = 250,
    save_steps    = 250,
    logging_steps = 10,

    # Precision
    bf16  = True,
    optim = "adamw_torch_fused",
    gradient_checkpointing = True,
)
```

To resume from a checkpoint:

```python
config = Config(
    resume_from_checkpoint = "/path/to/checkpoint-XXXX",
    ...
)
```

## Training Details

| Setting | Value |
|---|---|
| Precision | bfloat16 (full, no quantization) |
| Attention | Flash Attention 2 |
| Optimizer | `adamw_torch_fused` |
| LR scheduler | Cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Gradient checkpointing | Enabled |
| TF32 (tensor cores) | Enabled |

### Prompt Format

Training data is formatted using the ChatML template:

```
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{solution}<|im_end|>
```

## Output

After training, the LoRA adapters are saved to:

```
./qwen_lora_outputs/final_model/
```

This directory contains only the adapter weights, not the full model. To run inference, load the base model and merge the adapters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto")
model = PeftModel.from_pretrained(base_model, "./qwen_lora_outputs/final_model")
tokenizer = AutoTokenizer.from_pretrained("./qwen_lora_outputs/final_model")
```

To merge adapters into the base model weights for standalone deployment:

```python
merged = model.merge_and_unload()
merged.save_pretrained("./qwen3_8b_aimo3_merged")
```

## OOM Mitigations

Several fixes are applied to avoid out-of-memory errors during evaluation:

- `load_best_model_at_end=False` — prevents double model load at the end of training; best checkpoint is tracked by a custom `BestModelCallback` instead
- `per_device_eval_batch_size=1` — minimal eval batch size
- `eval_accumulation_steps=4` — evaluation gradients are accumulated in small chunks
- `model.config.use_cache = False` — disables KV cache during training
- `dataloader_persistent_workers=False` — prevents worker memory accumulation across epochs

## Callbacks

Two custom `TrainerCallback` classes are included:

- **`MetricsCallback`** — prints training loss, learning rate, and validation loss at each logging/eval step with a running summary at the end of training
- **`BestModelCallback`** — tracks the checkpoint path with the lowest `eval_loss` without loading it into memory (replaces `load_best_model_at_end=True`)

## License

Please refer to the [Qwen3 model license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE) and the dataset license on Kaggle before using this work commercially.
