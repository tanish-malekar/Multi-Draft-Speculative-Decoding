# Drafter Fine-Tuning Workflow

Fine-tune `Qwen2.5-1.5B-Instruct` as a speculative-decoding drafter, then
measure acceptance rate improvement against `Qwen2.5-14B-Instruct` as the target.

All commands are run from the **repo root** (`/workspace/NLP Project/` or wherever
you cloned the project).

---

## Prerequisites

```bash
bash finetuning/setup.sh
```

Installs vLLM 0.19.0, PyTorch (CUDA 12.8), Transformers, PEFT, and Accelerate.
Run once on a fresh pod. Takes ~5 minutes.

You will also need to accept the model terms on Hugging Face and log in:

```bash
huggingface-cli login    # paste your HF token
```

---

## Step 0 — Data split (already done, safe to skip)

The eval/train split was already generated at 10% / 90%:

| Domain      | Train     | Eval   |
|-------------|-----------|--------|
| code        | 80,145    | 8,904  |
| math        | 80,543    | 8,949  |
| translation | 89,720    | 9,968  |

To regenerate (e.g. if you want a different percentage):

```bash
python finetuning/prepare_eval_split.py \
    --data_dir  datasets/distillation_data \
    --output_dir datasets \
    --eval_pct  10 \
    --seed      1337
```

> **Do not change `--seed` between runs** — it determines which rows go to eval vs train.

---

## Step 1 — Measure baseline acceptance (before any training)

This establishes the baseline: how well does the **unmodified** `Qwen2.5-1.5B-Instruct`
draft tokens for the 14B target?

```bash
python finetuning/measure_acceptance.py \
    --drafter Qwen/Qwen2.5-1.5B-Instruct \
    --target  Qwen/Qwen2.5-14B-Instruct \
    --eval_dir datasets/eval_split \
    --domains code math translation \
    --num_speculative_tokens 5 \
    --max_eval_per_domain 500 \
    --output_json results/base.json
```

**What it does:** loads both models into a single vLLM engine with speculative
decoding enabled, generates on 500 prompts per domain, and captures
`SpecDecodeWorkerMetrics` (α, mean accepted length, system efficiency).

**Expected runtime:** ~20–30 min on A100 80GB (both models load together: 14B + 1.5B ≈ 31 GB).

**Expected baseline numbers** (untrained 1.5B vs 14B, K=5):

| Metric | Typical range |
|---|---|
| acceptance\_rate (α) | 0.55 – 0.75 |
| mean\_accepted\_length | 2.8 – 3.8 |
| system\_efficiency | 0.7 – 0.9 |

---

## Step 2 — Fine-tune the drafter

### Train on a single domain (recommended first run)

```bash
tmux new -s finetune    # protects against SSH disconnects

python finetuning/finetune_draft.py \
    --domains      code \
    --num_samples  50000 \
    --train_dir    datasets/train_split \
    --output_dir   checkpoints/code_50k \
    --model_name   Qwen/Qwen2.5-1.5B-Instruct \
    --epochs       1 \
    --batch_size   8 \
    --grad_accum   4 \
    --lr           2e-5 \
    --max_seq_len  1024 \
    --save_steps   500 \
    --seed         42

# Detach: Ctrl+B then D
# Re-attach: tmux attach -t finetune
```

**What it does:** loads 80K code training rows, shuffles, takes 50K, tokenises with
loss masked over prompt tokens, and runs one epoch of full-parameter SFT in bf16.
Saves a checkpoint every 500 gradient steps. The final model lands in
`checkpoints/code_50k/final/`.

**Expected runtime:** ~3–4 hours on A100 80GB.

### If training crashes — resume automatically

Just re-run the **exact same command**. The script detects the latest `checkpoint-*`
directory and resumes from there (optimizer state, LR schedule, and data cursor
are all restored):

```bash
python finetuning/finetune_draft.py \
    --domains      code \
    --num_samples  50000 \
    --output_dir   checkpoints/code_50k \
    --epochs       1 \
    --seed         42
    # same args, no --fresh
```

The script prints `Resuming from checkpoint: checkpoints/code_50k/checkpoint-XXXX`.

> **Do not change `--domains`, `--num_samples`, or `--seed` between the original
> run and a resume.** The script checks a `dataset_manifest.json` and aborts with a
> clear error if these differ, preventing silent data-mismatch bugs.

### Start completely fresh

```bash
python finetuning/finetune_draft.py \
    --domains code --num_samples 50000 \
    --output_dir checkpoints/code_50k \
    --epochs 1 --seed 42 \
    --fresh         # deletes checkpoints/code_50k/ and starts over
```

### Other useful configurations

**All three domains, 30K samples each:**
```bash
python finetuning/finetune_draft.py \
    --domains all \
    --num_samples 30000 \
    --output_dir checkpoints/all_30k \
    --epochs 1 --seed 42
```

**LoRA instead of full fine-tuning** (faster, less memory, weaker alignment):
```bash
python finetuning/finetune_draft.py \
    --domains code --num_samples 50000 \
    --output_dir checkpoints/code_50k_lora \
    --epochs 1 --seed 42 \
    --lora
```

---

## Step 3 — Measure acceptance after fine-tuning

Point `--drafter` at the `final/` subdirectory of your checkpoint:

```bash
python finetuning/measure_acceptance.py \
    --drafter checkpoints/code_50k/final \
    --target  Qwen/Qwen2.5-14B-Instruct \
    --eval_dir datasets/eval_split \
    --domains code math translation \
    --num_speculative_tokens 5 \
    --max_eval_per_domain 500 \
    --output_json results/code_50k.json
```

**Expected runtime:** same as Step 1 (~20–30 min).

---

## Step 4 — Compare baseline vs. fine-tuned

```bash
python finetuning/measure_acceptance.py \
    --compare results/base.json results/code_50k.json
```

**Example output:**

```
================================================================================
  Comparison
  Base drafter:      Qwen/Qwen2.5-1.5B-Instruct
  Finetuned drafter: checkpoints/code_50k/final
  Target:            Qwen/Qwen2.5-14B-Instruct
  K (spec tokens):   5
================================================================================
  Domain          α base    α ft      Δα        len base  len ft   Δlen     eff base  eff ft   Δeff
  -------------------------------------------------------------------------------
  code            0.6210    0.7450  +0.1240      3.1050   3.7250  +0.6200    0.7800   0.8700  +0.0900
  math            0.6580    0.6620  +0.0040      3.2900   3.3100  +0.0200    0.8100   0.8120  +0.0020
  translation     0.7010    0.7030  +0.0020      3.5050   3.5150  +0.0100    0.8600   0.8610  +0.0010
  -------------------------------------------------------------------------------
  OVERALL         0.6600    0.7033  +0.0433
================================================================================
```

The code domain should show a clear lift; math and translation stay roughly flat
because they were not in the training set.

---

## Key flags reference

### `finetune_draft.py`

| Flag | Default | Description |
|---|---|---|
| `--domains` | `all` | `code`, `math`, `translation`, or `all` (space-separated) |
| `--num_samples` | `50000` | Max samples **per domain** (shuffled before truncation) |
| `--train_dir` | `datasets/train_split` | Where `<domain>_train.jsonl` files live |
| `--output_dir` | `checkpoints/drafter` | Checkpoint and final model destination |
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Base drafter to fine-tune |
| `--epochs` | `1` | Training epochs |
| `--batch_size` | `8` | Per-device batch size |
| `--grad_accum` | `4` | Gradient accumulation steps (effective batch = 8 × 4 = 32) |
| `--lr` | `2e-5` | Learning rate |
| `--max_seq_len` | `1024` | Truncate sequences longer than this |
| `--save_steps` | `500` | Checkpoint every N gradient steps |
| `--save_total_limit` | `2` | Keep only the 2 most recent checkpoints |
| `--seed` | `42` | Controls data shuffle order |
| `--lora` | off | Use LoRA instead of full fine-tuning |
| `--fresh` | off | Wipe `output_dir` and start from scratch |

### `measure_acceptance.py`

| Flag | Default | Description |
|---|---|---|
| `--drafter` | `Qwen/Qwen2.5-1.5B-Instruct` | HF name or path to `final/` directory |
| `--target` | `Qwen/Qwen2.5-14B-Instruct` | Target model (stays fixed) |
| `--eval_dir` | `datasets/eval_split` | Where `<domain>_eval.jsonl` files live |
| `--domains` | all three | Which domains to evaluate |
| `--num_speculative_tokens` | `5` | K — draft tokens proposed per step |
| `--max_new_tokens` | `160` | Max tokens generated per prompt |
| `--max_eval_per_domain` | `500` | Prompts per domain (shuffled) |
| `--gpu_memory_utilization` | `0.90` | vLLM GPU fraction |
| `--output_json` | `results/acceptance.json` | Output file |
| `--compare A B` | — | Print diff table between two result JSONs and exit |

---

## Disk and memory budget on A100 80GB

| Component | Size |
|---|---|
| Qwen2.5-14B bf16 (target) | ~28 GB |
| Qwen2.5-1.5B bf16 (drafter) | ~3 GB |
| KV cache + activations (eval) | ~5–8 GB |
| Qwen2.5-1.5B full FT (training) | ~30 GB peak |
| Training data in RAM (50K rows) | ~2 GB |

Training and eval **cannot** run simultaneously — the full FT optimizer state
fills most of the GPU. Run Steps 1 and 3 (eval) without the trainer loaded.

---

## Troubleshooting

**`FileNotFoundError: Training split not found`**
→ Run `prepare_eval_split.py` first (Step 0).

**`Dataset manifest mismatch`**
→ You changed `--domains`, `--num_samples`, or `--seed` from the original run.
  Either restore the original args or pass `--fresh`.

**`Failed to initialise vLLM with speculative decoding`**
→ vLLM version mismatch. Confirm `vllm==0.19.0` is installed (`pip show vllm`).
  Both model paths must exist and be accessible.

**CUDA OOM during training**
→ Reduce `--batch_size 4` or add `--grad_accum 8` to keep effective batch the same.

**CUDA OOM during eval**
→ Reduce `--gpu_memory_utilization 0.85` or `--max_eval_per_domain 200`.

**`acceptance_rate: null` in output JSON**
→ vLLM's internal API for spec-decode metrics changed. The script logs a warning
  and falls back to a direct worker-counter read. If both fail, `null` is written.
  Check the vLLM version and the warning message printed during the run.
