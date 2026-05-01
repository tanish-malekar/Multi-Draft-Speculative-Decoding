#!/usr/bin/env python3
"""
finetune_draft.py — SFT fine-tuning of Qwen2.5-1.5B-Instruct as a
speculative-decoding drafter aligned to Qwen2.5-14B-Instruct.

Reads from datasets/train_split/<domain>_train.jsonl (produced by prepare_eval_split.py).
Loss is computed on response tokens only; prompt tokens are masked with -100.

No chat template is applied — the distillation dataset was generated from raw
text prompts (vLLM generate() called directly on the prompt string), so the
drafter must learn the same raw prompt→response mapping the target uses.

Resume behaviour:
  On re-launch with the same --output_dir, the script auto-detects the latest
  checkpoint-* directory and calls trainer.train(resume_from_checkpoint=...).
  Pass --fresh to clear the output dir and start from scratch.
  A dataset_manifest.json is saved on first run; resume aborts with a clear
  error if --domains / --num_samples / --seed were changed.

Usage:
  python finetuning/finetune_draft.py \\
      --domains code \\
      --num_samples 50000 \\
      --train_dir datasets/train_split \\
      --output_dir checkpoints/code_50k \\
      --epochs 1 \\
      --save_steps 500

  # Resume after a crash (same command, no --fresh):
  python finetuning/finetune_draft.py \\
      --domains code --num_samples 50000 \\
      --output_dir checkpoints/code_50k --epochs 1

  # Start over:
  python finetuning/finetune_draft.py --fresh ...
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DOMAINS = ["code", "math", "translation"]


# ─────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────

def load_and_sample(
    train_dir: str,
    domains: list[str],
    num_samples_per_domain: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    all_rows: list[dict[str, str]] = []

    for domain in domains:
        path = os.path.join(train_dir, f"{domain}_train.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Training split not found: {path}\n"
                "Run prepare_eval_split.py first."
            )
        rows: list[dict[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append({"prompt": rec["prompt"], "response": rec["response"]})

        rng.shuffle(rows)
        if len(rows) < num_samples_per_domain:
            print(
                f"  [{domain}] Only {len(rows):,} samples available "
                f"(requested {num_samples_per_domain:,}); using all."
            )
        rows = rows[:num_samples_per_domain]
        print(f"  [{domain}] Loaded {len(rows):,} training samples")
        all_rows.extend(rows)

    rng.shuffle(all_rows)
    return all_rows


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class PromptResponseDataset(Dataset):
    """Tokenises prompt+response; labels -100 over prompt tokens."""

    def __init__(self, rows: list[dict[str, str]], tokenizer, max_seq_len: int):
        self.items: list[dict[str, list[int]]] = []
        skipped = 0

        for row in rows:
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            resp_ids = tokenizer.encode(row["response"], add_special_tokens=False)
            ids = prompt_ids + resp_ids

            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]

            prompt_len = min(len(prompt_ids), len(ids))
            if prompt_len >= len(ids):
                skipped += 1
                continue

            labels = [-100] * prompt_len + ids[prompt_len:]
            self.items.append({"input_ids": ids, "labels": labels})

        if skipped:
            print(
                f"  Skipped {skipped:,} rows where response was fully "
                f"truncated at max_seq_len={max_seq_len}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


# ─────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────

class SFTCollator:
    """Right-pad input_ids and labels to the longest sequence in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids_list, labels_list, attn_mask_list = [], [], []

        for item in batch:
            ids = item["input_ids"]
            lbls = item["labels"]
            pad_len = max_len - len(ids)
            input_ids_list.append(ids + [self.pad_token_id] * pad_len)
            labels_list.append(lbls + [-100] * pad_len)
            attn_mask_list.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> str | None:
    pattern = os.path.join(output_dir, "checkpoint-*")
    ckpt_dirs = glob.glob(pattern)
    if not ckpt_dirs:
        return None

    def step_num(p: str) -> int:
        try:
            return int(os.path.basename(p).split("-")[1])
        except Exception:
            return -1

    ckpt_dirs.sort(key=step_num)
    return ckpt_dirs[-1]


# ─────────────────────────────────────────────────────────
# Dataset manifest (guards against silent resume-with-wrong-data)
# ─────────────────────────────────────────────────────────

def _manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "dataset_manifest.json")


def save_manifest(output_dir: str, args: Any) -> None:
    manifest = {
        "model_name": args.model_name,
        "domains": sorted(args.domains),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "train_dir": str(args.train_dir),
    }
    with open(_manifest_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def check_manifest(output_dir: str, args: Any) -> None:
    path = _manifest_path(output_dir)
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as f:
        saved = json.load(f)

    errors: list[str] = []
    if sorted(saved.get("domains", [])) != sorted(args.domains):
        errors.append(f"domains: saved={saved.get('domains')} current={args.domains}")
    if saved.get("num_samples") != args.num_samples:
        errors.append(
            f"num_samples: saved={saved.get('num_samples')} current={args.num_samples}"
        )
    if saved.get("seed") != args.seed:
        errors.append(f"seed: saved={saved.get('seed')} current={args.seed}")

    if errors:
        raise ValueError(
            f"Dataset manifest mismatch in {output_dir}! "
            "Resume would train on different data.\n  "
            + "\n  ".join(errors)
            + "\n\nPass --fresh to start over, or restore the original CLI args."
        )


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning of a Qwen2.5 drafter for speculative decoding"
    )
    parser.add_argument(
        "--domains", nargs="+",
        choices=DOMAINS + ["all"],
        default=["all"],
        help="Which domain(s) to train on. 'all' trains on code+math+translation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=50_000,
        help="Max samples per domain (randomly subsampled with --seed).",
    )
    parser.add_argument(
        "--train_dir", type=str, default="datasets/train_split",
        help="Directory containing <domain>_train.jsonl files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/drafter",
        help="Where to save checkpoints and the final model.",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size.")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps. Effective batch = batch_size × grad_accum.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save a checkpoint every N gradient-update steps.",
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=2,
        help="Keep only the N most recent checkpoints on disk.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora", action="store_true",
        help="Use LoRA instead of full fine-tuning (saves memory, slightly weaker alignment).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete output_dir and start from scratch (ignores any existing checkpoint).",
    )
    args = parser.parse_args()

    # Resolve domains
    args.domains = DOMAINS if "all" in args.domains else sorted(set(args.domains))

    # Handle --fresh
    if args.fresh and os.path.exists(args.output_dir):
        print(f"  --fresh: removing {args.output_dir}")
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    check_manifest(args.output_dir, args)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Drafter Fine-Tuning                                      ║
║  Model:    {args.model_name:<44}║
║  Domains:  {", ".join(args.domains):<44}║
║  Samples:  {args.num_samples:>7,} per domain                             ║
║  Epochs:   {args.epochs:<44}║
║  LR:       {args.lr:<44}║
║  LoRA:     {str(args.lora):<44}║
║  Output:   {args.output_dir:<44}║
╚══════════════════════════════════════════════════════════╝
""")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[1/3] Loading training data...")
    rows = load_and_sample(args.train_dir, args.domains, args.num_samples, args.seed)
    print(f"  Total training rows: {len(rows):,}")
    save_manifest(args.output_dir, args)

    # ── Tokeniser ──────────────────────────────────────────────────────────────
    print("\n[2/3] Loading tokeniser & model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = PromptResponseDataset(rows, tokenizer, args.max_seq_len)
    print(f"  Dataset size after tokenisation: {len(dataset):,}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ── Trainer ────────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=2,
        report_to="none",
    )

    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ── Resume ─────────────────────────────────────────────────────────────────
    print("\n[3/3] Training...")
    last_ckpt = find_latest_checkpoint(args.output_dir)
    if last_ckpt is not None:
        print(f"  Resuming from checkpoint: {last_ckpt}")
    else:
        print("  Starting fresh training run.")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n  ✓ Final model saved to: {final_path}")
    print(f"  Pass --drafter {final_path} to measure_acceptance.py")


if __name__ == "__main__":
    main()
