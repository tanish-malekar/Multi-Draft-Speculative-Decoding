#!/usr/bin/env python3
"""
prepare_eval_split.py — One-time held-out split (run before training).

For each domain (code / math / translation), shuffles <domain>_distillation.jsonl
with a fixed seed and holds out --eval_pct percent of rows as evaluation data.

Outputs:
  <output_dir>/eval_split/<domain>_eval.jsonl    (held-out)
  <output_dir>/train_split/<domain>_train.jsonl  (training)

Run once before any finetune_draft.py invocation. Safe to re-run — it
overwrites the split files with the same deterministic result.
"""

from __future__ import annotations

import argparse
import json
import os
import random

DOMAINS = ["code", "math", "translation"]


def split_domain(
    src_path: str,
    eval_dir: str,
    train_dir: str,
    domain: str,
    eval_pct: float,
    seed: int,
) -> None:
    rows: list[str] = []
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)

    rng = random.Random(seed)
    rng.shuffle(rows)

    n_eval = max(1, int(len(rows) * eval_pct / 100))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    eval_path = os.path.join(eval_dir, f"{domain}_eval.jsonl")
    train_path = os.path.join(train_dir, f"{domain}_train.jsonl")

    with open(eval_path, "w", encoding="utf-8") as f:
        f.write("\n".join(eval_rows) + "\n")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_rows) + "\n")

    print(
        f"  [{domain}] total={len(rows):,} → "
        f"eval={len(eval_rows):,} ({eval_pct:.1f}%), "
        f"train={len(train_rows):,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split distillation JSONL files into train / eval splits"
    )
    parser.add_argument(
        "--data_dir", type=str, default="datasets/distillation_data",
        help="Directory containing <domain>_distillation.jsonl files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets",
        help="Root output dir; eval_split/ and train_split/ are created here",
    )
    parser.add_argument(
        "--eval_pct", type=float, default=10.0,
        help="Percentage of rows to hold out for evaluation (0 < eval_pct < 100)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not (0 < args.eval_pct < 100):
        parser.error("--eval_pct must be strictly between 0 and 100")

    eval_dir = os.path.join(args.output_dir, "eval_split")
    train_dir = os.path.join(args.output_dir, "train_split")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    print(f"\nSplitting datasets from: {args.data_dir}")
    print(f"  eval_pct={args.eval_pct:.1f}%  seed={args.seed}\n")

    found = 0
    for domain in DOMAINS:
        src = os.path.join(args.data_dir, f"{domain}_distillation.jsonl")
        if not os.path.exists(src):
            print(f"  [{domain}] SKIP — {src} not found")
            continue
        split_domain(src, eval_dir, train_dir, domain, args.eval_pct, args.seed)
        found += 1

    if found == 0:
        print("No distillation files found. Check --data_dir.")
        raise SystemExit(1)

    print(f"\nEval split:  {eval_dir}/")
    print(f"Train split: {train_dir}/")
    print("\nDone. Re-run with the same args to reproduce the same split.")


if __name__ == "__main__":
    main()
