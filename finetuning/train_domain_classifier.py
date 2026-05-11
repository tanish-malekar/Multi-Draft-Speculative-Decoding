#!/usr/bin/env python3
"""Train a fastText prompt-domain classifier.

Uses the same JSONL prompt rows produced by dataset generation / split prep.
Default training set size is 500 prompts per domain.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

DOMAINS = ["code", "math", "translation"]


def clean_fasttext_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def find_domain_file(data_dir: str, domain: str, split: str) -> str:
    candidates = [
        os.path.join(data_dir, f"{domain}_{split}.jsonl"),
        os.path.join(data_dir, f"{domain}_distillation.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No JSONL found for {domain}. Tried: " + ", ".join(candidates)
    )


def load_prompts(path: str) -> list[str]:
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)
    return prompts


def write_fasttext_file(
    data_dir: str,
    output_path: str,
    domains: list[str],
    samples_per_domain: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    rows: list[str] = []

    for domain in domains:
        path = find_domain_file(data_dir, domain, "train")
        prompts = load_prompts(path)
        rng.shuffle(prompts)
        selected = prompts[:samples_per_domain]
        counts[domain] = len(selected)
        rows.extend(
            f"__label__{domain} {clean_fasttext_text(prompt)}\n"
            for prompt in selected
        )

    rng.shuffle(rows)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fastText domain classifier")
    parser.add_argument("--data_dir", type=str, default="datasets/train_split")
    parser.add_argument("--output_model", type=str, default="checkpoints/domain_classifier/domain_classifier.bin")
    parser.add_argument("--samples_per_domain", type=int, default=500)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word_ngrams", type=int, default=2)
    args = parser.parse_args()

    try:
        import fasttext
    except ImportError as exc:
        raise SystemExit("fasttext is required: pip install fasttext") from exc

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    train_txt = output_model.with_suffix(".train.txt")

    counts = write_fasttext_file(
        data_dir=args.data_dir,
        output_path=str(train_txt),
        domains=args.domains,
        samples_per_domain=args.samples_per_domain,
        seed=args.seed,
    )

    print("Training rows: " + ", ".join(f"{d}={n}" for d, n in counts.items()))
    model = fasttext.train_supervised(
        input=str(train_txt),
        epoch=args.epoch,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
        loss="softmax",
    )
    model.save_model(str(output_model))
    print(f"Saved classifier to: {output_model}")
    print(f"Training text written to: {train_txt}")


if __name__ == "__main__":
    main()
