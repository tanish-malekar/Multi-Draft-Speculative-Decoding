#!/usr/bin/env python3
"""Evaluate a fastText prompt-domain classifier.

Default test set size is 100 prompts per domain.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Test fastText domain classifier")
    parser.add_argument("--model", type=str, default="checkpoints/domain_classifier/domain_classifier.bin")
    parser.add_argument("--data_dir", type=str, default="datasets/eval_split")
    parser.add_argument("--samples_per_domain", type=int, default=100)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        import fasttext
    except ImportError as exc:
        raise SystemExit("fasttext is required: pip install fasttext") from exc

    model = fasttext.load_model(args.model)
    rng = random.Random(args.seed)

    confusion = {gold: {pred: 0 for pred in args.domains} for gold in args.domains}
    total = correct = 0

    for domain in args.domains:
        path = find_domain_file(args.data_dir, domain, "eval")
        prompts = load_prompts(path)
        rng.shuffle(prompts)
        selected = prompts[:args.samples_per_domain]
        domain_correct = 0

        for prompt in selected:
            labels, probs = model.predict(clean_fasttext_text(prompt), k=1)
            pred = labels[0].replace("__label__", "") if labels else "unknown"
            if pred not in confusion[domain]:
                confusion[domain][pred] = 0
            confusion[domain][pred] += 1
            total += 1
            if pred == domain:
                correct += 1
                domain_correct += 1

        denom = max(1, len(selected))
        print(f"[{domain}] accuracy={domain_correct / denom:.4f} ({domain_correct}/{len(selected)})")

    print(f"\nOverall accuracy={correct / max(1, total):.4f} ({correct}/{total})")
    print("\nConfusion matrix (rows=gold, cols=pred):")
    header = "gold\\pred".ljust(14) + "".join(d.rjust(14) for d in args.domains)
    print(header)
    for gold in args.domains:
        print(gold.ljust(14) + "".join(str(confusion[gold].get(pred, 0)).rjust(14) for pred in args.domains))


if __name__ == "__main__":
    main()
