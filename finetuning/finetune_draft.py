#!/usr/bin/env python3
"""Fine-tune a draft model with KL loss against stored target distributions.

Rows are read from datasets/train_split/<domain>_train.jsonl. New dataset rows
contain target_token_ids plus target_top_logprobs from dataset generation.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import shutil
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

DOMAINS = ["code", "math", "translation"]


def resolve_domains(values: list[str]) -> list[str]:
    return list(DOMAINS) if "all" in values else sorted(set(values))


def load_data(
    train_dir: str,
    domains: list[str],
    samples_per_domain: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_rows: list[dict[str, Any]] = []

    for domain in domains:
        path = os.path.join(train_dir, f"{domain}_train.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training split not found: {path}\nRun prepare_eval_split.py first.")

        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["domain"] = rec.get("domain", domain)
                rows.append(rec)

        rng.shuffle(rows)
        selected = rows[:samples_per_domain]
        print(f"  [{domain}] Loaded {len(selected):,} samples")
        all_rows.extend(selected)

    rng.shuffle(all_rows)
    return all_rows


def parse_top_logprobs(row: dict[str, Any]) -> tuple[list[int], list[list[int]], list[list[float]]]:
    token_ids = [int(x) for x in row.get("target_token_ids") or []]
    top_rows = row.get("target_top_logprobs") or []
    top_ids: list[list[int]] = []
    top_lps: list[list[float]] = []

    for entries in top_rows[:len(token_ids)]:
        ids: list[int] = []
        lps: list[float] = []
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                if "token_id" not in item or "logprob" not in item:
                    continue
                ids.append(int(item["token_id"]))
                lps.append(float(item["logprob"]))
        top_ids.append(ids)
        top_lps.append(lps)

    n = min(len(token_ids), len(top_ids), len(top_lps))
    return token_ids[:n], top_ids[:n], top_lps[:n]


class KLDistillationDataset(Dataset):
    """Tokenizes prompts and attaches target top-k distributions by token index."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int, top_k: int):
        self.items: list[dict[str, Any]] = []
        skipped = 0
        no_kl = 0

        for row in rows:
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                skipped += 1
                continue

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids, response_top_ids, response_top_lps = parse_top_logprobs(row)
            if not response_ids:
                response_text = row.get("response")
                if not isinstance(response_text, str) or not response_text.strip():
                    skipped += 1
                    continue
                response_ids = tokenizer.encode(response_text, add_special_tokens=False)
                response_top_ids = []
                response_top_lps = []
                no_kl += 1

            max_response_len = max(0, max_seq_len - len(prompt_ids))
            if max_response_len <= 0:
                skipped += 1
                continue

            response_ids = response_ids[:max_response_len]
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
            seq_top_ids = [[-1] * top_k for _ in input_ids]
            seq_top_lps = [[float("-inf")] * top_k for _ in input_ids]

            for offset, (ids, lps) in enumerate(zip(response_top_ids, response_top_lps)):
                pos = len(prompt_ids) + offset
                if pos >= len(input_ids):
                    break
                k = min(top_k, len(ids), len(lps))
                seq_top_ids[pos][:k] = ids[:k]
                seq_top_lps[pos][:k] = lps[:k]

            self.items.append({
                "input_ids": input_ids,
                "labels": labels,
                "target_top_token_ids": seq_top_ids,
                "target_top_logprobs": seq_top_lps,
            })

        if skipped:
            print(f"  Skipped {skipped:,} unusable/truncated rows")
        if no_kl:
            print(f"  Warning: {no_kl:,} rows had no stored top-logprobs; batches with only these rows use CE fallback")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class KLCollator:
    def __init__(self, pad_token_id: int, top_k: int):
        self.pad_token_id = pad_token_id
        self.top_k = top_k

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        top_ids, top_lps = [], []

        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
            top_ids.append(item["target_top_token_ids"] + [[-1] * self.top_k for _ in range(pad)])
            top_lps.append(item["target_top_logprobs"] + [[float("-inf")] * self.top_k for _ in range(pad)])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_top_token_ids": torch.tensor(top_ids, dtype=torch.long),
            "target_top_logprobs": torch.tensor(top_lps, dtype=torch.float32),
        }


class KLTrainer(Trainer):
    def compute_loss(self, model: Any, inputs: dict[str, torch.Tensor], return_outputs: bool = False, **kwargs: Any):
        top_ids = inputs.pop("target_top_token_ids")
        target_lps = inputs.pop("target_top_logprobs")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].float()
        shift_top_ids = top_ids[:, 1:, :]
        shift_target_lps = target_lps[:, 1:, :]
        valid = shift_top_ids >= 0
        position_mask = valid.any(dim=-1)

        if not bool(position_mask.any()):
            if labels is None:
                loss = logits.sum() * 0.0
            else:
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return (loss, outputs) if return_outputs else loss

        safe_ids = shift_top_ids.clamp_min(0)
        draft_log_probs = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_ids)
        target_probs = torch.where(valid, shift_target_lps.exp(), torch.zeros_like(shift_target_lps))
        mass = target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        target_probs = target_probs / mass
        target_log_probs = torch.where(
            target_probs > 0,
            target_probs.clamp_min(1e-20).log(),
            torch.zeros_like(target_probs),
        )
        per_position_kl = (target_probs * (target_log_probs - draft_log_probs)).sum(dim=-1)
        loss = per_position_kl[position_mask].mean()
        return (loss, outputs) if return_outputs else loss


def find_latest_checkpoint(output_dir: str) -> str | None:
    ckpt_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpt_dirs:
        return None

    def step_num(p: str) -> int:
        try:
            return int(os.path.basename(p).split("-")[1])
        except Exception:
            return -1

    return max(ckpt_dirs, key=step_num)


def main() -> None:
    parser = argparse.ArgumentParser(description="KL fine-tune a Qwen3 draft model from target top-k distributions")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS + ["all"], default=["all"])
    parser.add_argument("--num_samples", type=int, default=50_000, help="Samples per domain")
    parser.add_argument("--train_dir", type=str, default="datasets/train_split")
    parser.add_argument("--output_dir", type=str, default="checkpoints/all_50k_kl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    args.domains = resolve_domains(args.domains)

    if args.lora and importlib.util.find_spec("peft") is None:
        raise SystemExit("peft is required for LoRA: pip install peft")

    if args.fresh and os.path.exists(args.output_dir):
        print(f"  --fresh: removing {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Draft Fine-Tuning with KL Distillation                  ║
║  Draft:    {args.model_name:<45}║
║  Domains:  {", ".join(args.domains):<45}║
║  Samples:  {args.num_samples:,} per domain{' ' * 23}║
║  Top-k:    {args.top_k:<45}║
║  Output:   {args.output_dir:<45}║
╚══════════════════════════════════════════════════════════╝
""")

    rows = load_data(args.train_dir, args.domains, args.num_samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = KLDistillationDataset(rows, tokenizer, args.max_seq_len, args.top_k)
    print(f"  Dataset size after tokenisation: {len(dataset):,}")

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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

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

    trainer = KLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=KLCollator(tokenizer.pad_token_id, args.top_k),
        processing_class=tokenizer,
    )

    last_ckpt = find_latest_checkpoint(args.output_dir)
    print(f"  Resuming from: {last_ckpt}" if last_ckpt else "  Starting fresh.")
    trainer.train(resume_from_checkpoint=last_ckpt)

    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n  Model saved to: {final_path}")


if __name__ == "__main__":
    main()
