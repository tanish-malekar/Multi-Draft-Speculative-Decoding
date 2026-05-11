#!/usr/bin/env python3
"""Measure speculative decoding with vLLM's native speculative feature."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import time
from collections import defaultdict
from typing import Any

DOMAINS = ["code", "math", "translation"]
DEFAULT_DRAFTER = "Qwen/Qwen3-0.6B"
DEFAULT_TARGET = "Qwen/Qwen3-8B"


def _stop_sequences_for_domain(domain: str) -> list[str]:
    if domain == "translation":
        return ["<END>", "\n\n", "\nTranslation:", "\nNote:"]
    if domain == "math":
        return ["<END>"]
    if domain == "code":
        return ["<END>", "\ndef ", "\nclass ", "\n```", "```"]
    return []


def _load_spec_llm(
    target_model: str,
    drafter_model: str,
    num_speculative_tokens: int,
    dtype: str,
    gpu_util: float,
    max_model_len: int,
) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": target_model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_util,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
    }
    sig = inspect.signature(LLM.__init__)
    if "speculative_config" in sig.parameters:
        kwargs["speculative_config"] = {
            "model": drafter_model,
            "num_speculative_tokens": num_speculative_tokens,
        }
    else:
        kwargs["speculative_model"] = drafter_model
        kwargs["num_speculative_tokens"] = num_speculative_tokens

    return LLM(**kwargs)


def _unload_llm(llm: Any) -> None:
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def load_eval_prompts(eval_dir: str, domain: str, max_eval: int, seed: int) -> list[str]:
    path = os.path.join(eval_dir, f"{domain}_eval.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval split not found: {path}\nRun prepare_eval_split.py first.")

    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompts.append(rec["prompt"])
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:max_eval]


def clean_fasttext_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def classifier_predict(model_path: str, prompts: list[str]) -> list[str]:
    try:
        import fasttext
    except ImportError as exc:
        raise SystemExit("fasttext is required for --prompt_classifier: pip install fasttext") from exc
    model = fasttext.load_model(model_path)
    labels: list[str] = []
    for prompt in prompts:
        pred, _ = model.predict(clean_fasttext_text(prompt), k=1)
        labels.append(pred[0].replace("__label__", "") if pred else "unknown")
    return labels


def _get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _first_number(output: Any, names: list[str]) -> int | None:
    roots = [output, getattr(output, "metrics", None)]
    if getattr(output, "outputs", None):
        roots.append(output.outputs[0])
        roots.append(getattr(output.outputs[0], "metrics", None))
    for root in roots:
        for name in names:
            value = _get_nested_attr(root, name)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, list):
                return len(value)
    return None


def extract_request_metrics(output: Any) -> dict[str, Any]:
    completion = output.outputs[0]
    generated = len(getattr(completion, "token_ids", []) or [])
    finish_reason = str(getattr(completion, "finish_reason", "unknown"))
    accepted = _first_number(output, [
        "num_accepted_tokens",
        "spec_decode_num_accepted_tokens",
        "speculative_num_accepted_tokens",
        "accepted_token_ids",
        "metrics.num_accepted_tokens",
    ])
    draft = _first_number(output, [
        "num_draft_tokens",
        "spec_decode_num_draft_tokens",
        "speculative_num_draft_tokens",
        "draft_token_ids",
        "metrics.num_draft_tokens",
    ])
    return {
        "num_generated_tokens": generated,
        "num_accepted_tokens": accepted,
        "num_draft_tokens": draft,
        "finish_reason": finish_reason,
    }


def run_vllm_speculative(
    target_model: str,
    drafter_model: str,
    prompts: list[tuple[str, str]],
    num_speculative_tokens: int,
    max_new_tokens: int,
    dtype: str,
    gpu_util: float,
    max_model_len: int,
) -> tuple[list[dict[str, Any]], float]:
    from vllm import SamplingParams

    llm = None
    started = time.time()
    try:
        llm = _load_spec_llm(
            target_model=target_model,
            drafter_model=drafter_model,
            num_speculative_tokens=num_speculative_tokens,
            dtype=dtype,
            gpu_util=gpu_util,
            max_model_len=max_model_len,
        )
        prompt_texts = [p for _, p in prompts]
        stop = sorted({s for domain, _ in prompts for s in _stop_sequences_for_domain(domain)})
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=stop)
        outputs = llm.generate(prompt_texts, sampling)
        elapsed = time.time() - started
        results = []
        for (domain, _prompt), output in zip(prompts, outputs):
            rec = extract_request_metrics(output)
            rec["domain"] = domain
            rec["drafter"] = drafter_model
            results.append(rec)
        return results, elapsed
    finally:
        if llm is not None:
            _unload_llm(llm)


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  N/A "


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "  N/A "
    return f"{b - a:+.4f}"


def compare_and_print(base_path: str, ft_path: str) -> None:
    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(ft_path, encoding="utf-8") as f:
        ft = json.load(f)
    print(f"\n{'='*76}")
    print("  Comparison")
    print(f"  Base drafter:      {base.get('drafter', '?')}")
    print(f"  Finetuned drafter: {ft.get('drafter', '?')}")
    print(f"  Target:            {base.get('target', ft.get('target', '?'))}")
    print(f"{'='*76}")
    header = f"  {'Domain':<14} {'alpha base':>10} {'alpha ft':>10} {'delta':>10} {'tok/s base':>11} {'tok/s ft':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    all_domains = sorted(set(base.get("per_domain", {})) | set(ft.get("per_domain", {})))
    for domain in all_domains:
        bd = base.get("per_domain", {}).get(domain, {})
        fd = ft.get("per_domain", {}).get(domain, {})
        print(
            f"  {domain:<14} {_fmt(bd.get('acceptance_rate')):>10}"
            f" {_fmt(fd.get('acceptance_rate')):>10}"
            f" {_delta(bd.get('acceptance_rate'), fd.get('acceptance_rate')):>10}"
            f" {_fmt(bd.get('tokens_per_second')):>11}"
            f" {_fmt(fd.get('tokens_per_second')):>10}"
        )
    print(f"{'='*76}\n")


def aggregate(results: list[dict[str, Any]], elapsed_by_drafter: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    per_domain: dict[str, dict[str, Any]] = {}
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in results:
        by_domain[rec["domain"]].append(rec)

    for domain, rows in by_domain.items():
        generated = sum(r["num_generated_tokens"] for r in rows)
        accepted_values = [r["num_accepted_tokens"] for r in rows if r.get("num_accepted_tokens") is not None]
        draft_values = [r["num_draft_tokens"] for r in rows if r.get("num_draft_tokens") is not None]
        accepted = sum(accepted_values) if accepted_values else None
        draft = sum(draft_values) if draft_values else None
        finish_reasons: dict[str, int] = {}
        for r in rows:
            reason = str(r.get("finish_reason", "unknown"))
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        elapsed = sum(elapsed_by_drafter.values())
        per_domain[domain] = {
            "acceptance_rate": round(accepted / draft, 4) if accepted is not None and draft else None,
            "mean_accepted_length": round(accepted / len(rows), 4) if accepted is not None and rows else None,
            "system_efficiency": round(generated / draft, 4) if draft else None,
            "num_accepted_tokens": accepted,
            "num_draft_tokens": draft,
            "num_generated_tokens": generated,
            "num_prompts": len(rows),
            "tokens_per_second": round(generated / elapsed, 4) if elapsed > 0 else None,
            "finish_reasons": finish_reasons,
        }

    total_generated = sum(r["num_generated_tokens"] for r in results)
    accepted_values = [r["num_accepted_tokens"] for r in results if r.get("num_accepted_tokens") is not None]
    draft_values = [r["num_draft_tokens"] for r in results if r.get("num_draft_tokens") is not None]
    total_accepted = sum(accepted_values) if accepted_values else None
    total_draft = sum(draft_values) if draft_values else None
    total_elapsed = sum(elapsed_by_drafter.values())
    overall = {
        "acceptance_rate": round(total_accepted / total_draft, 4) if total_accepted is not None and total_draft else None,
        "mean_accepted_length": round(total_accepted / len(results), 4) if total_accepted is not None and results else None,
        "system_efficiency": round(total_generated / total_draft, 4) if total_draft else None,
        "num_generated_tokens": total_generated,
        "tokens_per_second": round(total_generated / total_elapsed, 4) if total_elapsed > 0 else None,
        "runtime_seconds": round(total_elapsed, 3),
    }
    return per_domain, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure vLLM native speculative decoding metrics")
    parser.add_argument("--drafter", type=str, default=DEFAULT_DRAFTER)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--eval_dir", type=str, default="datasets/eval_split")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--num_speculative_tokens", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--max_eval_per_domain", type=int, default=500)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_json", type=str, default="results/acceptance.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_classifier", type=str, default=None,
                        help="fastText .bin classifier. If set, predicted domain selects the drafter.")
    parser.add_argument("--code_drafter", type=str, default=None)
    parser.add_argument("--math_drafter", type=str, default=None)
    parser.add_argument("--translation_drafter", type=str, default=None)
    parser.add_argument("--draft_map_json", type=str, default=None,
                        help='Optional JSON map like {"code": "...", "math": "...", "translation": "..."}')
    parser.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "FT_JSON"))
    args = parser.parse_args()

    if args.compare:
        compare_and_print(args.compare[0], args.compare[1])
        return

    draft_map = {domain: args.drafter for domain in DOMAINS}
    if args.draft_map_json:
        with open(args.draft_map_json, encoding="utf-8") as f:
            draft_map.update(json.load(f))
    if args.code_drafter:
        draft_map["code"] = args.code_drafter
    if args.math_drafter:
        draft_map["math"] = args.math_drafter
    if args.translation_drafter:
        draft_map["translation"] = args.translation_drafter

    prompts: list[tuple[str, str]] = []
    for domain in args.domains:
        for prompt in load_eval_prompts(args.eval_dir, domain, args.max_eval_per_domain, args.seed):
            prompts.append((domain, prompt))
    print(f"Loaded {len(prompts):,} eval prompts")

    assigned: list[tuple[str, str, str]] = []
    if args.prompt_classifier:
        predicted = classifier_predict(args.prompt_classifier, [p for _, p in prompts])
        for (gold_domain, prompt), pred_domain in zip(prompts, predicted):
            assigned.append((gold_domain, prompt, draft_map.get(pred_domain, args.drafter)))
        print(f"Using prompt classifier for draft selection: {args.prompt_classifier}")
    else:
        assigned = [(domain, prompt, args.drafter) for domain, prompt in prompts]

    by_drafter: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for domain, prompt, drafter in assigned:
        by_drafter[drafter].append((domain, prompt))

    all_results: list[dict[str, Any]] = []
    elapsed_by_drafter: dict[str, float] = {}
    for drafter, group in by_drafter.items():
        print(f"\nRunning vLLM speculative decoding: target={args.target} drafter={drafter} prompts={len(group):,}")
        results, elapsed = run_vllm_speculative(
            target_model=args.target,
            drafter_model=drafter,
            prompts=group,
            num_speculative_tokens=args.num_speculative_tokens,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            gpu_util=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        all_results.extend(results)
        elapsed_by_drafter[drafter] = elapsed
        print(f"  Runtime {elapsed:.1f}s")

    per_domain, overall = aggregate(all_results, elapsed_by_drafter)
    output = {
        "drafter": args.drafter,
        "target": args.target,
        "draft_map": draft_map if args.prompt_classifier else None,
        "prompt_classifier": args.prompt_classifier,
        "K": args.num_speculative_tokens,
        "max_new_tokens": args.max_new_tokens,
        "mode": "vllm_native_speculative_decoding",
        "per_domain": per_domain,
        "overall": overall,
    }
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  {'Domain':<14} {'alpha':>8} {'mean acc':>10} {'gen tok':>10} {'tok/s':>10}")
    print("  " + "-" * 58)
    for domain in args.domains:
        r = per_domain.get(domain, {})
        print(
            f"  {domain:<14} {_fmt(r.get('acceptance_rate')):>8}"
            f" {_fmt(r.get('mean_accepted_length')):>10}"
            f" {str(r.get('num_generated_tokens', 0)):>10}"
            f" {_fmt(r.get('tokens_per_second')):>10}"
        )
    print("  " + "-" * 58)
    print(f"  {'OVERALL':<14} {_fmt(overall.get('acceptance_rate')):>8} {'':>10} {overall.get('num_generated_tokens', 0):>10} {_fmt(overall.get('tokens_per_second')):>10}")
    if overall.get("acceptance_rate") is None:
        print("\n  Note: this vLLM build did not expose accepted/draft token counters on RequestOutput; throughput and generation metrics were still recorded.")
    print(f"\n  Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
