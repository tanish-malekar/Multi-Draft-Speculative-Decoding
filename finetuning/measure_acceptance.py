#!/usr/bin/env python3
"""
measure_acceptance.py — Speculative-decoding acceptance measurement.

Uses a manual full speculative-decoding loop that works despite vocabulary-size
mismatches between drafter and target (e.g. Qwen2.5-1.5B vs 14B cannot be
paired with vLLM's built-in speculative decoding):

  1. Drafter proposes up to K tokens from the current context.
  2. Target scores the proposed tokens with prompt_logprobs.
  3. Accepted tokens plus the target replacement/bonus token are appended.
  4. The loop repeats until max_new_tokens, EOS, or domain stop strings.

Reported metrics:
  acceptance_rate (α):   fraction of draft positions where the draft token
                         matches the target's greedy (argmax) prediction.
  mean_accepted_length:  accepted tokens / speculative steps.
  system_efficiency:     emitted tokens / draft tokens.

Usage:
  # Baseline (base 1.5B drafter):
  python finetuning/measure_acceptance.py \\
      --drafter Qwen/Qwen2.5-1.5B-Instruct \\
      --target  Qwen/Qwen2.5-14B-Instruct \\
      --output_json results/base.json

  # After fine-tuning:
  python finetuning/measure_acceptance.py \\
      --drafter checkpoints/code_50k/final \\
      --target  Qwen/Qwen2.5-14B-Instruct \\
      --output_json results/code_50k.json

  # Compare two result files:
  python finetuning/measure_acceptance.py \\
      --compare results/base.json results/code_50k.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

DOMAINS = ["code", "math", "translation"]


# ─────────────────────────────────────────────────────────
# vLLM helpers
# ─────────────────────────────────────────────────────────

def _load_llm(model: str, dtype: str, gpu_util: float, max_model_len: int) -> Any:
    from vllm import LLM  # type: ignore[import]
    return LLM(
        model=model,
        dtype=dtype,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
        disable_log_stats=True,
    )


def _unload_llm(llm: Any) -> None:
    """Release GPU memory used by a vLLM LLM instance."""
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────
# Manual speculative decoding loop
# ─────────────────────────────────────────────────────────

def _stop_sequences_for_domain(domain: str) -> list[str]:
    """Match dataset-generation.py stop sequences for teacher generation."""
    if domain == "translation":
        return [
            "\n\n",
            "\nTranslation:",
            "\nNote:",
            "\nTo stop",
            "\nTo provide",
            "\nAs per",
            "\nTranslation complete",
            "\nStopping here",
        ]
    if domain == "code":
        return [
            "\ndef ",
            "\n    def ",
            "\nclass ",
            "\n    class ",
            "\nimport ",
            "\nfrom ",
            "\n```",
            "```",
            "\n# Continue",
            "\n# Add",
        ]
    return []


@dataclass
class SpecLoopState:
    domain: str
    prompt: str
    target_ids: list[int]
    drafter_ids: list[int]
    generated_text: str = ""
    generated_target_ids: list[int] | None = None
    num_accepted: int = 0
    num_draft: int = 0
    num_emitted: int = 0
    num_steps: int = 0
    finished: bool = False
    finish_reason: str = ""

    def __post_init__(self) -> None:
        if self.generated_target_ids is None:
            self.generated_target_ids = []


def _eos_token_ids(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(t) for t in eos if t is not None}
    return {int(eos)}


def _best_token(logprobs: Any) -> int | None:
    if not logprobs:
        return None
    return int(max(logprobs, key=lambda t: logprobs[t].logprob))


def _contains_stop(text: str, stop: list[str]) -> bool:
    return any(s and s in text for s in stop)


def _refresh_drafter_context(state: SpecLoopState, drafter_tokenizer: Any) -> None:
    state.drafter_ids = drafter_tokenizer.encode(state.prompt + state.generated_text)


def _append_target_tokens(
    state: SpecLoopState,
    token_ids: list[int],
    target_tokenizer: Any,
    drafter_tokenizer: Any,
) -> None:
    if not token_ids:
        return
    state.target_ids.extend(token_ids)
    assert state.generated_target_ids is not None
    state.generated_target_ids.extend(token_ids)
    state.generated_text += target_tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
    )
    _refresh_drafter_context(state, drafter_tokenizer)


def _state_result(state: SpecLoopState) -> dict[str, Any]:
    return {
        "num_accepted_tokens": state.num_accepted,
        "num_draft_tokens": state.num_draft,
        "num_emitted_tokens": state.num_emitted,
        "num_generated_tokens": len(state.generated_target_ids or []),
        "num_steps": state.num_steps,
        "finish_reason": state.finish_reason or "unknown",
    }


def _run_speculative_loop(
    drafter_model: str,
    target_model: str,
    prompts_by_domain: dict[str, list[str]],
    domains: list[str],
    K: int,
    max_new_tokens: int,
    dtype: str,
    drafter_gpu_util: float,
    target_gpu_util: float,
    max_model_len: int,
) -> list[dict[str, Any]]:
    """
    Run a full greedy speculative-decoding loop.

    Each iteration drafts up to K tokens, verifies the draft with the target,
    appends the accepted prefix plus the target replacement/bonus token, and
    repeats until max_new_tokens, EOS, or the domain stop strings are reached.
    """
    from vllm import SamplingParams  # type: ignore[import]

    max_context_len = max_model_len - K - 1
    if max_context_len <= 0:
        raise ValueError(
            f"max_model_len={max_model_len} is too small for K={K}; "
            "increase --max_model_len or lower --num_speculative_tokens."
        )

    print("  Loading target and drafter for manual speculative loop...")
    target_llm: Any | None = None
    drafter_llm: Any | None = None

    results: list[dict[str, Any]] = []
    try:
        target_llm = _load_llm(target_model, dtype, target_gpu_util, max_model_len)
        drafter_llm = _load_llm(drafter_model, dtype, drafter_gpu_util, max_model_len)
        target_tokenizer = target_llm.get_tokenizer()
        drafter_tokenizer = drafter_llm.get_tokenizer()
        target_eos_ids = _eos_token_ids(target_tokenizer)

        for domain in domains:
            stop = _stop_sequences_for_domain(domain)
            states = [
                SpecLoopState(
                    domain=domain,
                    prompt=prompt,
                    target_ids=target_tokenizer.encode(prompt),
                    drafter_ids=drafter_tokenizer.encode(prompt),
                )
                for prompt in prompts_by_domain[domain]
            ]
            print(
                f"  [{domain}] Running full loop on {len(states):,} prompts "
                f"with {len(stop)} stop sequences."
            )

            round_idx = 0
            while True:
                active = [
                    s for s in states
                    if not s.finished
                    and len(s.generated_target_ids or []) < max_new_tokens
                ]
                if not active:
                    break

                round_idx += 1
                if round_idx == 1 or round_idx % 10 == 0:
                    print(
                        f"  [{domain}] round {round_idx}: "
                        f"{len(active):,} active prompts"
                    )

                draft_inputs = [
                    {"prompt_token_ids": s.drafter_ids[-max_context_len:]}
                    for s in active
                ]
                draft_sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=K,
                    stop=stop,
                    ignore_eos=False,
                )
                draft_outputs = drafter_llm.generate(draft_inputs, draft_sp)

                target_inputs: list[dict[str, list[int]]] = []
                target_context_lens: list[int] = []
                draft_ids_list: list[list[int]] = []
                drafter_stopped: list[bool] = []

                for state, out in zip(active, draft_outputs):
                    remaining = max_new_tokens - len(state.generated_target_ids or [])
                    choice = out.outputs[0]
                    d_ids = list(choice.token_ids[: min(K, remaining)])
                    finish_reason = str(getattr(choice, "finish_reason", "") or "")
                    stopped = finish_reason not in {"", "length", "None"}

                    if not d_ids:
                        state.finished = True
                        state.finish_reason = (
                            f"drafter_{finish_reason}" if finish_reason else "drafter_stop"
                        )
                        continue

                    target_context = state.target_ids[-max_context_len:]
                    target_inputs.append({"prompt_token_ids": target_context + d_ids})
                    target_context_lens.append(len(target_context))
                    draft_ids_list.append(d_ids)
                    drafter_stopped.append(stopped)

                if not target_inputs:
                    continue

                target_sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    prompt_logprobs=1,
                    stop=stop,
                    ignore_eos=False,
                )
                target_outputs = target_llm.generate(target_inputs, target_sp)

                scored_states = [s for s in active if not s.finished]
                for state, d_ids, prompt_len, stopped, out in zip(
                    scored_states,
                    draft_ids_list,
                    target_context_lens,
                    drafter_stopped,
                    target_outputs,
                ):
                    state.num_steps += 1
                    state.num_draft += len(d_ids)
                    plp = out.prompt_logprobs

                    accepted = 0
                    replacement_tok: int | None = None
                    for j, d_tok in enumerate(d_ids):
                        pos = prompt_len + j
                        if plp is None or pos >= len(plp) or plp[pos] is None:
                            replacement_tok = None
                            break
                        best_tok = _best_token(plp[pos])
                        if best_tok is None:
                            replacement_tok = None
                            break
                        if d_tok == best_tok:
                            accepted += 1
                            continue
                        replacement_tok = best_tok
                        break

                    accepted_ids = d_ids[:accepted]
                    if accepted_ids:
                        _append_target_tokens(
                            state,
                            accepted_ids,
                            target_tokenizer,
                            drafter_tokenizer,
                        )
                        state.num_accepted += accepted
                        state.num_emitted += accepted

                    if _contains_stop(state.generated_text, stop):
                        state.finished = True
                        state.finish_reason = "stop"
                        continue
                    if accepted_ids and accepted_ids[-1] in target_eos_ids:
                        state.finished = True
                        state.finish_reason = "eos"
                        continue
                    if len(state.generated_target_ids or []) >= max_new_tokens:
                        state.finished = True
                        state.finish_reason = "length"
                        continue

                    if accepted == len(d_ids):
                        if stopped:
                            state.finished = True
                            state.finish_reason = "drafter_stop"
                            continue

                        bonus_ids = list(out.outputs[0].token_ids[:1])
                        if not bonus_ids:
                            state.finished = True
                            state.finish_reason = (
                                str(getattr(out.outputs[0], "finish_reason", "") or "target_stop")
                            )
                            continue
                        _append_target_tokens(
                            state,
                            bonus_ids,
                            target_tokenizer,
                            drafter_tokenizer,
                        )
                        state.num_emitted += len(bonus_ids)
                        if bonus_ids[-1] in target_eos_ids:
                            state.finished = True
                            state.finish_reason = "eos"
                        elif _contains_stop(state.generated_text, stop):
                            state.finished = True
                            state.finish_reason = "stop"
                    else:
                        if replacement_tok is None:
                            state.finished = True
                            state.finish_reason = "missing_target_logprob"
                            continue
                        _append_target_tokens(
                            state,
                            [replacement_tok],
                            target_tokenizer,
                            drafter_tokenizer,
                        )
                        state.num_emitted += 1
                        if replacement_tok in target_eos_ids:
                            state.finished = True
                            state.finish_reason = "eos"
                        elif _contains_stop(state.generated_text, stop):
                            state.finished = True
                            state.finish_reason = "stop"

            for state in states:
                if not state.finished:
                    state.finish_reason = "length"
                results.append(_state_result(state))
    finally:
        if drafter_llm is not None:
            _unload_llm(drafter_llm)
        if target_llm is not None:
            _unload_llm(target_llm)

    return results


# ─────────────────────────────────────────────────────────
# Eval data loading
# ─────────────────────────────────────────────────────────

def load_eval_prompts(
    eval_dir: str, domain: str, max_eval: int, seed: int
) -> list[str]:
    path = os.path.join(eval_dir, f"{domain}_eval.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Eval split not found: {path}\n"
            "Run prepare_eval_split.py first."
        )
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompts.append(rec["prompt"])

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:max_eval]


# ─────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────

def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  N/A "


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "  N/A "
    d = b - a
    return f"{d:+.4f}"


def compare_and_print(base_path: str, ft_path: str) -> None:
    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(ft_path, encoding="utf-8") as f:
        ft = json.load(f)

    print(f"\n{'='*80}")
    print(f"  Comparison")
    print(f"  Base drafter:      {base.get('drafter', '?')}")
    print(f"  Finetuned drafter: {ft.get('drafter', '?')}")
    print(f"  Target:            {base.get('target', ft.get('target', '?'))}")
    print(f"  K (spec tokens):   {base.get('K', '?')}")
    print(f"{'='*80}")

    header = (
        f"  {'Domain':<14}"
        f"  {'α base':>8} {'α ft':>8} {'Δα':>8}"
        f"  {'len base':>9} {'len ft':>7} {'Δlen':>7}"
        f"  {'eff base':>9} {'eff ft':>7} {'Δeff':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_domains = sorted(
        set(list(base.get("per_domain", {}).keys()) + list(ft.get("per_domain", {}).keys()))
    )
    for domain in all_domains:
        bd = base.get("per_domain", {}).get(domain, {})
        fd = ft.get("per_domain", {}).get(domain, {})
        ba, fa = bd.get("acceptance_rate"), fd.get("acceptance_rate")
        bl, fl = bd.get("mean_accepted_length"), fd.get("mean_accepted_length")
        be, fe = bd.get("system_efficiency"), fd.get("system_efficiency")
        print(
            f"  {domain:<14}"
            f"  {_fmt(ba):>8} {_fmt(fa):>8} {_delta(ba, fa):>8}"
            f"  {_fmt(bl):>9} {_fmt(fl):>7} {_delta(bl, fl):>7}"
            f"  {_fmt(be):>9} {_fmt(fe):>7} {_delta(be, fe):>7}"
        )

    ov_b = base.get("overall", {})
    ov_f = ft.get("overall", {})
    ba, fa = ov_b.get("acceptance_rate"), ov_f.get("acceptance_rate")
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'OVERALL':<14}"
        f"  {_fmt(ba):>8} {_fmt(fa):>8} {_delta(ba, fa):>8}"
    )
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure speculative-decoding acceptance rate using a manual full "
            "drafter→target loop. Works with mismatched vocabulary sizes."
        )
    )
    parser.add_argument(
        "--drafter", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model name or path to finetuned checkpoint/final.",
    )
    parser.add_argument(
        "--target", type=str, default="Qwen/Qwen2.5-14B-Instruct",
    )
    parser.add_argument(
        "--eval_dir", type=str, default="datasets/eval_split",
        help="Directory containing <domain>_eval.jsonl files.",
    )
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS),
    )
    parser.add_argument(
        "--num_speculative_tokens", type=int, default=5,
        help="K: number of tokens the drafter proposes per step.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=160,
        help="Maximum generated tokens per prompt in the full speculative loop.",
    )
    parser.add_argument(
        "--max_eval_per_domain", type=int, default=500,
        help="Cap on prompts per domain (shuffled; keeps eval time reasonable).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.70,
        help="Target-model vLLM GPU fraction. Lowered by default so drafter can stay loaded.",
    )
    parser.add_argument(
        "--drafter_gpu_memory_utilization",
        type=float,
        default=0.20,
        help="Drafter-model vLLM GPU fraction for the manual full loop.",
    )
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--output_json", type=str, default="results/acceptance.json",
        help="Where to write the JSON results.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASE_JSON", "FT_JSON"),
        help="Print a diff table between two result files and exit.",
    )
    args = parser.parse_args()

    if args.compare:
        compare_and_print(args.compare[0], args.compare[1])
        return

    K = args.num_speculative_tokens

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Speculative-Decoding Acceptance Measurement             ║
║  Target:   {args.target:<45}║
║  Drafter:  {args.drafter:<45}║
║  K={K}  max_new_tokens={args.max_new_tokens:<4}  max_model_len={args.max_model_len:<6} ║
╚══════════════════════════════════════════════════════════╝
""")

    # ── Collect all prompts across domains ──────────────────────────────────────
    all_prompts: list[str] = []
    prompts_by_domain: dict[str, list[str]] = {}
    domain_slices: dict[str, tuple[int, int]] = {}

    for domain in args.domains:
        prompts = load_eval_prompts(
            args.eval_dir, domain, args.max_eval_per_domain, args.seed
        )
        prompts_by_domain[domain] = prompts
        start = len(all_prompts)
        all_prompts.extend(prompts)
        domain_slices[domain] = (start, len(all_prompts))
        print(f"  [{domain}] {len(prompts):,} eval prompts loaded.")

    print(f"\n  Total prompts: {len(all_prompts):,}")

    # ── Full speculative loop ───────────────────────────────────────────────────
    print("\n  ── Full speculative loop ───────────────────────────────────────────────")
    t0 = time.time()
    per_prompt = _run_speculative_loop(
        drafter_model=args.drafter,
        target_model=args.target,
        prompts_by_domain=prompts_by_domain,
        domains=args.domains,
        K=K,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        drafter_gpu_util=args.drafter_gpu_memory_utilization,
        target_gpu_util=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    print(f"  Full loop runtime: {time.time() - t0:.1f}s")

    # ── Aggregate per domain ────────────────────────────────────────────────────
    print()
    per_domain: dict[str, dict[str, Any]] = {}
    for domain in args.domains:
        start, end = domain_slices[domain]
        slice_results = per_prompt[start:end]
        num_accepted = sum(r["num_accepted_tokens"] for r in slice_results)
        num_draft = sum(r["num_draft_tokens"] for r in slice_results)
        num_emitted = sum(r["num_emitted_tokens"] for r in slice_results)
        num_generated = sum(r["num_generated_tokens"] for r in slice_results)
        num_steps = sum(r["num_steps"] for r in slice_results)
        finish_reasons: dict[str, int] = {}
        for r in slice_results:
            reason = str(r.get("finish_reason", "unknown"))
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        n_prompts = end - start

        if num_draft == 0:
            result: dict[str, Any] = {
                "acceptance_rate": None,
                "mean_accepted_length": None,
                "system_efficiency": None,
                "num_accepted_tokens": 0,
                "num_draft_tokens": 0,
                "num_emitted_tokens": 0,
                "num_generated_tokens": num_generated,
                "num_steps": num_steps,
                "num_prompts": n_prompts,
                "finish_reasons": finish_reasons,
            }
        else:
            alpha = num_accepted / num_draft
            result = {
                "acceptance_rate": round(alpha, 4),
                "mean_accepted_length": (
                    round(num_accepted / num_steps, 4) if num_steps else None
                ),
                "system_efficiency": round(num_emitted / num_draft, 4),
                "num_accepted_tokens": num_accepted,
                "num_draft_tokens": num_draft,
                "num_emitted_tokens": num_emitted,
                "num_generated_tokens": num_generated,
                "num_steps": num_steps,
                "num_prompts": n_prompts,
                "finish_reasons": finish_reasons,
            }

        per_domain[domain] = result
        print(
            f"  [{domain}] "
            f"α={_fmt(result.get('acceptance_rate'))}  "
            f"mean_accepted_len={_fmt(result.get('mean_accepted_length'))}  "
            f"system_efficiency={_fmt(result.get('system_efficiency'))}  "
            f"({n_prompts:,} prompts)"
        )

    # ── Overall aggregation ─────────────────────────────────────────────────────
    total_acc = sum(r.get("num_accepted_tokens", 0) or 0 for r in per_domain.values())
    total_dft = sum(r.get("num_draft_tokens", 0) or 0 for r in per_domain.values())
    total_emi = sum(r.get("num_emitted_tokens", 0) or 0 for r in per_domain.values())
    total_gen = sum(r.get("num_generated_tokens", 0) or 0 for r in per_domain.values())
    total_steps = sum(r.get("num_steps", 0) or 0 for r in per_domain.values())

    if total_dft > 0:
        oa = total_acc / total_dft
        overall: dict[str, Any] = {
            "acceptance_rate": round(oa, 4),
            "mean_accepted_length": (
                round(total_acc / total_steps, 4) if total_steps else None
            ),
            "system_efficiency": round(total_emi / total_dft, 4),
            "num_generated_tokens": total_gen,
            "num_steps": total_steps,
        }
    else:
        overall = {
            "acceptance_rate": None,
            "mean_accepted_length": None,
            "system_efficiency": None,
            "num_generated_tokens": total_gen,
            "num_steps": total_steps,
        }

    # ── Write output ────────────────────────────────────────────────────────────
    output = {
        "drafter": args.drafter,
        "target": args.target,
        "K": K,
        "max_new_tokens": args.max_new_tokens,
        "mode": "manual_full_speculative_loop",
        "generation_stop_sequences": {
            domain: _stop_sequences_for_domain(domain) for domain in args.domains
        },
        "per_domain": per_domain,
        "overall": overall,
    }
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Results written to: {args.output_json}")

    # ── Summary table ───────────────────────────────────────────────────────────
    print(f"\n  {'Domain':<14}  {'α':>8}  {'mean len':>10}  {'sys eff':>10}")
    print("  " + "-" * 48)
    for domain, r in per_domain.items():
        print(
            f"  {domain:<14}  {_fmt(r.get('acceptance_rate')):>8}"
            f"  {_fmt(r.get('mean_accepted_length')):>10}"
            f"  {_fmt(r.get('system_efficiency')):>10}"
        )
    if overall.get("acceptance_rate") is not None:
        print("  " + "-" * 48)
        print(f"  {'OVERALL':<14}  {_fmt(overall['acceptance_rate']):>8}")

    print(
        "\n  To compare against another run:\n"
        f"  python finetuning/measure_acceptance.py "
        f"--compare <other.json> {args.output_json}"
    )


if __name__ == "__main__":
    main()
