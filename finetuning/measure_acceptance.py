#!/usr/bin/env python3
"""
measure_acceptance.py — Speculative-decoding acceptance measurement via vLLM.

Loads the target (14B) and drafter (1.5B or finetuned) into a single vLLM
engine configured for speculative decoding, runs generation on held-out eval
prompts, and captures vLLM's native SpecDecodeWorkerMetrics.

Per-domain runs are kept separate so each domain gets its own acceptance stats.

Reported metrics (standard speculative-decoding terminology):
  acceptance_rate (α):      fraction of draft tokens the target accepted.
  mean_accepted_length:     α × K — average accepted run per drafter call.
  system_efficiency:        emitted tokens / draft tokens ≈ (K·α + 1) / K
                            directly measures effective speedup over AR decoding.

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
import json
import os
import random
import time
from typing import Any

DOMAINS = ["code", "math", "translation"]


# ─────────────────────────────────────────────────────────
# Spec-decode metrics capture
# ─────────────────────────────────────────────────────────

class SpecMetricsCapture:
    """Accumulates SpecDecodeWorkerMetrics emitted by the vLLM stat logger."""

    def __init__(self) -> None:
        self.num_accepted: int = 0
        self.num_draft: int = 0
        self.num_emitted: int = 0
        # Snapshots taken before each domain run so we can compute deltas.
        self._snap_accepted: int = 0
        self._snap_draft: int = 0
        self._snap_emitted: int = 0

    def snapshot(self) -> None:
        """Record current totals; next call to delta_summary() gives since-snapshot stats."""
        self._snap_accepted = self.num_accepted
        self._snap_draft = self.num_draft
        self._snap_emitted = self.num_emitted

    def update(self, m: Any) -> None:
        self.num_accepted += int(getattr(m, "num_accepted_tokens", 0) or 0)
        self.num_draft += int(getattr(m, "num_draft_tokens", 0) or 0)
        self.num_emitted += int(getattr(m, "num_emitted_tokens", 0) or 0)

    def delta(self) -> tuple[int, int, int]:
        return (
            self.num_accepted - self._snap_accepted,
            self.num_draft - self._snap_draft,
            self.num_emitted - self._snap_emitted,
        )

    def delta_summary(self, K: int) -> dict[str, Any]:
        accepted, draft, emitted = self.delta()
        if draft == 0:
            return {
                "acceptance_rate": None,
                "mean_accepted_length": None,
                "system_efficiency": None,
                "num_accepted_tokens": accepted,
                "num_draft_tokens": draft,
                "num_emitted_tokens": emitted,
                "note": "No draft-token counts received from vLLM — see fallback note below.",
            }
        alpha = accepted / draft
        # system_efficiency = emitted / draft tokens; theoretically (K*alpha+1)/K for greedy
        sys_eff = emitted / draft if draft else None
        return {
            "acceptance_rate": round(alpha, 4),
            "mean_accepted_length": round(alpha * K, 4),
            "system_efficiency": round(sys_eff, 4) if sys_eff is not None else None,
            "num_accepted_tokens": accepted,
            "num_draft_tokens": draft,
            "num_emitted_tokens": emitted,
        }


def _try_inject_stat_logger(llm: Any, capture: SpecMetricsCapture) -> bool:
    """
    Inject a custom StatLoggerBase into the vLLM engine to capture
    SpecDecodeWorkerMetrics on every logging tick.

    Returns True if injection succeeded.
    """
    try:
        from vllm.engine.metrics import StatLoggerBase  # type: ignore[import]

        cap = capture

        class _Hook(StatLoggerBase):
            def info(self, type_: str, obj: Any) -> None:  # noqa: A002
                pass

            def log(self, stats: Any) -> None:
                m = getattr(stats, "spec_decode_metrics", None)
                if m is not None:
                    cap.update(m)

        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return False
        loggers = getattr(engine, "stat_loggers", None)
        if isinstance(loggers, dict):
            loggers["_capture"] = _Hook(local_interval=0, logger=None)  # type: ignore[call-arg]
            return True
        return False
    except Exception as e:
        print(f"  ⚠  Stat-logger injection failed ({e}).")
        print("     Trying direct worker-counter fallback instead.")
        return False


def _try_hook_logger_simple(llm: Any, capture: SpecMetricsCapture) -> bool:
    """
    Simpler injection that doesn't need StatLoggerBase constructor args.
    Tries to monkey-patch the log method of an existing logger.
    """
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return False
        loggers = getattr(engine, "stat_loggers", None)
        if not isinstance(loggers, dict):
            return False

        cap = capture

        class _Hook:
            def info(self, *a: Any, **kw: Any) -> None:
                pass

            def log(self, stats: Any) -> None:
                m = getattr(stats, "spec_decode_metrics", None)
                if m is not None:
                    cap.update(m)

        loggers["_capture"] = _Hook()
        return True
    except Exception as e:
        print(f"  ⚠  Simple logger injection also failed ({e}).")
        return False


def _try_read_worker_counters(llm: Any, capture: SpecMetricsCapture) -> bool:
    """
    Fallback: read spec-decode counters directly from the driver worker
    after generation completes.
    """
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return False
        driver_worker = getattr(engine, "driver_worker", None)
        if driver_worker is None:
            return False
        sd_worker = getattr(driver_worker, "spec_decode_worker", None)
        if sd_worker is None:
            return False
        # Try common attribute paths used across vLLM versions
        for attr in ("metrics_collector", "metrics", "_metrics"):
            m = getattr(sd_worker, attr, None)
            if m is not None:
                capture.update(m)
                return True
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────────────────
# vLLM engine construction
# ─────────────────────────────────────────────────────────

def build_llm(args: Any) -> Any:
    """
    Initialise a vLLM LLM with speculative decoding.
    Tries multiple API forms to handle different vLLM versions.
    """
    from vllm import LLM  # type: ignore[import]

    common_kwargs: dict[str, Any] = dict(
        model=args.target,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        disable_log_stats=False,
    )

    spec_variants = [
        # vLLM ≥ 0.6: speculative_config dict
        {"speculative_config": {
            "model": args.drafter,
            "num_speculative_tokens": args.num_speculative_tokens,
        }},
        # Older vLLM: separate keyword args
        {
            "speculative_model": args.drafter,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
    ]

    last_err: Exception | None = None
    for spec_kw in spec_variants:
        try:
            llm = LLM(**common_kwargs, **spec_kw)
            print(f"  ✓ vLLM loaded with spec kwargs: {list(spec_kw.keys())}")
            return llm
        except Exception as e:
            last_err = e
            print(f"  Tried {list(spec_kw.keys())}: {e}")

    raise RuntimeError(
        "Failed to initialise vLLM with speculative decoding. "
        f"Last error: {last_err}\n"
        "Check vLLM version compatibility and that both model paths exist."
    )


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
# Per-domain measurement
# ─────────────────────────────────────────────────────────

def measure_domain(
    llm: Any,
    capture: SpecMetricsCapture,
    prompts: list[str],
    sp: Any,
    K: int,
    has_logger: bool,
) -> dict[str, Any]:
    capture.snapshot()
    t0 = time.time()
    llm.generate(prompts, sp)
    elapsed = time.time() - t0

    # If logger injection failed, attempt direct counter read after generate()
    if not has_logger:
        _try_read_worker_counters(llm, capture)

    result = capture.delta_summary(K)
    result["num_prompts"] = len(prompts)
    result["total_time_s"] = round(elapsed, 2)
    result["throughput_prompts_per_s"] = round(len(prompts) / max(elapsed, 1e-6), 2)
    return result


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
            "Measure speculative-decoding acceptance rate using vLLM. "
            "Runs the target + drafter together and reports α, mean accepted "
            "length, and system efficiency per domain."
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
        help="Maximum tokens to generate per prompt.",
    )
    parser.add_argument(
        "--max_eval_per_domain", type=int, default=500,
        help="Cap on prompts per domain (shuffled; keeps eval time reasonable).",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
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

    from vllm import SamplingParams  # type: ignore[import]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Speculative-Decoding Acceptance Measurement             ║
║  Target:   {args.target:<45}║
║  Drafter:  {args.drafter:<45}║
║  K={args.num_speculative_tokens}  max_eval/domain={args.max_eval_per_domain:<4}  max_new_tokens={args.max_new_tokens:<6}   ║
╚══════════════════════════════════════════════════════════╝
""")

    # ── Build vLLM engine ──────────────────────────────────────────────────────
    print("  Loading models via vLLM...")
    llm = build_llm(args)

    # ── Inject metrics capture ─────────────────────────────────────────────────
    capture = SpecMetricsCapture()
    has_logger = (
        _try_inject_stat_logger(llm, capture)
        or _try_hook_logger_simple(llm, capture)
    )
    if has_logger:
        print("  ✓ Spec-decode metrics capture injected into vLLM stat logger.")
    else:
        print(
            "  ⚠  Logger injection failed. Will attempt direct worker-counter read "
            "after each domain run (less precise but still informative)."
        )

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    # ── Per-domain evaluation ──────────────────────────────────────────────────
    per_domain: dict[str, dict[str, Any]] = {}
    for domain in args.domains:
        print(f"\n  [{domain}] Loading up to {args.max_eval_per_domain:,} eval prompts...")
        prompts = load_eval_prompts(
            args.eval_dir, domain, args.max_eval_per_domain, args.seed
        )
        print(f"  [{domain}] Running speculative decoding on {len(prompts):,} prompts...")
        result = measure_domain(llm, capture, prompts, sp, args.num_speculative_tokens, has_logger)
        per_domain[domain] = result
        alpha = result.get("acceptance_rate")
        mal = result.get("mean_accepted_length")
        eff = result.get("system_efficiency")
        print(
            f"  [{domain}] α={_fmt(alpha)}  "
            f"mean_accepted_len={_fmt(mal)}  "
            f"system_efficiency={_fmt(eff)}  "
            f"({len(prompts):,} prompts in {result['total_time_s']:.1f}s)"
        )
        if result.get("note"):
            print(f"  [{domain}] ⚠  {result['note']}")

    # ── Overall aggregation ────────────────────────────────────────────────────
    total_acc = sum(r.get("num_accepted_tokens", 0) or 0 for r in per_domain.values())
    total_dft = sum(r.get("num_draft_tokens", 0) or 0 for r in per_domain.values())
    total_emi = sum(r.get("num_emitted_tokens", 0) or 0 for r in per_domain.values())

    K = args.num_speculative_tokens
    if total_dft > 0:
        oa = total_acc / total_dft
        overall: dict[str, Any] = {
            "acceptance_rate": round(oa, 4),
            "mean_accepted_length": round(oa * K, 4),
            "system_efficiency": round(total_emi / total_dft, 4),
        }
    else:
        overall = {
            "acceptance_rate": None,
            "mean_accepted_length": None,
            "system_efficiency": None,
            "note": "No draft-token counts — metrics not available for this vLLM version.",
        }

    # ── Write output ───────────────────────────────────────────────────────────
    output = {
        "drafter": args.drafter,
        "target": args.target,
        "K": K,
        "per_domain": per_domain,
        "overall": overall,
    }
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Results written to: {args.output_json}")

    # ── Summary table ──────────────────────────────────────────────────────────
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
