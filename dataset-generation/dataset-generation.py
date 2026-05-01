#!/usr/bin/env python3
"""
Distillation Dataset Generator — short prompts / short outputs
==============================================================

Generates teacher outputs for three domains:
  1. Python code completion   — 80K default
  2. Math problem solving     — 80K default
  3. Translation             — 40K English→French + 40K French→English by default

Code domain uses CodeSearchNet Python exclusively.
Prompts are domain-specific and output-only. Code prompts use the partial code only; math and translation prompts force compact answers.

Rows are filtered by combined (docstring + code) token length before inference.

Example:
python /workspace/distillation/dataset-generation.py   --samples_per_domain 100   --model_name Qwen/Qwen2.5-14B-Instruct   --batch_size 128  --output_dir /workspace/distillation/distillation_data   --domains all --dtype bfloat16   --gpu_memory_utilization 0.90   --max_model_len 1024   --temperature 0.0   --checkpoint_every 100   --oversample_factor 1.0   --fresh
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import pickle
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


# ═══════════════════════════════════════════════
# 1. DATASET SOURCES
# ═══════════════════════════════════════════════

# Code domain: CodeSearchNet Python only.
# Fields used:
#   func_documentation_string  — the docstring (may be empty)
#   func_code_string           — the full function source
#   func_name                  — used for deduplication / metadata
CODE_SOURCES = [
    {
        "name": "code_search_net",
        "split": "train",
        "config": "python",          # language-specific config in HF datasets
        "content_field": "func_code_string",
        "docstring_field": "func_documentation_string",
        "name_field": "func_name",
        "source_type": "python_function",
        "streaming": True,
    },
]

MATH_SOURCES = [
    {"name": "openai/gsm8k", "split": "train", "config": "main", "prompt_field": "question", "streaming": False},
    {"name": "meta-math/MetaMathQA", "split": "train", "config": None, "prompt_field": "query", "streaming": True},
    {"name": "nvidia/OpenMathInstruct-2", "split": "train_1M", "config": None, "prompt_field": "problem", "streaming": True},
    # Optional fallback. Some environments use hendrycks/competition_math instead.
    {"name": "lighteval/MATH", "split": "train", "config": "all", "prompt_field": "problem", "streaming": False},
]

# Translation prompt sources are parallel corpora. We use the same parallel
# sentence pairs in opposite directions, but deduplicate by direction.
# OPUS-100 has an explicit en-fr config; WMT14 uses fr-en for the same pair.
TRANSLATION_SOURCES_BY_DIRECTION = {
    "en-fr": [
        {"name": "Helsinki-NLP/opus-100", "split": "train", "config": "en-fr", "prompt_field": "translation", "source_key": "en", "target_key": "fr", "streaming": True},
        {"name": "wmt/wmt14", "split": "train", "config": "fr-en", "prompt_field": "translation", "source_key": "en", "target_key": "fr", "streaming": True},
    ],
    "fr-en": [
        {"name": "Helsinki-NLP/opus-100", "split": "train", "config": "en-fr", "prompt_field": "translation", "source_key": "fr", "target_key": "en", "streaming": True},
        {"name": "wmt/wmt14", "split": "train", "config": "fr-en", "prompt_field": "translation", "source_key": "fr", "target_key": "en", "streaming": True},
    ],
}


# ═══════════════════════════════════════════════
# 2. PROMPT TEMPLATES
# ═══════════════════════════════════════════════

# Code completion: partial function body only.
# The partial code already contains the function signature and inline docstring
# when present, so do NOT prepend the docstring again. Duplicating it bloats the
# prompt and makes the teacher more likely to explain or repeat text.
CODE_TEMPLATE = """Continue the given Python function from the last line.
Return only the missing Python code.
Do not repeat the given code.
Do not add explanations, notes, examples, tests, markdown, imports, classes, or new functions.
After the completion, write <END>.

{partial_code}"""

MATH_TEMPLATE = """Solve the arithmetic word problem.
Use compact arithmetic. No long explanation.
Do not use LaTeX notation. Use plain numbers only.
Use at most 4 lines.
The last line must be exactly: Final answer: <answer>
Then write <END>.

Problem: {question}

Solution:
"""

TRANSLATION_TEMPLATE = """Translate from {source_lang_name} to {target_lang_name}.
Output only the translation, then write <END>.
Do not repeat the translation or add commentary.

Text:
{source_text}

Translation:
"""


# ═══════════════════════════════════════════════
# 3. LENGTH / QUALITY CONFIG
# ═══════════════════════════════════════════════

@dataclass(frozen=True)
class DomainLimits:
    min_prompt_tokens: int
    max_prompt_tokens: int
    max_new_tokens: int
    min_response_chars: int
    max_response_words: int
    # Code-specific token-budget gates applied before prompt construction.
    # Set to 0 to disable a gate (non-code domains always pass 0).
    max_combined_tokens: int = 0   # docstring tokens + code tokens
    max_docstring_tokens: int = 0  # drop rows where docstring alone is too large
    max_code_tokens: int = 0       # drop rows where function body is too large


DEFAULT_LIMITS = {
    # max_combined_tokens / max_docstring_tokens / max_code_tokens guard against
    # rows where even a short partial-body cut would produce a huge teacher prompt.
    "code": DomainLimits(
        min_prompt_tokens=60,
        max_prompt_tokens=700,
        max_new_tokens=160,
        min_response_chars=3,
        max_response_words=200,
        max_combined_tokens=600,   # docstring + code tokens combined
        max_docstring_tokens=200,  # drop rows where the docstring alone is huge
        max_code_tokens=450,       # drop rows where the function body is huge
    ),
    "math": DomainLimits(
        min_prompt_tokens=12, max_prompt_tokens=180,
        max_new_tokens=128, min_response_chars=3, max_response_words=80,
    ),
    "translation": DomainLimits(
        min_prompt_tokens=8, max_prompt_tokens=120,
        max_new_tokens=96, min_response_chars=3, max_response_words=70,
    ),
}


FRENCH_MARKERS = [
    " le ", " la ", " les ", " des ", " du ", " de ", " un ", " une ",
    " et ", " est ", " dans ", " pour ", " pas ", " que ", " qui ", " avec ",
    " sur ", " aux ", " au ", " en ", " je ", " vous ", " nous ", " il ", " elle ",
]

ENGLISH_MARKERS = [
    " the ", " and ", " is ", " are ", " was ", " were ", " with ", " for ",
    " this ", " that ", " from ", " have ", " has ", " not ", " you ", " we ",
    " they ", " will ", " would ", " can ", " should ",
]

TRANSLATION_DIRECTIONS = {
    "en-fr": {
        "source_key": "en", "target_key": "fr",
        "source_label": "English", "target_label": "French",
        "source_lang_name": "English", "target_lang_name": "French",
    },
    "fr-en": {
        "source_key": "fr", "target_key": "en",
        "source_label": "French", "target_label": "English",
        "source_lang_name": "French", "target_lang_name": "English",
    },
}


# ═══════════════════════════════════════════════
# 4. CHECKPOINT MANAGER
# ═══════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, output_dir: str, domain: str):
        self.output_dir = output_dir
        self.domain = domain
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    @property
    def state_path(self) -> str:
        return os.path.join(self.ckpt_dir, f"{self.domain}_state.json")

    @property
    def prompts_cache_path(self) -> str:
        return os.path.join(self.ckpt_dir, f"{self.domain}_prompts.pkl")

    @property
    def raw_output_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.domain}_distillation_raw.jsonl")

    def has_checkpoint(self) -> bool:
        return os.path.exists(self.state_path)

    def save_state(self, completed: int, total: int) -> None:
        state = {
            "domain": self.domain,
            "completed": completed,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    def load_state(self) -> dict[str, Any]:
        with open(self.state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_prompts(self, prompts: list[dict[str, Any]]) -> None:
        with open(self.prompts_cache_path, "wb") as f:
            pickle.dump(prompts, f)

    def load_prompts(self) -> list[dict[str, Any]]:
        with open(self.prompts_cache_path, "rb") as f:
            return pickle.load(f)

    def completed_prompt_indices(self) -> set[int]:
        """Return prompt indices that already produced non-empty raw responses.

        New raw rows include prompt_index. For older raw files without it, fall
        back to line number (the original writer emitted one line per prompt in
        order). Empty / malformed rows are not considered complete.
        """
        if not os.path.exists(self.raw_output_path):
            return set()

        completed: set[int] = set()
        with open(self.raw_output_path, "r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                response = rec.get("response", "")
                if not isinstance(response, str) or not response.strip():
                    continue
                prompt_index = rec.get("prompt_index")
                if isinstance(prompt_index, int) and prompt_index >= 0:
                    completed.add(prompt_index)
                else:
                    completed.add(line_index)
        return completed

    def count_completed(self) -> int:
        return len(self.completed_prompt_indices())

    def clear(self) -> None:
        for p in [self.state_path, self.prompts_cache_path]:
            if os.path.exists(p):
                os.remove(p)


# ═══════════════════════════════════════════════
# 5. TOKENIZATION HELPERS
# ═══════════════════════════════════════════════

class TokenCounter:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"    ✓ Loaded tokenizer for length filtering: {model_name}")
        except Exception as e:
            print(f"    ⚠ Could not load tokenizer ({e}); falling back to rough word/char estimates")

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, int(len(text) / 4))
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def get_nested(example: dict[str, Any], field: str, nested_key: str | None = None) -> Any:
    value = example.get(field)
    if nested_key is not None and isinstance(value, dict):
        return value.get(nested_key)
    return value


# ═══════════════════════════════════════════════
# 6. SOURCE LOADING
# ═══════════════════════════════════════════════

def load_hf_dataset(source: dict[str, Any]):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": source["split"],
        "trust_remote_code": True,
    }
    if source.get("streaming", False):
        kwargs["streaming"] = True

    if source.get("config"):
        return load_dataset(source["name"], source["config"], **kwargs)
    return load_dataset(source["name"], **kwargs)


def shuffled_iter(ds, seed: int, buffer_size: int = 20_000):
    """Shuffle a HuggingFace dataset (streaming or in-memory) and return an iterator."""
    try:
        return iter(ds.shuffle(seed=seed, buffer_size=buffer_size))
    except TypeError:
        return iter(ds.shuffle(seed=seed))
    except Exception:
        return iter(ds)


# ═══════════════════════════════════════════════
# 7. PROMPT FILTERS AND BUILDERS
# ═══════════════════════════════════════════════

# ---------------------------------------------------------------------------
# CodeSearchNet-specific helpers
# ---------------------------------------------------------------------------

def clean_docstring(raw: str) -> str:
    """Strip surrounding triple-quote markers that some datasets leave in the
    docstring field, then normalise internal whitespace."""
    s = raw.strip()
    for q in ('"""', "'''"):
        if s.startswith(q):
            s = s[len(q):]
        if s.endswith(q):
            s = s[: -len(q)]
    return s.strip()


def is_bad_csn_function(
    code: str,
    docstring: str,
    func_name: str,
    doc_tokens: int,
    code_tokens: int,
    combined_tokens: int,
    limits: DomainLimits,
) -> bool:
    """Return True if this CodeSearchNet row should be skipped.

    Checks (cheapest first):
    1. Non-string or empty code.
    2. Token-budget gates: docstring, code, and combined.
    3. Absolute character-length bounds.
    4. Noise heuristics (auto-generated, test helpers, etc.).
    5. Syntactic validity (ast.parse).
    """
    if not isinstance(code, str) or not code.strip():
        return True

    # ------------------------------------------------------------------
    # Token-budget gates — the key new filter for CodeSearchNet.
    # These run before any expensive operations.
    # ------------------------------------------------------------------
    if limits.max_docstring_tokens > 0 and doc_tokens > limits.max_docstring_tokens:
        return True
    if limits.max_code_tokens > 0 and code_tokens > limits.max_code_tokens:
        return True
    if limits.max_combined_tokens > 0 and combined_tokens > limits.max_combined_tokens:
        return True

    # Absolute size sanity (character-level).
    if len(code) < 80 or len(code) > 60_000:
        return True
    if "\x00" in code:
        return True

    lower_code = code[:2000].lower()
    if any(p in lower_code for p in ("generated by", "auto-generated", "do not edit")):
        return True

    lines = code.splitlines()
    if len(lines) < 4 or len(lines) > 600:
        return True
    if max((len(l) for l in lines), default=0) > 240:
        return True

    # Huge literal blobs (data files masquerading as Python).
    if code.count("{") > 800 or code.count("[") > 800:
        return True

    # Skip test/fixture helpers — trivial bodies, poor completion targets.
    lower_name = (func_name or "").lower()
    if lower_name.startswith(("test_", "setup", "teardown", "fixture")):
        return True

    # Require syntactic validity so partial-body cuts stay realistic.
    try:
        ast.parse(code)
    except Exception:
        return True

    return False


def extract_function_head(code: str) -> str | None:
    """Return the 'def …:' line (possibly multi-line for long signatures)."""
    lines = code.splitlines()
    head_lines: list[str] = []
    in_head = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            in_head = True
        if in_head:
            head_lines.append(line)
            joined = "\n".join(head_lines)
            if joined.count("(") <= joined.count(")") and joined.rstrip().endswith(":"):
                break
        if in_head and len(head_lines) > 12:
            return None   # pathologically complex signature
    return "\n".join(head_lines) if head_lines else None


def choose_partial_body(code: str, rng: random.Random) -> str | None:
    """Cut the function body at a random interior line to create a completion target.

    Returns partial code (function head + partial body) or None if no valid cut
    point is found.

    Strategy:
    - Locate the function head line(s) (def … :).
    - Identify the real body lines, skipping leading blank lines and the
      inline docstring block when choosing the cut point. The docstring is still
      kept in the returned partial code so it appears exactly once.
    - Choose a cut point between 30% and 70% of the non-blank body lines so the
      model always has a meaningful amount remaining to generate.
    - Reject cuts that end on a continuation character.
    - Require at least 2 non-blank lines remaining after the cut.
    """
    lines = code.splitlines()
    if len(lines) < 6:
        return None

    # ── Locate where the function signature ends ──────────────────────────────
    head_end: int | None = None
    in_head = False
    paren_depth = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            in_head = True
        if in_head:
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0 and line.rstrip().endswith(":"):
                head_end = i
                break
            if i > 15:
                return None   # pathological signature

    if head_end is None or head_end >= len(lines) - 2:
        return None

    body_lines = lines[head_end + 1:]

    # ── Skip leading blank lines and the inline docstring block ───────────────
    body_start_offset = 0
    in_docstring = False
    docstring_quote: str | None = None
    for j, bline in enumerate(body_lines):
        stripped = bline.strip()
        if not stripped:
            body_start_offset = j + 1
            continue
        if not in_docstring:
            for q in ('"""', "'''"):
                if stripped.startswith(q):
                    in_docstring = True
                    docstring_quote = q
                    rest = stripped[len(q):]
                    if rest.endswith(q) and len(rest) >= len(q):
                        in_docstring = False   # single-line docstring
                        body_start_offset = j + 1
                    break
            if not in_docstring:
                break
        else:
            if docstring_quote and docstring_quote in bline:
                in_docstring = False
                body_start_offset = j + 1

    real_body = body_lines[body_start_offset:]
    non_blank = [l for l in real_body if l.strip()]
    if len(non_blank) < 3:
        return None   # too little body to be a useful completion target

    # ── Choose cut point between 30–70% of the non-blank body lines ──────────
    lo = max(1, int(len(non_blank) * 0.30))
    hi = max(lo + 1, int(len(non_blank) * 0.70))
    cut_count = rng.randint(lo, hi)

    kept_body: list[str] = []
    nb_seen = 0
    for bline in real_body:
        if nb_seen >= cut_count:
            break
        kept_body.append(bline)
        if bline.strip():
            nb_seen += 1

    if not kept_body:
        return None

    # Reject cuts that leave a hanging continuation character.
    if kept_body[-1].rstrip().endswith(("\\", ",", "(", "[", "{")):
        return None

    # Require at least 2 non-blank lines remaining after the cut.
    remaining_nb = sum(1 for l in real_body[len(kept_body):] if l.strip())
    if remaining_nb < 2:
        return None

    # ── Reassemble: function head + skipped blanks/docstring + body cut ───────
    partial = "\n".join(lines[: head_end + 1]) + "\n"
    if body_start_offset > 0:
        partial += "\n".join(body_lines[:body_start_offset]) + "\n"
    partial += "\n".join(kept_body)
    return partial.rstrip() + "\n"


def build_code_prompt(docstring: str, partial_code: str) -> str:
    """Assemble the teacher prompt without duplicating the docstring.

    `partial_code` already includes the function head and the inline docstring
    from the original source when one exists. The separate `docstring` argument
    is kept for metadata/dedup compatibility, but it is intentionally not
    inserted into the prompt.
    """
    return CODE_TEMPLATE.format(partial_code=partial_code)


def load_code_prompts(
    target_count: int,
    token_counter: TokenCounter,
    limits: DomainLimits,
    seed: int,
) -> list[dict[str, Any]]:
    """Load CodeSearchNet Python, apply token-budget + quality filters, and build prompts.

    Each output prompt has the form:
        Complete the following Python function.

        <docstring>     ← omitted if the function has no docstring

        def func_name(...):
            <existing body lines up to 30-70% of the non-blank total>
    """
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    rng = random.Random(seed)

    for source in CODE_SOURCES:
        if len(prompts) >= target_count:
            break
        print(
            f"    Loading {source['name']} [{source['config']}] "
            f"(need {target_count - len(prompts):,} prompts)..."
        )
        try:
            ds = load_hf_dataset(source)
            it = shuffled_iter(ds, seed=seed)
        except Exception as e:
            print(f"    ⚠ Failed to load {source['name']}: {e}")
            continue

        skipped_token = skipped_quality = skipped_cut = skipped_seen = scanned = 0

        for ex in it:
            if len(prompts) >= target_count:
                break
            scanned += 1

            try:
                code: str = ex.get(source["content_field"]) or ""
                raw_doc: str = ex.get(source["docstring_field"]) or ""
                func_name: str = ex.get(source["name_field"]) or ""

                docstring = clean_docstring(raw_doc)

                # ── Token-budget pre-filter (cheap, before any slicing) ───────
                doc_tokens = token_counter.count(docstring) if docstring else 0
                code_tokens = token_counter.count(code)
                combined_tokens = doc_tokens + code_tokens

                if is_bad_csn_function(
                    code, docstring, func_name,
                    doc_tokens, code_tokens, combined_tokens, limits,
                ):
                    skipped_token += 1
                    continue

                # ── Create the partial-code completion target ─────────────────
                partial_code = choose_partial_body(code, rng)
                if partial_code is None:
                    skipped_cut += 1
                    continue

                # ── Build and length-check the full teacher prompt ────────────
                teacher_prompt = build_code_prompt(docstring, partial_code)
                n_tok = token_counter.count(teacher_prompt)
                if n_tok < limits.min_prompt_tokens or n_tok > limits.max_prompt_tokens:
                    skipped_token += 1
                    continue

                # ── Deduplicate by (docstring + function head) ────────────────
                head = extract_function_head(code) or func_name
                dedup_key = stable_hash((docstring + "\n" + head).strip())
                if dedup_key in seen:
                    skipped_seen += 1
                    continue
                seen.add(dedup_key)

                prompts.append({
                    "domain": "code",
                    "source_dataset": source["name"],
                    "prompt": teacher_prompt,
                    "raw_prefix": partial_code,   # stored for filtering / auditing
                    "prompt_tokens": n_tok,
                    "metadata": {
                        "func_name": func_name,
                        "has_docstring": bool(docstring),
                        "docstring_tokens": doc_tokens,
                        "code_tokens": code_tokens,
                        "combined_tokens": combined_tokens,
                    },
                })

            except Exception:
                skipped_quality += 1
                continue

            if scanned % 50_000 == 0:
                print(
                    f"      scanned={scanned:,} kept={len(prompts):,} "
                    f"skip_token={skipped_token:,} skip_cut={skipped_cut:,} "
                    f"skip_seen={skipped_seen:,} skip_err={skipped_quality:,}"
                )

        print(
            f"    Finished {source['name']}: kept={len(prompts):,}/{target_count:,} "
            f"| scanned={scanned:,} skip_token={skipped_token:,} "
            f"skip_cut={skipped_cut:,} skip_seen={skipped_seen:,} "
            f"skip_err={skipped_quality:,}"
        )

    if len(prompts) < target_count:
        print(
            f"    ⚠ Only {len(prompts):,} code prompts found after filtering "
            f"(target {target_count:,}). Consider raising --oversample_factor or "
            f"relaxing --code_max_combined_tokens / --code_max_code_tokens."
        )

    return prompts[:target_count]


def is_good_math_problem(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = normalize_text(text)
    if len(t) < 20 or len(t) > 900:
        return False
    words = t.split()
    if len(words) < 6 or len(words) > 140:
        return False
    lower = t.lower()
    bad_phrases = (
        "prove that", "show that", "derive", "write a proof", "olympiad", "diagram",
        "figure", "graph below", "following table", "asymptote", "complex plane",
    )
    if any(p in lower for p in bad_phrases):
        return False
    if lower.count("\\") > 15:
        return False
    return True


def is_good_translation_source(text: str, source_lang: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 20 or len(t) > 220:
        return False
    if t.startswith(("[{", '{"', "<", "=")):
        return False
    lower = f" {t.lower()} "
    words = t.split()
    if len(words) < 5 or len(words) > 32:
        return False
    if any(x in lower for x in ["http://", "https://", "www.", "doi:", ".pdf", ".jpg", ".png", "@"]):
        return False
    if t.endswith("..."):
        return False
    alpha_words = [w for w in words if sum(c.isalpha() for c in w) >= 3]
    if len(alpha_words) < 4:
        return False
    french_hits = sum(1 for marker in FRENCH_MARKERS if marker in lower)
    english_hits = sum(1 for marker in ENGLISH_MARKERS if marker in lower)
    if source_lang == "en" and french_hits >= 4 and english_hits <= 1:
        return False
    if source_lang == "fr" and english_hits >= 4 and french_hits <= 1:
        return False
    return True


def load_simple_text_prompts(
    domain: str,
    sources: list[dict[str, Any]],
    target_count: int,
    token_counter: TokenCounter,
    limits: DomainLimits,
    seed: int,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source in sources:
        if len(prompts) >= target_count:
            break
        print(f"    Loading {source['name']} for {domain} prompts (need {target_count - len(prompts):,})...")
        try:
            ds = load_hf_dataset(source)
            it = shuffled_iter(ds, seed=seed)
        except Exception as e:
            print(f"    ⚠ Failed to load {source['name']}: {e}")
            continue

        scanned = skipped = 0
        for ex in it:
            if len(prompts) >= target_count:
                break
            scanned += 1
            try:
                raw = get_nested(ex, source["prompt_field"], source.get("nested_key"))
                if raw is None:
                    skipped += 1
                    continue
                raw = normalize_text(str(raw))

                if domain == "math":
                    if not is_good_math_problem(raw):
                        skipped += 1
                        continue
                    teacher_prompt = MATH_TEMPLATE.format(question=raw)
                    extra: dict[str, Any] = {"question": raw}
                elif domain == "translation":
                    raise ValueError("Use load_translation_prompts() for bidirectional translation data")
                else:
                    raise ValueError(f"Unsupported text domain: {domain}")

                n_tok = token_counter.count(teacher_prompt)
                if n_tok < limits.min_prompt_tokens or n_tok > limits.max_prompt_tokens:
                    skipped += 1
                    continue
                h = stable_hash(raw)
                if h in seen:
                    skipped += 1
                    continue
                seen.add(h)

                prompts.append({
                    "domain": domain,
                    "source_dataset": source["name"],
                    "prompt": teacher_prompt,
                    "prompt_tokens": n_tok,
                    "metadata": extra,
                })
            except Exception:
                skipped += 1
                continue

            if scanned % 100_000 == 0:
                print(f"      scanned={scanned:,}, kept={len(prompts):,}, skipped={skipped:,}")

        print(f"    Got {len(prompts):,} total {domain} prompts so far from {source['name']} "
              f"(scanned {scanned:,}, skipped {skipped:,})")

    return prompts[:target_count]


def load_translation_prompts(
    total_target_count: int,
    token_counter: TokenCounter,
    limits: DomainLimits,
    seed: int,
) -> list[dict[str, Any]]:
    """Load 50/50 translation prompts: English→French and French→English."""
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_direction_target = total_target_count // 2
    remainder = total_target_count - 2 * per_direction_target
    direction_targets = {
        "en-fr": per_direction_target + remainder,
        "fr-en": per_direction_target,
    }

    for direction, dir_target in direction_targets.items():
        dir_info = TRANSLATION_DIRECTIONS[direction]
        dir_kept = 0
        print(f"    Translation direction {direction}: target raw prompts={dir_target:,}")

        for source in TRANSLATION_SOURCES_BY_DIRECTION[direction]:
            if dir_kept >= dir_target:
                break
            print(f"      Loading {source['name']} [{source['config']}] for {direction} "
                  f"(need {dir_target - dir_kept:,})...")
            try:
                ds = load_hf_dataset(source)
                it = shuffled_iter(ds, seed=seed + (17 if direction == "fr-en" else 0))
            except Exception as e:
                print(f"      ⚠ Failed to load {source['name']} [{source.get('config')}]: {e}")
                continue

            scanned = skipped = 0
            for ex in it:
                if dir_kept >= dir_target:
                    break
                scanned += 1
                try:
                    pair = ex.get(source["prompt_field"])
                    if not isinstance(pair, dict):
                        skipped += 1
                        continue
                    src = normalize_text(str(pair.get(source["source_key"], "")))
                    ref = normalize_text(str(pair.get(source["target_key"], "")))
                    if not src or not ref:
                        skipped += 1
                        continue
                    if not is_good_translation_source(src, source["source_key"]):
                        skipped += 1
                        continue
                    teacher_prompt = TRANSLATION_TEMPLATE.format(
                        source_text=src,
                        source_label=dir_info["source_label"],
                        target_label=dir_info["target_label"],
                        source_lang_name=dir_info["source_lang_name"],
                        target_lang_name=dir_info["target_lang_name"],
                    )
                    n_tok = token_counter.count(teacher_prompt)
                    if n_tok < limits.min_prompt_tokens or n_tok > limits.max_prompt_tokens:
                        skipped += 1
                        continue
                    h = stable_hash(direction + "\n" + src)
                    if h in seen:
                        skipped += 1
                        continue
                    seen.add(h)
                    prompts.append({
                        "domain": "translation",
                        "source_dataset": source["name"],
                        "prompt": teacher_prompt,
                        "prompt_tokens": n_tok,
                        "metadata": {
                            "direction": direction,
                            "source_lang": source["source_key"],
                            "target_lang": source["target_key"],
                            "source_text": src,
                            "reference_translation": ref,
                        },
                    })
                    dir_kept += 1
                except Exception:
                    skipped += 1
                    continue

                if scanned % 100_000 == 0:
                    print(f"        scanned={scanned:,}, kept_direction={dir_kept:,}, "
                          f"kept_total={len(prompts):,}, skipped={skipped:,}")

            print(f"      Got {dir_kept:,}/{dir_target:,} {direction} prompts so far "
                  f"from {source['name']} (scanned {scanned:,}, skipped {skipped:,})")

        if dir_kept < dir_target:
            print(f"    ⚠ Only {dir_kept:,}/{dir_target:,} raw prompts found for {direction}")

    random.Random(seed).shuffle(prompts)
    return prompts[:total_target_count]


def load_domain_prompts(
    domain: str,
    target_count: int,
    model_name: str,
    seed: int,
    limits: DomainLimits,
) -> list[dict[str, Any]]:
    token_counter = TokenCounter(model_name)
    if domain == "code":
        return load_code_prompts(target_count, token_counter, limits, seed)
    if domain == "math":
        return load_simple_text_prompts(domain, MATH_SOURCES, target_count, token_counter, limits, seed)
    if domain == "translation":
        return load_translation_prompts(target_count, token_counter, limits, seed)
    raise ValueError(f"Unknown domain: {domain}")


# ═══════════════════════════════════════════════
# 8. TEACHER INFERENCE
# ═══════════════════════════════════════════════

def run_teacher_inference(
    prompt_records: list[dict[str, Any]],
    domain: str,
    model_name: str,
    batch_size: int,
    max_tokens: int,
    ckpt: CheckpointManager,
    checkpoint_every: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    dtype: str,
    temperature: float,
):
    from vllm import LLM, SamplingParams

    completed_indices = ckpt.completed_prompt_indices()
    total = len(prompt_records)
    already_done = len(completed_indices)

    if already_done >= total:
        print(f"    ✓ All {total:,} non-empty samples already generated. Skipping inference.")
        return

    if already_done > 0:
        print(f"    ↻ RESUMING with {already_done:,}/{total:,} non-empty samples complete "
              f"({already_done / total * 100:.1f}% done)")

    # Math: no stop token — let the model emit <END> naturally so it appears in the raw
    # file and can be preserved by clean_response.
    stop: list[str] = []
    if domain == "translation":
        stop = [
            "\n\n",
            "\nTranslation:",
            "\nNote:",
            "\nTo stop",
            "\nTo provide",
            "\nAs per",
            "\nTranslation complete",
            "\nStopping here",
        ]
    elif domain == "code":
        stop = [
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

    sampling_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "min_tokens": 3,
        "stop": stop,
    }
    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("min_tokens", None)
        sampling_params = SamplingParams(**sampling_kwargs)
        print("    ⚠ vLLM SamplingParams does not support min_tokens; continuing without it")

    print(f"\n    Loading model: {model_name}")
    print(f"    dtype={dtype} | gpu_memory_utilization={gpu_memory_utilization} | max_model_len={max_model_len}")
    print(f"    max_new_tokens[{domain}]={max_tokens}; outputs that hit this limit will be rejected during filtering")

    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    remaining = [(idx, rec) for idx, rec in enumerate(prompt_records) if idx not in completed_indices]
    written = already_done
    raw_generated = 0
    since_ckpt = 0
    start = time.time()

    with open(ckpt.raw_output_path, "a", encoding="utf-8") as f:
        for batch_start in range(0, len(remaining), batch_size):
            batch_items = remaining[batch_start: batch_start + batch_size]
            batch_prompts = [rec["prompt"] for _, rec in batch_items]

            try:
                outputs = llm.generate(batch_prompts, sampling_params)
            except Exception as e:
                print(f"\n    ⚠ ERROR at remaining batch index {batch_start}: {e}")
                f.flush()
                os.fsync(f.fileno())
                ckpt.save_state(written, total)
                print(f"    Checkpoint saved at {written:,} samples. Re-run to resume.")
                raise

            for (prompt_index, rec), out in zip(batch_items, outputs):
                response = out.outputs[0].text
                out_rec = {
                    "domain": domain,
                    "prompt_index": prompt_index,
                    "source_dataset": rec.get("source_dataset"),
                    "prompt": rec["prompt"],
                    "response": response,
                    "tokens_generated": len(out.outputs[0].token_ids),
                    "prompt_tokens": rec.get("prompt_tokens"),
                    "metadata": rec.get("metadata", {}),
                }
                if "raw_prefix" in rec:
                    out_rec["raw_prefix"] = rec["raw_prefix"]
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                raw_generated += 1

                if response:
                    completed_indices.add(prompt_index)
                    written = len(completed_indices)
                    since_ckpt += 1

            if since_ckpt >= checkpoint_every:
                f.flush()
                os.fsync(f.fileno())
                ckpt.save_state(written, total)
                since_ckpt = 0

            elapsed = time.time() - start
            rate = (written - already_done) / elapsed if elapsed > 0 else 0
            eta = (total - written) / rate if rate > 0 else 0
            print(
                f"    [{domain}] usable raw {written:,}/{total:,} ({written / total * 100:.1f}%) "
                f"| generated this run {raw_generated:,} | {rate:.1f} usable samples/s "
                f"| ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}"
            )

        f.flush()
        os.fsync(f.fileno())

    ckpt.save_state(written, total)
    print(f"\n    ✓ Inference pass complete: {written:,} non-empty raw samples for [{domain}] "
          f"({raw_generated:,} generations this run)")

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ═══════════════════════════════════════════════
# 9. OUTPUT FILTERING
# ═══════════════════════════════════════════════

def check_text(resp: str) -> str:
    """Return a stripped view used only for filtering checks.

    IMPORTANT: this function must never be used to overwrite rec["response"].
    For speculative-decoding distillation, the accepted response should be the
    exact text emitted by the teacher under the configured SamplingParams.
    Filtering may reject a row, but it must not rewrite an accepted row.
    """
    return resp.strip()


def has_repeated_line_loop(resp: str) -> bool:
    """Detect obvious degenerate repetition without judging correctness."""
    lines = [l.strip() for l in resp.splitlines() if l.strip()]
    if len(lines) >= 8 and max(lines.count(l) for l in set(lines)) >= 4:
        return True
    return False


def has_artifact_marker(resp: str) -> bool:
    """Detect dataset/edit artifacts that are not useful teacher behavior.

    These are not semantic-quality checks. They target patch/cursor markers and
    templating junk observed in generated rows, while preserving normal comments,
    imperfect code, wrong math, and awkward translations exactly as emitted.
    """
    lower = resp.lower()
    artifact_patterns = (
        "@@", "[problem]", "__missing", "$$$$", "<<<<", "<----",
        "# line ", "# eof", "# cursor", "# @", "# $$$", "# ____",
        "missing continuation", "missing line", "to complete the code",
        "# continue the function", "# add the missing", "# continue from here",
    )
    return any(p in lower for p in artifact_patterns)


def is_bad_response(
    rec: dict[str, Any],
    domain: str,
    limits: DomainLimits,
    filter_mode: str = "behavior",
    keep_max_token_hits: bool = False,
) -> tuple[bool, str]:
    """Minimal output filter for pure speculative-decoding distribution matching.

    Goal: train a drafter to match the target model's token distribution.
    Therefore, do NOT reject teacher outputs for correctness, formatting,
    verbosity, markdown, think leaks, repeated loops, artifacts, comments,
    imports, labels, missing final answers, or other target-model behaviors.

    The only rows rejected are those that are unusable or externally truncated:
    1. response is not a string
    2. response is empty / too short
    3. response hit max_new_tokens, unless --keep_max_token_hits is set

    Stop sequences are still applied during generation in run_teacher_inference(),
    and the accepted response is kept exactly as emitted by vLLM.
    """
    raw_resp = rec.get("response", "")
    if not isinstance(raw_resp, str):
        return True, "non_string_response"

    # Strip only for checking emptiness. Do not rewrite rec["response"].
    resp = raw_resp.strip()
    if not resp or len(resp) < limits.min_response_chars:
        return True, "empty_or_too_short"

    rec["hit_max_new_tokens"] = rec.get("tokens_generated", 0) >= limits.max_new_tokens
    if rec["hit_max_new_tokens"] and not keep_max_token_hits:
        return True, "hit_max_new_tokens"

    # Keep these as stats only. Do not reject on them.
    rec["response_exceeds_soft_word_limit"] = len(resp.split()) > limits.max_response_words

    return False, "ok"

_TRANSLATION_STOP_PATTERNS = [
    # newline-prefixed patterns
    "\n\n", "\nTranslation:", "\nNote:", "\nTo stop", "\nTo provide",
    "\nAs per", "\nTranslation complete", "\nStopping here", "\nDo not",
    # inline / same-line patterns observed in outputs
    ". To ensure", ". To provide", ". To translate", ". To the given",
    ". To stop", ". To avoid", ". To give", ". To make", ". To preserve",
    " To ensure", " To provide",
    " (Corrected", " (Note:", " (This part",
    " (Revised", " (Updated",
]


def clean_response(response: str, domain: str) -> str:
    """Truncate trailing artifacts from teacher responses without rejecting them.

    Applied to the final JSONL only; the raw file is never modified.
    """
    if domain == "translation":
        # Primary: truncate at <END> if the model emitted it
        end_idx = response.find("<END>")
        if end_idx != -1:
            return response[:end_idx + len("<END>")]
        # Fallback: pattern-based truncation for records without <END>
        earliest = len(response)
        for pat in _TRANSLATION_STOP_PATTERNS:
            idx = response.find(pat)
            if 0 < idx < earliest:
                earliest = idx
        return response[:earliest].strip()

    if domain == "math":
        lines = response.split("\n")
        clean: list[str] = []
        found_final = False
        for line in lines:
            if not found_final:
                clean.append(line)
                if re.match(r"^\s*Final answer:", line, re.IGNORECASE):
                    found_final = True
            elif line.strip() == "<END>":
                clean.append("<END>")
                break
            else:
                break  # stop at first non-<END> content after Final answer:
        return "\n".join(clean).strip()

    if domain == "code":
        # Truncate before any line containing an artifact marker
        lines = response.split("\n")
        clean_lines: list[str] = []
        for line in lines:
            if has_artifact_marker(line):
                break
            clean_lines.append(line)
        response = "\n".join(clean_lines)
        # Keep up to and including <END> if the model emitted it
        idx = response.find("<END>")
        if idx != -1:
            return response[:idx + len("<END>")]
        return response

    return response


def filter_outputs(
    input_path: str,
    output_path: str,
    domain: str,
    limits: DomainLimits,
    target_n: int,
    rejected_path: str | None = None,
    filter_mode: str = "behavior",
    keep_max_token_hits: bool = False,
) -> int:
    kept = 0
    removed = 0
    reasons: dict[str, int] = {}
    rejected_fout = open(rejected_path, "w", encoding="utf-8") if rejected_path else None

    direction_targets: dict[str, int] | None = None
    direction_kept: dict[str, int] = {}
    if domain == "translation":
        per_dir = target_n // 2
        direction_targets = {
            "en-fr": per_dir + (target_n - 2 * per_dir),
            "fr-en": per_dir,
        }
        direction_kept = {"en-fr": 0, "fr-en": 0}

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if kept >= target_n:
                break
            rec = json.loads(line)

            if direction_targets is not None:
                direction = rec.get("metadata", {}).get("direction")
                if direction not in direction_targets:
                    removed += 1
                    reasons["missing_translation_direction"] = reasons.get("missing_translation_direction", 0) + 1
                    continue
                if direction_kept[direction] >= direction_targets[direction]:
                    continue

            bad, reason = is_bad_response(rec, domain, limits, filter_mode=filter_mode, keep_max_token_hits=keep_max_token_hits)
            if bad:
                removed += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                if rejected_fout is not None:
                    rec["rejection_reason"] = reason
                    rejected_fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            cleaned = clean_response(rec["response"], domain)
            if cleaned != rec["response"]:
                rec = {**rec, "response": cleaned, "response_cleaned": True}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if direction_targets is not None:
                direction_kept[rec["metadata"]["direction"]] += 1

            if direction_targets is not None and all(
                direction_kept[d] >= direction_targets[d] for d in direction_targets
            ):
                break

    total_seen = kept + removed
    print(f"    [{domain}] kept {kept:,}, removed {removed:,} from first {total_seen:,} checked")
    print(f"    Filter mode: {filter_mode}; keep_max_token_hits={keep_max_token_hits}")
    if direction_targets is not None:
        print(
            "    Translation balance: "
            + ", ".join(f"{d}={direction_kept.get(d, 0):,}/{direction_targets[d]:,}" for d in direction_targets)
        )
    if reasons:
        top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:6]
        print("    Removal reasons: " + ", ".join(f"{k}={v:,}" for k, v in top))
    if kept < target_n:
        print(f"    ⚠ Only {kept:,}/{target_n:,} survived filtering. "
              "Re-run with higher --oversample_factor or improve stop sequences/loosen filters.")
    if rejected_fout is not None:
        rejected_fout.close()
        print(f"    Rejected exact rows written to: {rejected_path}")
    return kept


def print_dataset_stats(filepath: str, domain: str) -> None:
    total = total_tok = max_token_hits = soft_word_limit_hits = has_docstring_count = 0
    direction_counts: dict[str, int] = {}
    p_lens: list[int] = []
    r_lens: list[int] = []

    if not os.path.exists(filepath):
        print(f"    [{domain}] Missing output file")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            total_tok += int(rec.get("tokens_generated") or 0)
            max_token_hits += int(bool(rec.get("hit_max_new_tokens")))
            soft_word_limit_hits += int(bool(rec.get("response_exceeds_soft_word_limit")))
            if rec.get("metadata", {}).get("has_docstring"):
                has_docstring_count += 1
            direction = rec.get("metadata", {}).get("direction")
            if direction:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            p_lens.append(int(rec.get("prompt_tokens") or len(rec.get("prompt", "").split())))
            r_lens.append(len(rec.get("response", "").split()))

    if total == 0:
        print(f"    [{domain}] Empty dataset!")
        return

    def pct(xs: list[int], q: float) -> int:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * q))]

    print(f"\n    [{domain}] Stats: {total:,} samples | generated tokens kept: {total_tok:,}")
    print(f"    Prompt tokens: avg {sum(p_lens) / total:.0f}, p50 {pct(p_lens, 0.50)}, p95 {pct(p_lens, 0.95)}")
    print(f"    Response words: avg {sum(r_lens) / total:.0f}, p50 {pct(r_lens, 0.50)}, "
          f"p95 {pct(r_lens, 0.95)}, max {max(r_lens)}")
    print(f"    Flagged (kept): max_new_tokens hits {max_token_hits:,} ({max_token_hits / total * 100:.1f}%), "
          f"soft word-limit hits {soft_word_limit_hits:,} ({soft_word_limit_hits / total * 100:.1f}%)")
    if domain == "code":
        print(f"    Samples with docstring: {has_docstring_count:,} ({has_docstring_count / total * 100:.1f}%)")
    if direction_counts:
        print("    Direction counts: " + ", ".join(f"{k}={v:,}" for k, v in sorted(direction_counts.items())))


# ═══════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate short-prompt distillation data for code/math/translation"
    )
    parser.add_argument("--samples_per_domain", type=int, default=80_000)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./distillation_data")
    parser.add_argument("--domains", nargs="+",
                        choices=["math", "code", "translation", "all"], default=["all"])
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--oversample_factor", type=float, default=1.2,
                        help="Generate extra raw samples for filtering. behavior mode needs only light oversampling")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear existing checkpoint/cache/raw output for selected domains")
    parser.add_argument("--skip_filter", action="store_true")
    parser.add_argument("--filter_mode", choices=["none", "behavior", "strict"], default="behavior",
                        help="Filtering strictness. Use behavior for speculative decoding distribution matching.")
    parser.add_argument("--keep_max_token_hits", action="store_true",
                        help="Keep exact responses even when they hit max_new_tokens. Usually leave off to avoid loop targets.")
    parser.add_argument("--save_rejected", action="store_true",
                        help="Write rejected exact raw rows to <domain>_distillation_rejected.jsonl for audit/debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Code token-budget overrides (the new knobs for CodeSearchNet filtering).
    parser.add_argument("--code_max_prompt_tokens", type=int,
                        default=DEFAULT_LIMITS["code"].max_prompt_tokens)
    parser.add_argument("--code_max_combined_tokens", type=int,
                        default=DEFAULT_LIMITS["code"].max_combined_tokens,
                        help="Max docstring+code tokens; rows above this are dropped before prompt building")
    parser.add_argument("--code_max_docstring_tokens", type=int,
                        default=DEFAULT_LIMITS["code"].max_docstring_tokens,
                        help="Max tokens in docstring alone; rows above this are dropped")
    parser.add_argument("--code_max_code_tokens", type=int,
                        default=DEFAULT_LIMITS["code"].max_code_tokens,
                        help="Max tokens in function code alone; rows above this are dropped")
    parser.add_argument("--code_max_new_tokens", type=int,
                        default=DEFAULT_LIMITS["code"].max_new_tokens)

    # Math / translation overrides (unchanged from original).
    parser.add_argument("--math_max_prompt_tokens", type=int,
                        default=DEFAULT_LIMITS["math"].max_prompt_tokens)
    parser.add_argument("--math_max_new_tokens", type=int,
                        default=DEFAULT_LIMITS["math"].max_new_tokens)
    parser.add_argument("--translation_max_prompt_tokens", type=int,
                        default=DEFAULT_LIMITS["translation"].max_prompt_tokens)
    parser.add_argument("--translation_max_new_tokens", type=int,
                        default=DEFAULT_LIMITS["translation"].max_new_tokens)

    args = parser.parse_args()

    if args.samples_per_domain < 10 or args.samples_per_domain > 100_000:
        parser.error("--samples_per_domain must be between 10 and 100000")
    if args.oversample_factor < 1.0:
        parser.error("--oversample_factor must be >= 1.0")

    os.makedirs(args.output_dir, exist_ok=True)
    domains = ["code", "math", "translation"] if "all" in args.domains else args.domains

    limits = {
        "code": DomainLimits(
            min_prompt_tokens=DEFAULT_LIMITS["code"].min_prompt_tokens,
            max_prompt_tokens=args.code_max_prompt_tokens,
            max_new_tokens=args.code_max_new_tokens,
            min_response_chars=DEFAULT_LIMITS["code"].min_response_chars,
            max_response_words=DEFAULT_LIMITS["code"].max_response_words,
            max_combined_tokens=args.code_max_combined_tokens,
            max_docstring_tokens=args.code_max_docstring_tokens,
            max_code_tokens=args.code_max_code_tokens,
        ),
        "math": DomainLimits(
            min_prompt_tokens=DEFAULT_LIMITS["math"].min_prompt_tokens,
            max_prompt_tokens=args.math_max_prompt_tokens,
            max_new_tokens=args.math_max_new_tokens,
            min_response_chars=DEFAULT_LIMITS["math"].min_response_chars,
            max_response_words=DEFAULT_LIMITS["math"].max_response_words,
        ),
        "translation": DomainLimits(
            min_prompt_tokens=DEFAULT_LIMITS["translation"].min_prompt_tokens,
            max_prompt_tokens=args.translation_max_prompt_tokens,
            max_new_tokens=args.translation_max_new_tokens,
            min_response_chars=DEFAULT_LIMITS["translation"].min_response_chars,
            max_response_words=DEFAULT_LIMITS["translation"].max_response_words,
        ),
    }

    raw_target = int(args.samples_per_domain * args.oversample_factor)

    col = 44
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Distillation Dataset Generator — short prompts            ║
║  Model:        {args.model_name:<{col}}║
║  Final target: {args.samples_per_domain:>7,} per domain{' ' * 26}║
║  Raw target:   {raw_target:>7,} per domain (oversampled){' ' * 17}║
║  Domains:      {', '.join(domains):<{col}}║
║  Code source:  CodeSearchNet Python (partial; no duplicate doc)    ║
║  Token limits  (code): doc={args.code_max_docstring_tokens} code={args.code_max_code_tokens} combined={args.code_max_combined_tokens}    ║
║  Max new toks  code/math/trans: {args.code_max_new_tokens}/{args.math_max_new_tokens}/{args.translation_max_new_tokens:<19}║
╚════════════════════════════════════════════════════════════╝
""")

    for domain in domains:
        domain_limits = limits[domain]
        ckpt = CheckpointManager(args.output_dir, domain)
        final_path = os.path.join(args.output_dir, f"{domain}_distillation.jsonl")

        print(f"\n{'━' * 70}")
        print(f"  DOMAIN: {domain.upper()} | final target={args.samples_per_domain:,} | raw target={raw_target:,}")
        print(f"{'━' * 70}")

        if args.fresh:
            print("  --fresh: clearing checkpoint/cache/raw/final outputs")
            ckpt.clear()
            for p in [ckpt.raw_output_path, final_path]:
                if os.path.exists(p):
                    os.remove(p)

        if ckpt.has_checkpoint() and os.path.exists(ckpt.prompts_cache_path):
            print("\n  [1/4] Restoring cached prompts...")
            prompt_records = ckpt.load_prompts()
            state = ckpt.load_state()
            print(f"    ✓ {len(prompt_records):,} prompts restored "
                  f"(state {state['completed']:,}/{state['total']:,})")
        else:
            print("\n  [1/4] Loading/filtering prompts from Hugging Face...")
            prompt_records = load_domain_prompts(
                domain=domain,
                target_count=raw_target,
                model_name=args.model_name,
                seed=args.seed,
                limits=domain_limits,
            )
            if len(prompt_records) < args.samples_per_domain:
                print(f"    ⚠ Only {len(prompt_records):,} prompts found; "
                      f"target final is {args.samples_per_domain:,}")
            ckpt.save_prompts(prompt_records)
            print(f"    ✓ Cached {len(prompt_records):,} prompt records")

        print("\n  [2/4] Teacher inference...")
        run_teacher_inference(
            prompt_records=prompt_records,
            domain=domain,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_tokens=domain_limits.max_new_tokens,
            ckpt=ckpt,
            checkpoint_every=args.checkpoint_every,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            temperature=args.temperature,
        )

        print("\n  [3/4] Filtering outputs...")
        if args.skip_filter:
            shutil.copy2(ckpt.raw_output_path, final_path)
            print("    Filtering skipped; copied raw output")
        else:
            rejected_path = os.path.join(args.output_dir, f"{domain}_distillation_rejected.jsonl") if args.save_rejected else None
            filter_outputs(ckpt.raw_output_path, final_path, domain, domain_limits, args.samples_per_domain, rejected_path, args.filter_mode, args.keep_max_token_hits)

        print("\n  [4/4] Final stats...")
        print_dataset_stats(final_path, domain)

        ckpt.clear()
        print(f"\n  ✓ [{domain.upper()}] DONE — checkpoint cleared")

    print(f"\n{'=' * 70}")
    print(f"  ALL DONE — outputs in {args.output_dir}/")
    print(f"{'=' * 70}")
    for d in domains:
        p = os.path.join(args.output_dir, f"{d}_distillation.jsonl")
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1048576
            lines = sum(1 for _ in open(p, encoding="utf-8"))
            print(f"  {p}: {lines:,} samples ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
