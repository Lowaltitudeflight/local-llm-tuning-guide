#!/usr/bin/env python3
"""check_model_matrix.py - re-derive the KV/token numbers in MODEL_MATRIX.md.

A KV figure is cheap to compute and expensive to get wrong: a table that says
64 KiB where the formula says 128 KiB understates the memory budget by 2x and
looks entirely plausible. This script recomputes every row whose model geometry
is public and fails the build on a mismatch.

Formula (per token, per full model):

    2 (K and V) * layers * kv_heads * head_dim * bytes_per_element

bytes_per_element is 2 for fp16/bf16 and 1 for fp8.

Usage: python tools/check_model_matrix.py
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_EN = os.path.join(ROOT, "MODEL_MATRIX.md")
MATRIX_ZH = os.path.join(ROOT, "MODEL_MATRIX.zh-CN.md")

# Publishable geometry: layers, KV heads, head_dim. Add a row only when the
# numbers come from the model's own config.json, not from a neighbouring model.
GEOMETRY: Dict[str, Tuple[int, int, int]] = {
    "Qwen3-8B": (36, 8, 128),
    "Qwen3.8-27B": (64, 4, 256),
    "Qwen3-32B": (64, 8, 128),
    "Qwen3-235B-A22B": (94, 4, 128),
    "MiniMax-M3.1-Flash": (60, 4, 128),
}

KIB = 1024
TOLERANCE = 0.02  # the table rounds to whole KiB


def kv_kib(layers: int, kv_heads: int, head_dim: int, bytes_per_element: int) -> float:
    return 2 * layers * kv_heads * head_dim * bytes_per_element / KIB


def expected(model: str, dtype: str) -> float:
    layers, kv_heads, head_dim = GEOMETRY[model]
    return kv_kib(layers, kv_heads, head_dim, 2 if dtype in ("fp16", "bf16") else 1)


def parse_number(cell: str) -> Optional[float]:
    match = re.search(r"([\d.]+)\s*KiB", cell)
    return float(match.group(1)) if match else None


def check_derivation_table(path: str) -> List[str]:
    """The '## KV/token derivation' table must match the formula exactly."""
    errors: List[str] = []
    if not os.path.exists(path):
        return [f"{os.path.basename(path)}: missing"]
    text = open(path, encoding="utf-8").read()
    headings = ("## KV/token derivation", "## KV/token 推导")
    start = next((text.index(h) for h in headings if h in text), None)
    if start is None:
        return [f"{os.path.basename(path)}: derivation section missing"]
    section = text[start:]
    end = section.find("\n## ", 1)
    if end != -1:
        section = section[:end]

    seen = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] not in GEOMETRY or cells[1] in ("Layers", "层数"):
            continue
        model = cells[0]
        layers, kv_heads, head_dim = (int(c) for c in cells[1:4])
        if (layers, kv_heads, head_dim) != GEOMETRY[model]:
            errors.append(
                f"{model}: geometry in the table is "
                f"{layers}/{kv_heads}/{head_dim}, checker expects "
                f"{GEOMETRY[model][0]}/{GEOMETRY[model][1]}/{GEOMETRY[model][2]}"
            )
            continue
        for cell, dtype, column in ((cells[4], "fp16", "fp16"), (cells[5], "fp8", "fp8")):
            got = parse_number(cell)
            if got is None:
                errors.append(f"{model}: no KiB value in the {column} column")
                continue
            want = expected(model, dtype)
            # A cell may show the full-model value, the per-rank value, or both
            # ("120 KiB (15 KiB/rank at TP8)"). Accept either, but nothing else.
            candidates = [want, want / 8] if "rank" in cell.lower() else [want]
            if all(abs(got - c) / c > TOLERANCE for c in candidates):
                errors.append(
                    f"{model} {column}: table says {got:g} KiB, formula says {want:g} KiB"
                )
        seen.add(model)

    missing = set(GEOMETRY) - seen
    if missing:
        errors.append(f"derivation table is missing rows for: {', '.join(sorted(missing))}")
    return errors


def check_main_table(path: str) -> List[str]:
    """Every main-table row for a known model must agree with the formula."""
    errors: List[str] = []
    text = open(path, encoding="utf-8").read()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] not in GEOMETRY:
            continue
        model, kv_cell = cells[0], cells[4]
        dtype = "fp16" if "fp16" in kv_cell else "fp8" if "fp8" in kv_cell else None
        if dtype is None:
            continue
        got = parse_number(kv_cell)
        if got is None:
            continue
        per_rank = "rank" in kv_cell.lower()
        want = expected(model, dtype)
        candidates = [want, want / 8] if per_rank else [want]
        if all(abs(got - c) / c > TOLERANCE for c in candidates):
            errors.append(
                f"{os.path.basename(path)}: {model} ({dtype}{', per rank' if per_rank else ''}) "
                f"says {got:g} KiB, formula says {want:g} KiB"
            )
    return errors


def main() -> int:
    errors = check_derivation_table(MATRIX_EN) + check_derivation_table(MATRIX_ZH)
    errors += check_main_table(MATRIX_EN) + check_main_table(MATRIX_ZH)
    if errors:
        print("MODEL_MATRIX KV/token check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"MODEL_MATRIX KV/token check OK ({len(GEOMETRY)} models re-derived)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
