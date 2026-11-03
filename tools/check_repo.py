#!/usr/bin/env python3
"""check_repo.py - keep the documentation from rotting.

Three checks that are cheap enough to run on every push:

1. Bilingual parity: every English README/docs/recipes file has its Chinese
   counterpart (and vice versa).
2. Relative links: every local Markdown link points at a file that exists.
3. Benchmark schema: every benchmarks/data/*.json carries lltg.benchmark.v1
   and the fields the planner needs.

Usage: python tools/check_repo.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Iterable, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".github/workflows"}
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.S)

PAIRS = [
    ("README.md", "README.zh-CN.md"),
    ("CONTRIBUTING.md", "CONTRIBUTING.zh-CN.md"),
    ("MODEL_MATRIX.md", "MODEL_MATRIX.zh-CN.md"),
]


def markdown_files() -> Iterable[str]:
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".md"):
                yield os.path.relpath(os.path.join(base, name), ROOT)


def check_pairs() -> List[str]:
    errors = []
    for en, zh in PAIRS:
        for rel in (en, zh):
            if not os.path.exists(os.path.join(ROOT, rel)):
                errors.append(f"missing bilingual file: {rel}")
    # docs chapters
    docs = [f for f in os.listdir(os.path.join(ROOT, "docs")) if f.endswith(".md")]
    for name in docs:
        if name.endswith(".zh-CN.md"):
            continue
        zh = os.path.join("docs", "zh-CN", name)
        if not os.path.exists(os.path.join(ROOT, zh)):
            errors.append(f"missing Chinese chapter: {zh}")
    # recipes
    recipes = [f for f in os.listdir(os.path.join(ROOT, "recipes")) if f.endswith(".md")]
    for name in recipes:
        zh = os.path.join("recipes", "zh-CN", name)
        if not os.path.exists(os.path.join(ROOT, zh)):
            errors.append(f"missing Chinese recipe: {zh}")
    # benchmark pages
    bench_dir = os.path.join(ROOT, "benchmarks")
    if os.path.isdir(bench_dir):
        for name in [f for f in os.listdir(bench_dir) if f.endswith(".md")]:
            zh = os.path.join("benchmarks", "zh-CN", name)
            if not os.path.exists(os.path.join(ROOT, zh)):
                errors.append(f"missing Chinese benchmark page: {zh}")
    return errors


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def _slug(text: str) -> str:
    """Approximate GitHub's heading anchor.

    GitHub lowercases, drops punctuation (including em dashes and full-width
    brackets), turns runs of whitespace into single dashes, and keeps CJK
    characters verbatim.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s", "-", text)
    return text.strip("-")


def _anchors(path: str) -> set:
    with open(path, encoding="utf-8") as fh:
        text = FENCE_RE.sub("", fh.read())
    return {_slug(m.group(2)) for m in HEADING_RE.finditer(text)}


def check_anchors() -> List[str]:
    """A link that points at a real file but a dead anchor is still broken."""
    errors = []
    cache = {}
    for rel in markdown_files():
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as fh:
            text = FENCE_RE.sub("", fh.read())
        for target in LINK_RE.findall(text):
            target = target.strip().split(" ")[0]
            if target.startswith(("http://", "https://", "mailto:")) or "#" not in target:
                continue
            local, _, anchor = target.partition("#")
            if not anchor:
                continue
            resolved = (
                path if not local
                else os.path.normpath(os.path.join(ROOT, os.path.dirname(rel), local))
            )
            if not resolved.endswith(".md") or not os.path.exists(resolved):
                continue
            if resolved not in cache:
                cache[resolved] = _anchors(resolved)
            if _slug(anchor) not in cache[resolved]:
                errors.append(f"{rel}: dead anchor -> {target}")
    return errors


def check_links() -> List[str]:
    errors = []
    for rel in markdown_files():
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as fh:
            text = FENCE_RE.sub("", fh.read())
        for target in LINK_RE.findall(text):
            target = target.strip().split(" ")[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = target.split("#")[0]
            if not local:
                continue
            resolved = os.path.normpath(os.path.join(ROOT, os.path.dirname(rel), local))
            if not os.path.exists(resolved):
                errors.append(f"{rel}: broken link -> {target}")
    return errors


def check_benchmarks() -> List[str]:
    errors = []
    data_dir = os.path.join(ROOT, "benchmarks", "data")
    if not os.path.isdir(data_dir):
        return ["benchmarks/data is missing"]
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(data_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            errors.append(f"{name}: invalid JSON ({exc})")
            continue
        if data.get("schema") != "lltg.benchmark.v1":
            errors.append(f"{name}: schema must be lltg.benchmark.v1")
        for key in ("id", "kind", "title", "hardware", "software", "model",
                    "workload", "results", "transferable", "not_transferable"):
            if key not in data:
                errors.append(f"{name}: missing field {key}")
        if data.get("kind") == "first-party":
            for key in ("variance", "submitted_by"):
                if not data.get(key):
                    errors.append(f"{name}: first-party data needs {key}")
        if not isinstance(data.get("results"), list) or not data["results"]:
            errors.append(f"{name}: results must be a non-empty list")
    return errors


def main() -> int:
    groups = [
        ("bilingual parity", check_pairs()),
        ("relative links", check_links()),
        ("heading anchors", check_anchors()),
        ("benchmark schema", check_benchmarks()),
    ]
    failed = False
    for title, errors in groups:
        if errors:
            failed = True
            print(f"FAIL {title}")
            for err in errors:
                print("  - " + err)
        else:
            print(f"OK   {title}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
