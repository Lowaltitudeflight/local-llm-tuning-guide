#!/usr/bin/env python3
"""build_gitee_variant.py - make a Chinese-primary copy for the Gitee mirror.

GitHub is English-primary (README.md English, README.zh-CN.md Chinese).
Gitee is Chinese-primary: README.md becomes Chinese and the English text moves
to README.en.md. The same swap is applied to MODEL_MATRIX and CONTRIBUTING.

Run this in a separate checkout/branch, never on the GitHub main branch:

    python tools/build_gitee_variant.py --root .
"""

from __future__ import annotations

import argparse
import os
import sys

# (English file, Chinese file, English backup name)
PAIRS = [
    ("README.md", "README.zh-CN.md", "README.en.md"),
    ("MODEL_MATRIX.md", "MODEL_MATRIX.zh-CN.md", "MODEL_MATRIX.en.md"),
    ("CONTRIBUTING.md", "CONTRIBUTING.zh-CN.md", "CONTRIBUTING.en.md"),
]

ZH_HEADER = "**简体中文** ｜ [English]({en})\n\n"
EN_HEADER = "**English** ｜ [简体中文]({zh})\n\n"


def strip_language_header(text: str) -> str:
    lines = text.splitlines()
    if lines and "｜" in lines[0] and ("English" in lines[0] or "简体中文" in lines[0]):
        lines = lines[2:] if len(lines) > 1 and not lines[1].strip() else lines[1:]
    return "\n".join(lines).lstrip("\n")


def build(root: str) -> None:
    for en_name, zh_name, en_backup in PAIRS:
        en_path = os.path.join(root, en_name)
        zh_path = os.path.join(root, zh_name)
        if not os.path.exists(zh_path):
            print(f"skip {en_name}: {zh_name} not found")
            continue
        with open(en_path, encoding="utf-8") as fh:
            en_text = fh.read()
        with open(zh_path, encoding="utf-8") as fh:
            zh_text = fh.read()
        en_body = strip_language_header(en_text)
        zh_body = strip_language_header(zh_text)
        with open(os.path.join(root, en_backup), "w", encoding="utf-8") as fh:
            fh.write(EN_HEADER.format(zh=en_name) + en_body + "\n")
        with open(en_path, "w", encoding="utf-8") as fh:
            fh.write(ZH_HEADER.format(en=en_backup) + zh_body + "\n")
        print(f"swapped {en_name} (Chinese primary), English in {en_backup}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Chinese-primary Gitee variant.")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args(argv)
    build(os.path.abspath(args.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
