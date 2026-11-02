# Data and documentation licenses

This repository contains three kinds of material, with different licenses.

## 1. Code and prose documentation

The Python tools, HTML generator, tests, and all Markdown prose are licensed
under the repository's [MIT License](LICENSE).

## 2. First-party benchmark data

The JSON files in `benchmarks/data/` whose `"kind"` is `"first-party"`, and the
measurement tables derived from them, are licensed under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**.

You may copy, redistribute, and adapt them for any purpose, including
commercially, provided you give appropriate credit:

> Local LLM Tuning Guide, first-party benchmark data, CC BY 4.0.
> https://github.com/Lowaltitudeflight/local-llm-tuning-guide

## 3. Third-party reference data

Entries whose `"kind"` is `"external-reference"` are summaries of measurements
published by other projects. Their `source` field names the original project
and URL. **Those rows remain subject to the original publisher's terms**; this
repository's CC BY 4.0 does not relicense them. When reusing a third-party row,
cite the original source as well.

## What is not covered

- Model weights, container images, and framework binaries are not distributed
  by this repository and carry their own licenses.
- Vendor names and trademarks belong to their respective owners.
