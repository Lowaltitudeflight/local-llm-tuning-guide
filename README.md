**English** ｜ [简体中文](README.zh-CN.md)

[![CI](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# Local LLM Tuning Guide

**A decision library for multi-GPU deployments without NVLink.**

Not an explanation of theory — a set of decisions, each tagged with its source and the conditions under which it holds.

---

## Why this exists

Most self-hosted inference today runs on machines **without NVLink**: RTX 4090 / 5090, RTX PRO 6000, PCIe variants of datacenter cards, connected over PCIe 5.0 x16 or a PCIe switch.

Yet the documentation, tuning advice, and default configurations of mainstream inference frameworks are almost all written under the assumption that GPUs talk over a fast interconnect. The consequences:

- The same four GPUs can differ by more than 2× in throughput (community reports; see [benchmarks](benchmarks/)) — the difference is configuration, not silicon
- Articles explaining the principles are easy to find, but **their parameter recommendations contradict each other and rarely state test conditions**
- You cannot tell whether a piece of advice applies to your machine

This project supplies the missing half: **binding each conclusion to the conditions under which it is valid.**

## Who this is for

- You run 2–8 GPUs on one box, over PCIe, without NVLink.
- You are choosing TP / PP / EP / replicas, quantization, or KV settings and want a starting point with its evidence attached.
- You have a benchmark number and need to know whether it transfers to your machine.

**Not for**: multi-node fabrics (InfiniBand, NVSwitch), training, or “which model is smartest” rankings. The guide is about serving configuration, not model quality.

## Start here

1. Run `python tools/detect_topology.py` and keep the JSON — it is the input to every later decision.
2. Open the [model matrix](MODEL_MATRIX.md) to see whether your model fits, and on how many cards.
3. Pick the closest [recipe](recipes/), copy the flags, then measure with [bench/run_matrix.py](bench/run_matrix.py) instead of trusting the defaults.

## Quick decision table

Six highest-frequency questions, answered first.

| Your situation | Recommendation | Basis |
|---|---|---|
| Model (plus KV) fits on one GPU | **Prefer replicas over tensor parallelism** | Measured + derivable |
| Model must be split; all GPUs under one PCIe switch | **Start with TP = GPUs under that switch, then measure.** On the 8×72 box TP8 beat TP4-on-one-node by 12% aggregate — but it used **2× the cards**; TP4 was **1.7× better per GPU** | Measured |
| GPUs span PCIe switches or CPU sockets | **Avoid letting TP cross that boundary by default** — use TP × PP or replicas. The same box measured one exception (TP8 across SYS still beat TP4-on-node); treat this as a default, not a law | Derivable + measured |
| You want a longer context | Lower KV precision or add GPUs first; DCP gains little over PCIe | Measured + derivable |
| You want speculative decoding for speed | **Longer drafts are not automatically better** — start at 2–3. The published 19–22% gap compares MTP3 with DFlash2-K7, which differ in draft method too | Measured (confounded) |
| Quantization hurts output quality | **Raise activation precision first** — W4A4 → W4A16 took checksum failures from 5.9% to 0. FP32 router weights are a free second safeguard | Measured |

## Contents

| Chapter | Topic | In one line |
|---|---|---|
| [01 Hardware & topology](docs/01-hardware.md) | The real gap between PCIe and NVLink | Find out how your GPUs are actually connected |
| [02 Parallelism](docs/02-parallelism.md) | TP / PP / DP / EP / DCP | When adding a GPU makes things slower |
| [03 Communication](docs/03-communication.md) | All-reduce modes and NCCL tuning | Why PCIe prefers one-shot |
| [04 Quantization](docs/04-quantization.md) | FP8 / MXFP8 / NVFP4 / AWQ trade-offs | What you pay for the memory you save |
| [05 Speculative decoding](docs/05-speculative.md) | MTP / DFlash / EAGLE families | Why guessing more can be slower |
| [06 KV cache](docs/06-kv-cache.md) | Memory budget, prefix cache, long context | How context eats your VRAM |
| [07 Troubleshooting](docs/07-troubleshooting.md) | Symptom → cause → fix | OOM, bad output, SM120 kernel gaps, hung containers |
| [08 Power & cost](docs/08-power-cost.md) | Tokens per watt and per dollar | Why a 300 W card stops scaling |
| [09 Failed configurations](docs/09-failures.md) | What does not work, and why | The negative half of the library |
| [Model matrix](MODEL_MATRIX.md) | What fits on 72 GB cards | Weights, KV/token, minimum GPU count, verified configs |
| [Recipes](recipes/) | Ready-to-use configs by hardware | [4×72](recipes/4x-72gb-pcie.md) · [8×72](recipes/8x-72gb-pcie.md) · [4×96](recipes/4x-96gb-pcie.md) · [8×24](recipes/8x-24gb-pcie.md) |
| [Benchmarks](benchmarks/) | Data tables and measurement method | Fill the table with your own machine |
| [Planner](tools/) | Hardware in, parallel strategy out | [CLI + offline web tool](tools/) · [try the web tool online](https://lowaltitudeflight.github.io/local-llm-tuning-guide/) |
| [Glossary](docs/GLOSSARY.md) | Terms and abbreviations | TP / PP / EP / MLA / NVFP4 / SM120 |
| [Articles](articles/) | Long-form write-ups | English abstracts + Chinese full text |

Every chapter is also available in [简体中文](docs/zh-CN/) — as are the
[recipes](recipes/zh-CN/), the [benchmark pages](benchmarks/zh-CN/), and
[CONTRIBUTING](CONTRIBUTING.zh-CN.md).

## Evidence grading

Every conclusion carries a basis tag:

| Tag | Meaning |
|---|---|
| **Measured** | Produced by a run on real hardware. Hardware, software versions and variance are recorded; new submissions must also record clock policy, image digest and the exact server/client commands (see [SCHEMA](benchmarks/data/SCHEMA.md)). Entries submitted before 2026-11 lack the last three and are marked as such — they were run, but are not yet fully reproducible |
| **Derivable** | Calculated from a model config, hardware specification or framework mechanic; independent of any single measurement. In tables, a derivable figure says so (for example the KV/token derivation block in [MODEL_MATRIX](MODEL_MATRIX.md)) |
| **Open** | Direction is clear but public data is missing; we provide a measurement method and leave the cell to be filled |

**What this project does not do: invent numbers.** An empty cell is worth more than a fabricated one.

## Key conclusions at a glance

Six of the highest-confidence judgements, expanded in the chapters.

1. **If it fits on one GPU, do not use tensor parallelism.** Two independent replicas have zero inter-GPU communication and scale throughput linearly; TP all-reduces at every layer, and over PCIe that cost eats most of the gain.
2. **The boundary for TP is the PCIe switch, not the GPU count.** Traffic crossing switches goes through CPU memory with a sharp latency penalty. With four GPUs split across two switches, TP=4 is often worse than TP=2 × PP=2 — sometimes worse than two TP=2 replicas.
3. **Over PCIe, prefer one-shot all-reduce.** The two-shot scheme is superior on a fast interconnect, but that superiority assumes ample bandwidth. PCIe is the opposite case.
4. **Speculative decoding gains are bandwidth-bound.** A longer draft wastes more compute per failed verification, and the draft model itself consumes bandwidth. The published comparison (MTP3 vs DFlash2-K7) shows the shorter draft ahead by 19–22%, but the two configurations also differ in draft method and weights — read it as "measure both", not "7 is always worse than 3".
5. **Raise activation precision before touching the router.** In MoE models, W4A4 → W4A16 alone took greedy-checksum failures from 5.9% to 0. Keeping FP32 router weights costs no measurable throughput and is the difference on pre-QAD checkpoints (19.5% → 4.0%), but it does not substitute for A16 activations.
6. **On one machine, configuration alone accounts for more than a 2× throughput difference.** This is not folklore — every parameter is either aligned with your real topology or fighting it.

## How to use this

**Deploying a new machine**: start with [01 Hardware & topology](docs/01-hardware.md), run `nvidia-smi topo -m` once, then find the closest match in [recipes](recipes/).

**Machine already running but slow**: run the planner and diff its parameters against your current ones.

```bash
python tools/planner.py --gpus A800-80 --count 8 --model qwen2.5-72b --ctx 8192
```

It reports the interconnect ceiling, the memory budget, and a TP / PP / replica
layout — each tagged with an evidence grade. See [tools/README.md](tools/README.md).

**Want to contribute data**: see [CONTRIBUTING.md](CONTRIBUTING.md). The tables are ready; what is missing is measurements from real machines.

## Related projects

- [llm-inference-bench](https://github.com/Lowaltitudeflight/llm-inference-bench) — the companion benchmarking platform that produces the data this project consumes
- [llm-private-deploy-handbook](https://github.com/Lowaltitudeflight/llm-private-deploy-handbook) — delivery process handbook for private deployments

## License

[MIT](LICENSE)
