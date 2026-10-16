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

- The same four GPUs produce 1,400 tok/s for one person and 600 tok/s for another — the difference is entirely configuration
- Articles explaining the principles are easy to find, but **their parameter recommendations contradict each other and rarely state test conditions**
- You cannot tell whether a piece of advice applies to your machine

This project supplies the missing half: **binding each conclusion to the conditions under which it is valid.**

## Quick decision table

Six highest-frequency questions, answered first.

| Your situation | Recommendation | Basis |
|---|---|---|
| Model (plus KV) fits on one GPU | **Prefer replicas over tensor parallelism** | Measured + derivable |
| Model must be split; all GPUs under one PCIe switch | TP = number of GPUs under that switch | Derivable |
| GPUs span PCIe switches or CPU sockets | **Do not let TP cross that boundary** — use TP × PP instead | Derivable |
| You want a longer context | Lower KV precision or add GPUs first; DCP gains little over PCIe | Measured + derivable |
| You want speculative decoding for speed | **Longer drafts are not better** — start at 2–3 and measure | Measured |
| Quantization hurts output quality | Check whether the MoE router was quantized along with the experts | Measured |

## Contents

| Chapter | Topic | In one line |
|---|---|---|
| [01 Hardware & topology](docs/01-hardware.md) | The real gap between PCIe and NVLink | Find out how your GPUs are actually connected |
| [02 Parallelism](docs/02-parallelism.md) | TP / PP / DP / EP / DCP | When adding a GPU makes things slower |
| [03 Communication](docs/03-communication.md) | All-reduce modes and NCCL tuning | Why PCIe prefers one-shot |
| [04 Quantization](docs/04-quantization.md) | FP8 / MXFP8 / NVFP4 / AWQ trade-offs | What you pay for the memory you save |
| [05 Speculative decoding](docs/05-speculative.md) | MTP / DFlash / EAGLE families | Why guessing more can be slower |
| [06 KV cache](docs/06-kv-cache.md) | Memory budget, prefix cache, long context | How context eats your VRAM |
| [Recipes](recipes/) | Ready-to-use configs by hardware | Copy and run |
| [Benchmarks](benchmarks/) | Data tables and measurement method | Fill the table with your own machine |
| [Planner](tools/) | Hardware in, parallel strategy out | [CLI + offline web tool](tools/) · [try the web tool online](https://lowaltitudeflight.github.io/local-llm-tuning-guide/) |

Every chapter is also available in [简体中文](docs/zh-CN/) — as are the
[recipes](recipes/zh-CN/), the [benchmark pages](benchmarks/zh-CN/), and
[CONTRIBUTING](CONTRIBUTING.zh-CN.md).

## Evidence grading

Every conclusion carries a basis tag:

| Tag | Meaning |
|---|---|
| **Measured** | Publicly verifiable measurement, with hardware, clocks, software versions and client command |
| **Derivable** | Follows directly from hardware specifications or framework mechanics; independent of any single measurement |
| **Open** | Direction is clear but public data is missing; we provide a measurement method and leave the cell to be filled |

**What this project does not do: invent numbers.** An empty cell is worth more than a fabricated one.

## Key conclusions at a glance

Six of the highest-confidence judgements, expanded in the chapters.

1. **If it fits on one GPU, do not use tensor parallelism.** Two independent replicas have zero inter-GPU communication and scale throughput linearly; TP all-reduces at every layer, and over PCIe that cost eats most of the gain.
2. **The boundary for TP is the PCIe switch, not the GPU count.** Traffic crossing switches goes through CPU memory with a sharp latency penalty. With four GPUs split across two switches, TP=4 is often worse than TP=2 × PP=2 — sometimes worse than two TP=2 replicas.
3. **Over PCIe, prefer one-shot all-reduce.** The two-shot scheme is superior on a fast interconnect, but that superiority assumes ample bandwidth. PCIe is the opposite case.
4. **Speculative decoding gains are bandwidth-bound.** A longer draft wastes more compute per failed verification, and the draft model itself consumes bandwidth. On a PCIe host, three draft tokens can be meaningfully faster than seven.
5. **Protect the router when quantizing.** In MoE models the expert weights can be quantized aggressively, but a router that picks the wrong expert destroys the output. Keeping the router at higher precision costs far less than it saves.
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

- [llm-inference-bench](https://gitee.com/wang-guangshuooo/llm-inference-bench) — the companion benchmarking platform that produces the data this project consumes
- [llm-private-deploy-handbook](https://gitee.com/wang-guangshuooo/llm-private-deploy-handbook) — delivery process handbook for private deployments

## License

[MIT](LICENSE)
