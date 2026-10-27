# Benchmark Data

This directory holds reproducible performance data. **Every entry must follow the measurement discipline in [METHODOLOGY.md](METHODOLOGY.md).**

## Machine-readable data

External reference data and community submissions are stored as structured JSON in [`data/`](data/) (format described in [data/SCHEMA.md](data/SCHEMA.md)). Tables are for humans, JSON is for tools:

```bash
python tools/planner.py --gpus RTX4090-24 --count 4 --params 35 --quant awq \
    --benchmarks benchmarks/data
```

The planner feeds existing measured data directly into its planning output.

## First-party measurements from this project

| Hardware | Model | Configuration | C1 output | C8 aggregate | Submitter | Link |
|---|---|---|---|---|---|---|
| 8 × RTX PRO 5000 72GB (PCIe, dual NUMA, no NVLink) | Qwen3.8-27B BF16 | TP8 (naive) | 75.3 tok/s | 352.4 tok/s | Lowaltitudeflight | [JSON](data/qwen38-27b-8x-72gb-pcie-parallelism.json) |
| 8 × RTX PRO 5000 72GB (PCIe, dual NUMA, no NVLink) | Qwen3.8-27B BF16 | TP8 + DP-attention 8 | 37.7 tok/s | 233.6 tok/s | Lowaltitudeflight | [JSON](data/qwen38-27b-8x-72gb-pcie-parallelism.json) |
| 8 × RTX PRO 5000 72GB (PCIe, dual NUMA, no NVLink) | Qwen3.8-27B BF16 | TP4 (one NUMA node) | 64.2 tok/s | 314.5 tok/s | Lowaltitudeflight | [JSON](data/qwen38-27b-8x-72gb-pcie-parallelism.json) |

Measured 2026-10-23, input 4096 / output 512 tokens, sglang 0.5.22. Key finding: for this dense model, naive TP8 beat both alternatives — see the JSON's `transferable` / `not_transferable` fields before reusing the ranking. Contributions of measurements from your own machines are welcome — **an empty cell is far more useful than a made-up number**.

### Single-card BF16 study across three Blackwell SKUs (2026-10-26)

Qwen3.8-27B BF16, TP1, vLLM 0.32.0 same digest, input 1024 / output 256 ([JSON](data/qwen38-27b-bf16-singlecard-three-machines.json)):

| Machine | c1 | c8 | c32 | c64 | 32K c1 | ShareGPT c64 |
|---|---:|---:|---:|---:|---:|---:|
| A: RTX 5090 96GB (575 W, ~1792 GB/s) | 29.68 | 177.58 | 434.42 | 559.63 | 14.46 | 900.34 |
| B: RTX PRO 6000 Server 96GB (600 W, ~1597 GB/s) | 26.10 | 182.08 | 491.85 | 697.91 | 16.67 | 933.12 |
| C: RTX PRO 5000 72GB (300 W) | 21.94 | 137.42 | **300.72** | **300.50** | 11.31 | 483.84 |

Key findings: c1 ranks by memory bandwidth; C's 300 W power wall makes c32 and c64 **identical** (hard ceiling); B's compute advantage grows with concurrency (+2.5% → +25%); ShareGPT does not extrapolate from fixed-length results.

### GLM-5.3-Flash FP8, TP8+EP8 on two 96 GB PCIe hosts (2026-10-22)

306 GB FP8 MoE, vLLM 0.32.0 TP8+EP8, max-model-len 131072 ([JSON](data/glm53-flash-fp8-8card-tp8ep.json)):

| Machine | c1 | c8 | c32 | c64 | ShareGPT c64 |
|---|---:|---:|---:|---:|---:|
| A: 8x RTX 5090 96GB | 74.37 | 405.55 | 955.44 | 1226.62 | 1621.41 |
| B: 8x RTX PRO 6000 Server 96GB | 63.99 | 378.19 | 865.24 | 1136.27 | 1462.01 |

Key finding: A leads 7-16% despite B's higher compute — memory bandwidth and small-message all-to-all dominate streaming FP8 MoE decode. After B's BIOS interconnect fix, 1 GiB NCCL busbw is ~99% of A on all four ops; large-message interconnect is **not** the differentiator.

## External reference data

Below are **publicly verifiable third-party measurements**, listed here to anchor parameter choices. **Note that they are results under specific conditions and cannot be extrapolated directly.**

### 1. Speculative decoding: longer draft lengths get slower

| Item | Condition |
|---|---|
| Hardware | 4 × RTX PRO 6000 **Max-Q** (~300 W low-power variant), no NVLink |
| Interconnect | PCIe, TP4 / DCP1 |
| Model | GLM-5.3-Flash, NVFP4-quantized MoE |
| Serving configuration | 4096-token prefill budget, 32 concurrent slots, FP8 KV cache, GPU-only prefix caching |
| Sampling | temperature 1 / top-p 0.95 |
| Measurement | 5 warmed-up 30-second windows per cell; C8 is the aggregate value |

| Mode | Draft tokens | C1 output | C8 aggregate | 32K prefill | Verifier rate |
|---|---|---|---|---|---|
| MTP3 | 3 | **281.3 tok/s** | **897.6 tok/s** | 13,816 tok/s | 112.14 steps/s |
| DFlash2 K7 | 7 | 229.0 tok/s | 702.6 tok/s | 13,955 tok/s | 87.56 steps/s |

**Source**: Local Inference Lab, `local-inference-lab/rtx6kpro`, GLM-5.3-Flash model page.

**Run-to-run variance self-reported by that source**: a separate set of five C1 checks measured MTP3 at 265.7–277.0 tok/s and DFlash2 at 212.9–225.2 tok/s. That is roughly **±5% variance between launches**.

**Transferable conclusion**: on the same hardware, growing the draft from 3 to 7 brought no benefit. **What is not transferable**: the specific numbers, and the judgment that "3 is definitely better than 7" — that depends on the combination of this model, this quantization scheme, and this hardware.

### 2. Without NVLink, replicas beat tensor parallelism

| Item | Condition |
|---|---|
| Hardware | 4 × RTX 3090, no NVLink |
| Model | AWQ-quantized Qwen3.6-35B-A3B |
| Framework | vLLM 0.21.0 |
| Workload | 8192-token context, 128 concurrent requests |

| Configuration | Aggregate throughput |
|---|---|
| Two instances, TP=2 each | **1437 tok/s** |
| Single instance, TP=4 | 610 tok/s |

**Source**: vllm-topology-bench, aggregated via third-party technical write-ups.

**Transferable conclusion**: when the model fits on fewer cards, independent replicas beat scaling up tensor parallelism. **What is not transferable**: the specific numbers (consumer cards, single measurement, specific model).

## Submitting data

See [CONTRIBUTING.md](../CONTRIBUTING.md). When submitting, please:

1. Use the template from [METHODOLOGY.md](METHODOLOGY.md)
2. Include the complete server-side and client-side commands
3. **Include the variance range**, not just your best run
4. Negative results are equally welcome
