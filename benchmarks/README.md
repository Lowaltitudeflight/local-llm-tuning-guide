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

Measured 2026-10-23, input 4096 / output 512 tokens, sglang 0.5.22. Key finding: for this dense model, naive TP8 had the highest single-instance throughput — but it used **8 cards against TP4's 4**, and TP4 was 1.7× better per GPU. See the JSON's per-config notes before reusing the ranking — see the JSON's `transferable` / `not_transferable` fields before reusing the ranking. Contributions of measurements from your own machines are welcome — **an empty cell is far more useful than a made-up number**.

### Single-card BF16 study across three Blackwell SKUs (2026-10-26)

Qwen3.8-27B BF16, TP1, vLLM 0.32.0 same digest, input 1024 / output 256 ([JSON](data/qwen38-27b-bf16-singlecard-three-machines.json)):

| Machine | c1 | c8 | c32 | c64 | 32K c1 | ShareGPT c64 |
|---|---:|---:|---:|---:|---:|---:|
| A: RTX 5090 96GB (modded, not a retail 32 GB card; 575 W, ~1792 GB/s) | 29.68 | 177.58 | 434.42 | 559.63 | 14.46 | 900.34 |
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

### MiniMax-M3.1-Flash NVFP4, TP8 + Marlin on 8x RTX PRO 5000 72GB (SM120, 2026-11-02)

252 GB checkpoint, MXFP8 dense + NVFP4 packed experts, served through the SM120 Marlin workaround (see [8x-72 recipe Case C](../recipes/8x-72gb-pcie.md#case-c-large-nvfp4-moe-minimax-m31-flash-252-gb--sm120-workaround)). vLLM bench, OpenAI completions, input 1024 / output 256, seed 42, cache-on 3-run mean ([JSON](data/minimax-m31-flash-8x72-sm120.json)):

| Configuration | C1 output | C8 aggregate | C32 aggregate | C64 aggregate | ShareGPT C64 |
|---|---:|---:|---:|---:|---:|
| TP8 + Marlin (SM120) | 94.4 | 489.8 | 740.4 | **1060.8** | **1093.8** |
| Prefix cache off (1 run; all concurrency levels) | — | 383.3 | 588.3 | **821.0** | 1084.1 |

Key findings:

- Prefix cache is load-bearing: disabling it cut c8-c64 throughput by **20.5%-22.6%** and raised TTFT p50 by 3.7-5.8x; `ctx32k_c4` lost **41.9%**.
- 1M context works: a cold **999,988-token** request finished in **517.7 s** with no OOM; a 1,047,992-token boundary request also succeeded. One 1M request consumes ~79% of the 1,329,280-token KV pool, so 1M concurrency is effectively 1.
- The 1M configuration costs ~23% C64 throughput (815.7 tok/s) because prefill graphs are disabled and chunked prefill drops to 2048.

### DeepSeek-V4.1-Flash-1017 (8x RTX PRO 5000 72GB, TP8+EP8, 2026-11-03)

476 GiB checkpoint (FP8 weights + FP4 experts), TP8 + EP8, three independent launches.
Short = 256 input / 128 output, long = 8192 / 256; means of three runs ([JSON](data/deepseek-v41-flash-8x72-tp8.json)):

| Workload | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| short, output tok/s | 35.6 | 112.4 | *unstable* | 253.9 | **357.5** |
| long, output tok/s | 34.1 | 154.5 | **252.9** | 189.2 | 216.6 |

- **Long requests peak at C8 and fall afterwards** (252.9 -> 189.2 at C16, TTFT p50 385 -> 1030 ms): prefill and decode contend for the same SMs.
- **The short-request curve is not reproducible**: C8 produced 71.4 / 71.8 / 162.6 tok/s in three launches of the identical command (128% spread), while long-request levels held within 0.1-3.2%. Do not quote it as a configuration result.
- **TP4 is impossible for this checkpoint**: 476 GiB / 4 = 119 GiB per card against 71.12 GiB usable; the load OOMs at 70.74 GiB (see [09 Failed configurations](../docs/09-failures.md) F-17).
- **KV is 785 B/token on this build** (`max_total_num_tokens=12,443,136`, 9.10 GiB), not the ~23 KiB the GQA formula predicts.

**Clean rerun, prefix cache disabled (2026-11-03).** Three independent launches agree within **0.2-3.4%** on all 15 configurations ([JSON](data/deepseek-v41-flash-8x72-tp8-clean.json)):

| Workload | C1 | C4 | C8 | C16 | C32 | C64 |
|---|---:|---:|---:|---:|---:|---:|
| short 256/128 | 35.8 | 107.7 | **71.7** | 242.1 | 350.7 | **372.0** |
| long 8192/256 | 34.1 | 101.3 | 131.5 | 184.7 | 211.9 | 218.9 |
| 32K prefill (c1/c2/c4) | 15.1 | 19.7 | 22.7 | — | — | — |

- **C8 is a reproducible dip, not noise**: 70.6-72.3 tok/s in all three launches, below C4 (107.7) and far below C16 (242.1). With the cache disabled the level is stable, so the 128% spread in the cache-on dataset came from cache state, not from scheduling.
- **Prefix caching is worth ~34% of long-request throughput at C4** (154.5 -> 101.3) and **~48% at C8** (255.1 -> 131.5); short requests are unaffected (351 -> 351 at C32).
- **32K prefill sets the ceiling**: 15.1 tok/s at C1 with a 1.9 s TTFT, 22.7 tok/s at C4.



Throughput alone does not tell you what a user feels. The same MiniMax-M3.1-Flash run recorded TTFT and TPOT at every level:

| Level | Aggregate output (tok/s) | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 |
|---|---:|---:|---:|---:|---:|
| c1 | 94.4 | 195 ms | 210 ms | 9.9 ms | 9.9 ms |
| c8 | 489.8 | 227 ms | 368 ms | 15.5 ms | 16.3 ms |
| c32 | 740.4 | 800 ms | — | 35.1 ms | — |
| c64 | 1060.8 | 1579 ms | — | 40.8 ms | — |
| 32K input, c1 | 31.1 | 5632 ms | 5821 ms | 11.2 ms | 11.7 ms |

How to read it:

- **TTFT is queueing + prefill.** From c8 to c64 TTFT grows 7× while TPOT grows 2.6×. The queue is what hurts, not the decoder.
- **TPOT grows because the GPU is shared.** Decode is bandwidth-bound; more sequences per step means each sequence gets less of the step. 9.9 → 40.8 ms is the price of 8× aggregate throughput.
- **Goodput is the number to quote against an SLO.** If the SLO is "TTFT ≤ 500 ms", this box supports roughly c8-c16, not c64 — even though c64 has the highest throughput. Decide `--max-num-seqs` from the SLO table below, not from the throughput peak.

| SLO | Maximum level that still meets it |
|---|---|
| TTFT p99 ≤ 400 ms | c8 |
| TTFT p99 ≤ 1 s | c32 |
| TTFT < 2 s | c64 |
| 32K prompt, TTFT < 6 s | c1 only |

> The c64 row is why "throughput went up" is not a capacity plan. At c64 the average request waits 1.6 s before its first token — fine for batch summarisation, unacceptable for chat.

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
