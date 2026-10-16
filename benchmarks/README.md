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

*To be filled. Contributions of measurements from your own machines are welcome — **an empty cell is far more useful than a made-up number**.*

| Hardware | Model | Configuration | C1 output | C8 aggregate | Submitter | Link |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

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
