# Model Matrix

Which models have actually been run on the hardware this guide targets, how much memory they need, and where the evidence lives.

**How to read this table**

- **Weights** are the on-disk checkpoint size, not the per-card footprint. Divide by TP for a first approximation.
- **KV/token** is per token per **full model** unless marked “measured (per rank)”. It depends on layers, KV heads, head dim, and KV dtype; treat it as a budget input, not a constant.
- **Min 72 GB cards** is a floor for weights plus a small KV pool. Add 10–30 GB per card for CUDA context, activations, CUDA graphs, and framework workspaces. A model that “fits” at 0.99 mem fraction usually does not serve well.
- **Evidence** links to a recipe or a benchmark JSON. “Derived” means calculated from the model config, not measured on this box.

| Model | Params (active) | Quantization | Weights | KV/token | Min 72 GB cards | Recommended parallel | Evidence |
|---|---|---:|---:|---:|---:|---|---|
| Qwen3-8B | 8B | BF16 | ~16 GB | ~128 KiB (fp16) | 1 | Single card / replica | Derived |
| Qwen3.8-27B | 27B | BF16 | ~52 GB | ~256 KiB (fp16) | 2 | TP2 within one NUMA node | [Single-card three-machine study](benchmarks/README.md#single-card-bf16-study-across-three-blackwell-skus-2026-10-26) |
| Qwen3.8-27B | 27B | NVFP4 | ~21 GB | ~64 KiB (fp8 KV) | 1 | Single card / replica | Derived |
| Qwen3-32B | 32B | BF16 | ~62 GB | ~256 KiB (fp16) | 2 | TP2/TP4 within one NUMA node | Derived |
| Qwen3-32B | 32B | NVFP4 | ~20 GB | ~64 KiB (fp8 KV) | 1 | Single card / replica | Derived |
| Qwen3-235B-A22B | 235B (22B) | NVFP4 | ~134 GB | ~64 KiB (fp8 KV) | 3–4 | TP4 + EP4; measure vs 2 replicas | Derived |
| GLM-5.3-Flash | ~355B | FP8 | ~306 GB | ~96 KiB (fp8 KV) | 6–8 | TP8 + EP8 | [GLM TP8+EP8 data](benchmarks/README.md#glm-53-flash-fp8-tp8ep8-on-two-96-gb-pcie-hosts-2026-10-22) |
| GLM-5.3-Flash | ~355B | NVFP4 | ~190 GB | ~96 KiB (fp8 KV) | 4–6 | TP4/TP8 + EP; measure | Derived |
| MiniMax-M3.1-Flash | 428B (23B) | MXFP8 + NVFP4 experts | ~252 GB | **~7.5 KiB measured (per rank, fp8)** | 8 | TP8 + Marlin on SM120; TP8 + MegaMoE on SM100 | [8×72 recipe Case C](recipes/8x-72gb-pcie.md#case-c-large-nvfp4-moe-minimax-m31-flash-252-gb--sm120-workaround) · [JSON](benchmarks/data/minimax-m31-flash-8x72-sm120.json) |
| DeepSeek-V4.1-Flash-1017 | 538B-class | FP8 + FP4 experts | ~510 GB (476 GiB) | ~2.6 KiB (MLA, fp8) | 8 | TP8 + EP8; Engram tables on host | Deployment notes in the [related handbook](https://gitee.com/wang-guangshuooo/llm-private-deploy-handbook) |

## What the table is trying to prevent

1. **“It fits on one card” is not the same as “it serves on one card.”** A 52 GB checkpoint on a 72 GB card leaves ~20 GB for KV, activations, and graphs; long context or concurrency can still OOM.
2. **A 252 GB model does not fit on four 72 GB cards.** 63 GB/card of weights leaves 8–12 GB, which is less than the runtime overhead alone. See the [4×72 recipe](recipes/4x-72gb-pcie.md).
3. **MoE is not dense with a different name.** Expert parallelism, router precision, and all-to-all change both the memory layout and the optimal TP/EP split.

## Adding a row

Use the [benchmark report template](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml) or open a PR with a `benchmarks/data/*.json` file following `lltg.benchmark.v1`. If you only have a “does not fit” result, that is still a row — put it in the [4×72 recipe](recipes/4x-72gb-pcie.md) negative column with the OOM log.
