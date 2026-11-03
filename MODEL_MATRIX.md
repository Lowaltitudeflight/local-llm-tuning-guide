# Model Matrix

Which models have actually been run on the hardware this guide targets, how much memory they need, and where the evidence lives.

**How to read this table**

- **Weights** are the on-disk checkpoint size, not the per-card footprint. Divide by TP for a first approximation.
- **KV/token** is per token per **full model** unless marked “per rank”. It depends on layers, KV heads, head dim, and KV dtype; treat it as a budget input, not a constant.
- **Every KV figure is re-derivable** from `2 × layers × kv_heads × head_dim × bytes_per_element` (the 2 is K *and* V). The [derivation block](#kvtoken-derivation) below lists the geometry we checked. Figures marked **†** come from deployment notes without a public config we could re-derive — treat them as a budget input, not a verified number.
- **Min 72 GB cards** is a floor for weights plus a small KV pool. Add 10–30 GB per card for CUDA context, activations, CUDA graphs, and framework workspaces. A model that “fits” at 0.99 mem fraction usually does not serve well.
- **Evidence** links to a recipe or a benchmark JSON. “Derived” means calculated from the model config, not measured on this box.

| Model | Params (active) | Quantization | Weights | KV/token | Min 72 GB cards | Recommended parallel | Evidence |
|---|---|---:|---:|---:|---:|---|---|
| Qwen3-8B | 8B | BF16 | ~16 GB | ~144 KiB (fp16) | 1 | Single card / replica | Derived |
| Qwen3.8-27B | 27B | BF16 | ~52 GB | ~256 KiB (fp16) | 2 | TP2 within one NUMA node | [Single-card three-machine study](benchmarks/README.md#single-card-bf16-study-across-three-blackwell-skus-2026-10-26) |
| Qwen3.8-27B | 27B | NVFP4 | ~21 GB | ~128 KiB (fp8 KV) | 1 | Single card / replica | Derived |
| Qwen3-32B | 32B | BF16 | ~62 GB | ~256 KiB (fp16) | 2 | TP2/TP4 within one NUMA node | Derived |
| Qwen3-32B | 32B | NVFP4 | ~20 GB | ~128 KiB (fp8 KV) | 1 | Single card / replica | Derived |
| Qwen3-235B-A22B | 235B (22B) | NVFP4 | ~134 GB | ~94 KiB (fp8 KV) | 3–4 | TP4 + EP4; measure vs 2 replicas | Derived |
| GLM-5.3-Flash | ~355B | FP8 | ~306 GB | ~96 KiB (fp8 KV) † | 6–8 | TP8 + EP8 | [GLM TP8+EP8 data](benchmarks/README.md#glm-53-flash-fp8-tp8ep8-on-two-96-gb-pcie-hosts-2026-10-22) |
| GLM-5.3-Flash | ~355B | NVFP4 | ~190 GB | ~96 KiB (fp8 KV) † | 4–6 | TP4/TP8 + EP; measure | Derived |
| MiniMax-M3.1-Flash | 428B (23B) | MXFP8 + NVFP4 experts | ~252 GB | **~7.5 KiB derived (per rank, fp8, TP8)** | 8 | TP8 + Marlin on SM120; TP8 + MegaMoE on SM100 | [8×72 recipe Case C](recipes/8x-72gb-pcie.md#case-c-large-nvfp4-moe-minimax-m31-flash-252-gb--sm120-workaround) · [JSON](benchmarks/data/minimax-m31-flash-8x72-sm120.json) |
| DeepSeek-V4.1-Flash-1017 | 538B-class | FP8 + FP4 experts | ~510 GB (476 GiB) | **~785 B measured** · ~2.6 KiB reported † | 8 | TP8 + EP8; Engram tables nominally on host (no VRAM saving observed yet) | [8×72 TP8 matrix](benchmarks/data/deepseek-v41-flash-8x72-tp8.json) · [handbook](https://github.com/Lowaltitudeflight/llm-private-deploy-handbook) |

## KV/token derivation

Formula: `2 × layers × kv_heads × head_dim × bytes_per_element`, where the 2 covers K and V, and `bytes_per_element` is 2 for fp16/bf16 and 1 for fp8.

| Model | Layers | KV heads | head_dim | fp16 | fp8 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 36 | 8 | 128 | 144 KiB | 72 KiB |
| Qwen3.8-27B | 64 | 4 | 256 | 256 KiB | 128 KiB |
| Qwen3-32B | 64 | 8 | 128 | 256 KiB | 128 KiB |
| Qwen3-235B-A22B | 94 | 4 | 128 | 188 KiB | 94 KiB |
| MiniMax-M3.1-Flash | 60 | 4 | 128 | 120 KiB (15 KiB/rank at TP8) | 7.5 KiB/rank at TP8 |

Two traps this table is meant to catch:

- **head_dim is not always 128.** Qwen3.8-27B uses 4 KV heads at head_dim 256, so its KV/token equals an 8-head/128-dim model even though the head count is halved. Deriving one row from another by scaling head count alone is wrong.
- **fp8 halves the element size, not the head count.** If a row's fp16 and fp8 figures differ by 4×, one of them is wrong.

Run `python tools/check_model_matrix.py` to re-derive the published-geometry rows above and fail on a mismatch. It runs in CI.

† GLM-5.3-Flash and DeepSeek-V4.1-Flash-1017 use configs we could not re-derive from a public checkpoint (MLA compression and Engram tables mean the GQA formula does not apply to DeepSeek at all).

For DeepSeek-V4.1-Flash-1017 the two published figures disagree by 3.4× and **both are kept on purpose**:

- `~2.6 KiB` comes from the deployment notes and has no stated denominator (per rank? per model? Engram included?).
- `~785 B` is measured on this box: the server reports `max_total_num_tokens=12,443,136` with `9.10 GiB` available for KV, i.e. `9.10 GiB / 12.44M = 785 B/token`.

The measured number is what the scheduler actually allocates, so it is the one to use for capacity planning on this build; the reported number stays because we cannot reproduce its denominator. Config geometry alone (40 layers × 576 elements × 1 B) would predict ~23 KiB/token, 30× higher than the measured value — this model's compressed attention and Engram layout are not describable by the GQA formula at all. Use `--kv-bytes-per-token` for your own build rather than trusting either figure.

## What the table is trying to prevent

1. **“It fits on one card” is not the same as “it serves on one card.”** A 52 GB checkpoint on a 72 GB card leaves ~20 GB for KV, activations, and graphs; long context or concurrency can still OOM.
2. **A 252 GB model does not fit on four 72 GB cards.** 63 GB/card of weights leaves 8–12 GB, which is less than the runtime overhead alone. See the [4×72 recipe](recipes/4x-72gb-pcie.md).
3. **MoE is not dense with a different name.** Expert parallelism, router precision, and all-to-all change both the memory layout and the optimal TP/EP split.

## Adding a row

Use the [benchmark report template](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml) or open a PR with a `benchmarks/data/*.json` file following `lltg.benchmark.v1`. If you only have a “does not fit” result, that is still a row — put it in the [4×72 recipe](recipes/4x-72gb-pcie.md) negative column with the OOM log.
