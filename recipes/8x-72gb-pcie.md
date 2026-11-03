# Recipe: 8 × 72 GB, PCIe interconnect, dual NUMA

Target hardware: 8 cards with 72 GB of VRAM (RTX PRO 5000 Blackwell class), no NVLink, 576 GB total. 4 cards per CPU socket, cross-socket links are `SYS`.

Environment this recipe was drafted from (first-party snapshot, 2026-10-23):

| Item | Value |
|---|---|
| GPU | 8 × NVIDIA RTX PRO 5000 72GB Blackwell |
| Topology (`nvidia-smi topo -m`) | GPU0–3 `NODE` within NUMA0, GPU4–7 `NODE` within NUMA1, cross-group `SYS` |
| CPU | 2 × Xeon Platinum 8488C (192 threads), GPU0–3 → NUMA0, GPU4–7 → NUMA1 |
| RAM | 1.5 TB |
| Driver / CUDA | 595.71.05 / 13.2 |
| NCCL | 2.26.5 (system) |
| OS | Ubuntu 22.04.5 LTS |

## Applicability boundaries

| Works | Does not work |
|---|---|
| Single-machine serving of large MoE models (hundreds of B parameters, quantized) | Unquantized very large dense models |
| 576 GB total VRAM — room for large models + long context | Scenarios that require scaling across nodes |
| TP=4 within one NUMA node as the fast path | Naive TP=8 that crosses sockets (all-reduce over `SYS` per layer) |

## Topology prerequisites

**Run this first, then decide the parameters:**

```bash
nvidia-smi topo -m
```

Decision rules for this topology (see [02 Parallelism](../docs/02-parallelism.md)):

| Observation | How to set parallelism |
|---|---|
| 4+4 split across two NUMA nodes, cross-group `SYS` | TP=4 within a node × PP=2 across nodes |
| All 8 within one group (not this machine) | TP=8 |
| Model fits in 4 × 72 GB | TP=4 on one NUMA node, leave the other node free |

## Launch configuration

### Case A: TP=4 × PP=2 (uses all 8 cards, cross-socket via PP)

```bash
vllm serve <your-model> \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

| Parameter | Why it is set this way |
|---|---|
| `--tensor-parallel-size 4` | Keep per-layer all-reduce inside one NUMA node (`NODE` links) |
| `--pipeline-parallel-size 2` | Cross-socket traffic only at layer boundaries, far cheaper than TP over `SYS` |
| `--gpu-memory-utilization 0.90` | Leave headroom; do not start at 0.95 |
| `--kv-cache-dtype fp8` | Cheapest VRAM lever at this scale |

SGLang equivalent (the snapshot machine runs SGLang in Docker):

```bash
python -m sglang.launch_server \
  --model-path <your-model> \
  --tp 4 --dp 2 \
  --mem-fraction-static 0.90 \
  --context-length 32768 \
  --host 0.0.0.0 --port 30000
```

> Note: SGLang's `--dp 2` replicates the model across the two NUMA groups (data parallel), which avoids cross-socket all-reduce entirely at the cost of doubled weight memory. Which of PP=2 vs DP=2 wins on this topology is an **open question** — measure both.

### Case B: model fits in 4 × 72 GB (TP=4, single NUMA node)

Use `CUDA_VISIBLE_DEVICES=0,1,2,3` (NUMA0) or `4,5,6,7` (NUMA1) with TP=4 and no PP. Simplest and fastest per-token path.

### Case C: large NVFP4 MoE (MiniMax-M3.1-Flash, 252 GB) — SM120 workaround

A 252 GB model does not fit on four 72 GB cards; it needs all eight. The image's native NVFP4 path targets SM100/SM103, so on SM120 (RTX PRO 5000) it needs three read-only code patches plus explicit flags. Verified on this snapshot machine on 2026-11-02.

```bash
sglang serve \
  --model-path /models2/MiniMax-M3.1-Flash \
  --trust-remote-code \
  --tp 8 \
  --attention-backend triton \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.85 \
  --context-length 1048576 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 128 \
  --host 0.0.0.0 --port 8200
```

Required changes (bind-mount the patched files; do not edit the image):

| File | Change | Why |
|---|---|---|
| `srt/configs/model_config.py` | `uses_minimax_m3_nvfp4_native_stack` returns False on SM120 | The native MSA-v1 decode indexer requires SM100/SM103 |
| `srt/arg_groups/model_overrides/minimax_m3.py` | take the non-Blackwell branch on SM120 | Avoids MegaMoE, which is hard-gated to SM100/103 |
| `srt/layers/quantization/minimax_m3_nvfp4.py` | `use_marlin = not is_blackwell or is_sm120` | Marlin is the only NVFP4 MoE path left on SM120 |

Also: **do not pass `--quantization`** — SGLang must auto-detect `minimax_m3_nvfp4`; forcing `mxfp8` makes the packed NVFP4 experts load as FP8 and fail with a 192-vs-384 tensor-size error.

Measured on this machine (vLLM bench, OpenAI completions, input 1024 / output 256, seed 42, cache-on 3-run mean):

| Configuration | C1 output | C8 aggregate | C32 aggregate | C64 aggregate | ShareGPT C64 |
|---|---:|---:|---:|---:|---:|
| TP8 + Marlin (this recipe) | 94.4 tok/s | 489.8 tok/s | 740.4 tok/s | **1060.8 tok/s** | **1093.8 tok/s** |

Long-context and memory notes:

- KV pool 1,329,280 tokens at `--mem-fraction-static 0.85`; 67.7–72.3 GB used per card.
- A cold 999,988-token request completed in **517.7 s** (1M context enabled) with no OOM; a 1,047,992-token boundary request also succeeded.
- With `--context-length 1048576`, one 1M request occupies ~79% of the KV pool, so concurrency at 1M is effectively 1. The 1M configuration costs ~23% C64 throughput (815.7 tok/s) versus 128K.
- Full data and methodology: [`benchmarks/data/minimax-m31-flash-8x72-sm120.json`](../benchmarks/data/minimax-m31-flash-8x72-sm120.json).

### Case D: DeepSeek-V4.1-Flash-1017 (476 GiB) — TP8 only, and why TP4 cannot exist

DeepSeek-V4.1-Flash-1017 is the largest model this recipe has hosted. It is also the clearest case of "the card count is not a tuning choice":

```
476 GiB of weights / 4 cards = 119 GiB per card
                              > 71.12 GiB usable per card
```

TP4 does not fail slowly — it OOMs while loading `create_fp8_moe_weight_`, at 70.74 GiB in use with 1.05 GiB still requested. See [09 Failed configurations](../docs/09-failures.md) F-17 for the log. **8 cards is the minimum, and the model uses all of them.**

```bash
docker run -d --name ds41 --init --gpus all --network host --ipc host --shm-size 64g   -v /path/to/models:/models2:ro   -e HF_HUB_OFFLINE=1   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   -e NCCL_BUFFSIZE=2097152   -e SGLANG_ENABLE_JIT_DEEPGEMM=1   -e SGLANG_DSV41_TORCH_PREFILL_INDEXER=1   -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1   lmsysorg/sglang:dev-minimax-m31-flash   sglang serve     --model-path /models2/DeepSeek-V4.1-Flash-1017-cfg0     --trust-remote-code     --tp 8 --ep-size 8     --moe-runner-backend flashinfer_mxfp4     --chunked-prefill-size 2048     --max-running-requests 32     --mem-fraction-static 0.85     --cuda-graph-max-bs-decode 32 --cuda-graph-max-bs-prefill 16     --context-length 32768     --host 0.0.0.0 --port 8200
```

Measured (three independent launches, mean; same box, driver 595.71.05):

| Workload | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| short 256/128, output tok/s | 35.6 | 112.4 | *unstable* | 253.9 | **357.5** |
| long 8192/256, output tok/s | 34.1 | 154.5 | **252.9** | 189.2 | 216.6 |

Reading it:

- **Long requests peak at C8 and then fall.** 252.9 → 189.2 at C16 while TTFT p50 goes 385 ms → 1030 ms. Prefill and decode are fighting for the same SMs; this is the case PD separation exists for.
- **Short requests keep climbing to C32**, but that curve is **not reproducible**: C8 produced 71.4, 71.8 and 162.6 tok/s in three launches of the identical command. Do not quote the short-request curve as a configuration result.
- **Single stream is ~34 tok/s** (TPOT 24.8 ms) for both shapes. Concurrency buys aggregate throughput, not per-stream speed.

Two things this model does that the GQA rules in [06 KV cache](../docs/06-kv-cache.md) do not cover:

- **KV is 785 B/token, not the ~23 KiB the formula predicts.** The server reports `max_total_num_tokens=12,443,136` with 9.10 GiB available, so the scheduler allocates 785 B/token — 30× less than `2 × 40 × 576 × 1`. Compressed attention plus Engram tables make the formula inapplicable.
- **Engram offload saves host memory, not VRAM, on this build.** With `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` the log shows `engram host table layer 1: layout=shared, 96681 MiB resident`, but per-card VRAM is unchanged (62,964 MiB) and `avail mem` stays at 10.33 GB. The table is in host RAM *and* still resident on the device.

**Clean rerun with `--disable-radix-cache`:** three launches agree within 0.2-3.4% on all 15 configurations, including C64 and 32K prefill ([clean JSON](../benchmarks/data/deepseek-v41-flash-8x72-tp8-clean.json)). Prefix caching is worth ~34% of long-request throughput at C4 (154.5 -> 101.3) and ~48% at C8, while short requests are unchanged. The C8 short-request level stays a **reproducible dip** (71-72 tok/s) when the cache is off, so it is a property of this stack rather than measurement noise.

Full data: [`benchmarks/data/deepseek-v41-flash-8x72-tp8.json`](../benchmarks/data/deepseek-v41-flash-8x72-tp8.json).

## First-party measurements (2026-10-23, dense 27B)

Measured on the snapshot machine with [sglang 0.5.22](../benchmarks/data/qwen38-27b-8x-72gb-pcie-parallelism.json) — Qwen3.8-27B BF16, input 4096 / output 512 tokens, fresh container per launch:

| Configuration | C1 output | C8 aggregate | TTFT (C1) |
|---|---|---|---|
| TP8 (naive, crosses NUMA) | **75.3 tok/s** | **352.4 tok/s** | 592 ms |
| TP8 + DP-attention 8 | 37.7 tok/s | 233.6 tok/s | 2825 ms |
| TP4 on NUMA0 only | 64.2 tok/s | 314.5 tok/s | 603 ms |

Takeaways (see the JSON's `transferable` / `not_transferable` before reusing):

- **Naive TP8 won.** At this model size, compute scaling beats the cross-socket all-reduce penalty — the "never cross `SYS`" heuristic did not pay off for a dense 27B.
- **DP-attention halved C1 throughput.** At concurrency 1, most DP ranks idle while weights stay partitioned 8 ways. It is an MoE-oriented layout on SM100; on this SM120 box the MiniMax M3.1 path uses TP8 + Marlin (Case C), not DP-attention.
- TP4 on one NUMA node kept **85.3%** of TP8's C1 throughput (64.2 vs 75.3) and 89.3% of its C8 aggregate (314.5 vs 352.4) — with **half the cards**, so its throughput per GPU is **1.7× higher** (16.05 vs 9.41 tok/s/GPU at C1). TP8 buys a single-instance peak; TP4 buys efficiency. Default to TP4 unless one instance's peak matters.

## Known limitations and open questions

| Item | Status |
|---|---|
| Topology, CPU/RAM, driver/CUDA/NCCL values above | **Measured** (first-party snapshot, 2026-10-23) |
| TP8 vs TP4-single-node vs TP8+DP-attention (dense 27B, BF16) | **Measured** — naive TP8 fastest (see table above) |
| TP4 × PP2 vs TP8 for models that need > 4 GPUs of headroom | **Open** — sglang PP was not exercised in this round |
| MoE models on this topology (DP-attention + EP, per the M3.1 launch layout) | **Measured for MiniMax-M3.1-Flash NVFP4 via the Marlin workaround (Case C)**; the DP-attention/EP layout itself remains Open on SM120 |
| NUMA pinning (`numactl --cpunodebind`) benefit | **Open** |
| NCCL tuning (`NCCL_P2P_DISABLE` etc.) on this box | **Open** (see [03 Communication](../docs/03-communication.md)) |
