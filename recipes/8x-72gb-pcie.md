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

## First-party measurements (2026-10-23, dense 27B)

Measured on the snapshot machine with [sglang 0.5.22](../benchmarks/data/qwen38-27b-8x-72gb-pcie-parallelism.json) — Qwen3.8-27B BF16, input 4096 / output 512 tokens, fresh container per launch:

| Configuration | C1 output | C8 aggregate | TTFT (C1) |
|---|---|---|---|
| TP8 (naive, crosses NUMA) | **75.3 tok/s** | **352.4 tok/s** | 592 ms |
| TP8 + DP-attention 8 | 37.7 tok/s | 233.6 tok/s | 2825 ms |
| TP4 on NUMA0 only | 64.2 tok/s | 314.5 tok/s | 603 ms |

Takeaways (see the JSON's `transferable` / `not_transferable` before reusing):

- **Naive TP8 won.** At this model size, compute scaling beats the cross-socket all-reduce penalty — the "never cross `SYS`" heuristic did not pay off for a dense 27B.
- **DP-attention halved C1 throughput.** At concurrency 1, most DP ranks idle while weights stay partitioned 8 ways. It is an MoE-oriented layout (the MiniMax M3.1 launch script on this machine uses it), not a dense-model default.
- TP4 on one NUMA node kept ~89% of TP8 C1 with half the hardware — the right choice when the other 4 GPUs are needed elsewhere.

## Known limitations and open questions

| Item | Status |
|---|---|
| Topology, CPU/RAM, driver/CUDA/NCCL values above | **Measured** (first-party snapshot, 2026-10-23) |
| TP8 vs TP4-single-node vs TP8+DP-attention (dense 27B, BF16) | **Measured** — naive TP8 fastest (see table above) |
| TP4 × PP2 vs TP8 for models that need > 4 GPUs of headroom | **Open** — sglang PP was not exercised in this round |
| MoE models on this topology (DP-attention + EP, per the M3.1 launch layout) | **Open** — MiniMax M3.1-Flash NVFP4 currently cannot start on SM120 (its MSA decode indexer kernel requires SM100/SM103; a port is in progress by the hardware owners) |
| NUMA pinning (`numactl --cpunodebind`) benefit | **Open** |
| NCCL tuning (`NCCL_P2P_DISABLE` etc.) on this box | **Open** (see [03 Communication](../docs/03-communication.md)) |
