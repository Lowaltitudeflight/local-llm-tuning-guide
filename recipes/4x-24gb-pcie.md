# Recipe: 4 × 24 GB, PCIe interconnect

Target hardware: 4 consumer cards with 24 GB of VRAM (RTX 4090 / 3090 class), no NVLink, PCIe 4.0 x16 interconnect. This is the most common no-NVLink consumer configuration.

## What makes this class special

Same as [8x-24gb-pcie](8x-24gb-pcie.md): on consumer Ada and later GPUs, **P2P access is restricted at the driver level**, so NCCL can only route through host memory, and TP is extremely inefficient on this class of hardware. **If you can avoid TP, avoid it.**

Compared with 8 cards, 4 cards add one practical constraint: fewer cards means fewer replicas. The actual trade-offs of the three options:

| Option | Prerequisite | Best for |
|---|---|---|
| 4 independent replicas | Model (+KV) fits on a single card | Small models, high-concurrency throughput scenarios |
| 2 replicas × TP=2 | Model fits on two cards | Quantized models around 30B, **the most common case** |
| TP=4 | Only when four cards are strictly required to fit the model | Avoid whenever possible, see "Avoid" |

## Topology prerequisites

```bash
nvidia-smi topo -m
```

On consumer platforms, card pairs usually show `PHB` or `SYS`, and the P2P row is unavailable — this is normal, not a fault. If your platform has a PCIe Switch (`PIX`), TP=2 gains a bit more than on `PHB`, but the strategy is unchanged.

## Launch configuration

### Option A: model fits on two cards (recommended, 2 replicas × TP=2)

```bash
# Replica 1
CUDA_VISIBLE_DEVICES=0,1 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --port 8001

# Replica 2
CUDA_VISIBLE_DEVICES=2,3 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --port 8002
```

Put a load balancer in front.

| Parameter | Why it is set this way |
|---|---|
| `--tensor-parallel-size 2` | Cap TP at 2; beyond that, communication cost starts to outweigh the benefit |
| `--gpu-memory-utilization 0.90` | Consumer cards fragment VRAM more; do not fill it too tightly |
| `--max-model-len 8192` | KV space on a 24 GB card is limited; start with short contexts and scale up using the formulas in [06 KV Cache](../docs/06-kv-cache.md) |
| `--kv-cache-dtype fp8` | On 24 GB cards this is the cheapest VRAM-saving lever |

**Basis**: public comparison data on homogeneous hardware — 4×RTX 3090, AWQ-quantized 35B-class MoE: two instances at TP=2 each reached 1437 tok/s, while a single instance at TP=4 reached only 610 tok/s (see [benchmarks](../benchmarks/README.md), `data/rtx3090-replicas-vs-tp.json`). The **transferable conclusion** is "replicas beat scaling up tensor parallelism"; the specific numbers are not transferable (single measurement, specific model).

### Option B: model fits on a single card (4 independent replicas)

```bash
# Replica 1; for the other replicas change CUDA_VISIBLE_DEVICES and --port
CUDA_VISIBLE_DEVICES=0 vllm serve <your-model> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --port 8001
```

Quantized models below 30B fit comfortably on a single 24 GB card. In that case, the best use of four cards is 4 zero-communication replicas, with throughput adding up roughly linearly.

### Avoid

| Configuration | Reason |
|---|---|
| `--tensor-parallel-size 4` | All 4 cards route through CPU memory; in public data it delivers only 40% of the throughput of TP=2 replicas |
| High `--max-num-seqs` combined with long context | 24 GB cards OOM easily; KV is the first thing to blow up |
| Expecting linear speedup from 4 cards | TP does not hold on this class of hardware; what scales linearly is replicas |

## Known limitations and open questions

| Item | Status |
|---|---|
| P2P restrictions make TP inefficient | **Derivable** (driver restriction + public measurements as supporting evidence) |
| 2 replicas × TP=2 beats TP=4 | **Measured** (public comparison on 4×RTX 3090, not the same card model) |
| Actual TP=2 vs. replica benefit on your platform | **Open** |
| Whether `--max-num-seqs 64` OOMs with your model + context | **Open** (start from 32 and work upward) |

## After you measure

Please record and submit your results following [METHODOLOGY.md](../benchmarks/METHODOLOGY.md). **Especially a direct TP=2-replicas vs. TP=4 comparison on the 4090 (rather than the 3090) — the only public data is from the 3090, so this cell is empty.**
