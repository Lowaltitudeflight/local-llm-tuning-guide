# Recipe: 8 × 24 GB, PCIe interconnect

Target hardware: 8 consumer cards with 24 GB of VRAM (RTX 4090 / 3090 class), no NVLink.

## What makes this class special

On consumer Ada and later GPUs, **inter-GPU P2P access is restricted at the driver level**. The consequences:

- NCCL cannot transfer data directly between cards; it can only **route through host memory** (GPU → CPU → GPU)
- Every all-reduce crosses PCIe twice, degrading both latency and bandwidth
- **TP is extremely inefficient on this class of hardware**, and the benefit of adding cards falls far short of expectations

So the configuration strategy for these machines is completely different from datacenter cards: **if you can avoid TP, avoid it.**

## Topology prerequisites

```bash
nvidia-smi topo -m
```

On this class of hardware, card pairs usually show `PHB` or `SYS`, and the P2P row shows as unavailable. This is normal, not a fault.

## Launch configuration

### First choice: multiple replicas (provided the model fits on one or two cards)

```bash
# Each replica gets one port and one set of cards, serving independently
# Zero communication between replicas; throughput adds up roughly linearly

# Replica 1
CUDA_VISIBLE_DEVICES=0,1 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --port 8001

# Replica 2
CUDA_VISIBLE_DEVICES=2,3 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --port 8002
```

Put a load balancer in front. **This is the best price/performance option on this class of hardware** — see the measured comparison in [02 Parallelism](../docs/02-parallelism.md) (4×RTX 3090: two instances at TP=2 reached 1437 tok/s, a single instance at TP=4 only 610 tok/s).

### Second choice: if you must shard, keep TP as small as possible

```bash
vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
```

| Parameter | Rationale |
|---|---|
| `--tensor-parallel-size 2` | Do not exceed 2; beyond that, communication cost outweighs the benefit |
| `--pipeline-parallel-size 2` | Use PP to extend capacity; communication only happens at segment boundaries |
| `--gpu-memory-utilization 0.90` | Consumer cards fragment VRAM more; do not fill it too tightly |
| `--max-model-len 8192` | KV space on a 24 GB card is limited; start with short contexts |

### Avoid

| Configuration | Reason |
|---|---|
| `--tensor-parallel-size 8` | All 8 cards route P2P through the CPU, almost certainly a net loss |
| High `--max-num-seqs` + long context | OOMs easily on 24 GB cards |
| Expecting linear speedup from 8 cards | Does not hold on this class of hardware |

## Known limitations and open questions

| Item | Status |
|---|---|
| P2P restrictions make TP inefficient | **Derivable** (driver restriction + public measurements as supporting evidence) |
| Multiple replicas beat TP | **Measured** (public comparison data on 4×RTX 3090) |
| Whether your specific card model has P2P | **Open** (confirm by running `nvidia-smi topo -m`) |
| The concrete optimal replica count for 8 cards | **Open** (depends on model size and VRAM) |

**One pragmatic reminder**: if the model is small enough to fit on a single card, do not use multiple cards. A 24 GB card handles quantized models below 30B just fine; in that case the best use of 8 cards is **8 independent replicas**, not one giant parallel instance.

## After you measure

Please record and submit your results following [METHODOLOGY.md](../benchmarks/METHODOLOGY.md).
