# Recipe: 4 × 72 GB, PCIe interconnect

Target hardware: 4 cards with 72 GB of VRAM (RTX PRO 5000 Blackwell class), no NVLink, 288 GB total. On a dual-socket 4+4 host this is one NUMA node; keep TP inside it.

This recipe is deliberately half negative: on this class of card, **many models that "look like they fit" do not**, because weights plus runtime overhead exceed the per-card budget before KV cache is even allocated.

## What fits and what does not

| Model | Weights | Per card (TP4) | Fits? | Notes |
|---|---:|---:|---|---|
| Qwen3-8B | ~16 GB | ~4 GB | ✅ | Leaves ample KV headroom |
| Qwen3.8-27B BF16 | ~52 GB | ~13 GB | ✅ | Long context and high concurrency both viable |
| Qwen3.8-27B NVFP4 | ~21 GB | ~5 GB | ✅ | Best throughput per watt on this box |
| Qwen3-32B | ~62 GB | ~16 GB | ✅ | Fits with room for KV |
| Qwen3-235B-A22B NVFP4 | ~134 GB | ~34 GB | ✅ | MoE; use TP4 + EP4 where supported, measure |
| GLM-5.3-Flash FP8 | ~306 GB | ~77 GB | ❌ | Exceeds 72 GB before KV/activations |
| MiniMax-M3.1-Flash NVFP4 | ~252 GB | ~63 GB | ❌ | Only ~8-12 GB left for KV + runtime; OOM (see 8×72 recipe Case C) |
| DeepSeek-V4.1-Flash-1017 | ~476 GB | ~119 GB | ❌ | Needs 8 cards |

> Rule of thumb: on a 72 GB card, keep **weights ≤ ~50 GB** if you want long context or real concurrency. The remaining 20+ GB goes to KV, activations, CUDA graphs, and framework workspaces faster than most people expect.

## Topology prerequisites

```bash
nvidia-smi topo -m
```

- If the four cards are on one NUMA node (`NODE` links), TP=4 is the simple fast path.
- If the four cards span two NUMA nodes, prefer 2× TP=2 replicas, or TP=2 × PP=2. Do not let TP cross `SYS`.

## Launch configuration

### Qwen3-32B / Qwen3.8-27B, TP4, single NUMA node

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <model> \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

### Qwen3-235B-A22B NVFP4, TP4 + EP4

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <model> \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
```

> Exact EP flag names differ between vLLM and SGLang; keep the same `--tp 4` and add the framework’s expert-parallel switch. Measure TP4 vs TP2×2 replicas — for a MoE this is not obvious.

## When to use 4 cards instead of 8

- The model fits (table above) and you want the other four cards for a second replica or a different model.
- You care about single-stream latency more than aggregate throughput: a smaller TP group has less collective communication.
- You are power-constrained: 4 × 300 W is easier to cool and supply than 8 × 300 W.

## Negative results are welcome

If a model in the "does not fit" column starts successfully on your 4×72 GB box, that is a useful correction — open an issue with the full launch command, memory logs, and a benchmark JSON. See [CONTRIBUTING.md](../CONTRIBUTING.md).
