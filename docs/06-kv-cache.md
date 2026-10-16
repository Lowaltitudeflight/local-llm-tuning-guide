# 06 KV cache

> Context length is a black hole for VRAM. This chapter explains how it consumes your memory, and what you can do to save it.

## 6.1 First, learn to do the math

The size of the KV cache can be estimated exactly — no trial and error needed:

```
KV bytes per token = 2 × layers × KV heads × head_dim × bytes per element
                       ↑
                 one copy each for K and V
```

**Note that it is the `KV heads`, not the attention heads** — for models using GQA/MQA, the KV head count is far smaller than the attention head count, and that is the key to their memory savings.

### Worked example

Assume a model with 64 layers, 8 KV heads, and head_dim = 128.

| KV precision | Per token | 32K context | 128K context | 1M context |
|---|---|---|---|---|
| BF16 (2 B) | 256 KB | 8 GB | 32 GB | 256 GB |
| FP8 (1 B) | 128 KB | 4 GB | 16 GB | 128 GB |

Look at the last column: **even with the KV cache compressed to FP8, a 1M-token context still needs 128 GB** — more than a single 96 GB card.

This is the gap between "supports 1M context" and "can actually run 1M context": the former is a model capability, the latter depends on your VRAM, and framework documentation usually only mentions the former.

**Basis: derivable**

## 6.2 How to divide the VRAM budget

At inference time, VRAM is split four ways:

```
Total VRAM = model weights + KV cache + activations and workspace + framework overhead
```

`gpu-memory-utilization` (commonly defaulting to 0.90–0.93 in vLLM) controls **how much of the last three can be used**. It is not a knob you can mindlessly turn up:

| Raising it | Lowering it |
|---|---|
| More KV space, longer context | More workspace, more stability |
| Risk: OOM, fragmentation, failures on bursty peaks | Risk: reduced concurrency capacity |

If you are only "slightly short on VRAM", **adjust this parameter or shorten the context first, rather than switching to a more aggressive quantization format** (see [04 Quantization](04-quantization.md)).

**Basis: derivable**

## 6.3 Three ways to save VRAM

| Technique | How it works | Cost | Applies to |
|---|---|---|---|
| **Lower KV precision** | KV from BF16 down to FP8 | Minimal quality loss | Almost every scenario; the first choice |
| **Prefix caching** | Reuses the KV of shared prefixes | Requires extra memory management | Multi-turn conversations, fixed system prompts, RAG |
| **External offloading (LMCache etc.)** | Stores KV in CPU memory or on disk | Retrieval latency | Long sessions, repeated prefixes |

### Prefix caching

Prefix caching pays off enormously in these scenarios:

- Multi-turn conversations (conversation history is a repeated prefix)
- RAG (retrieved document chunks repeat across turns)
- Batched requests with a fixed system prompt

**A pitfall that is easy to hit**: for hybrid-architecture models (those with linear attention or SSM components), the "cache" is not just the KV — there is also a **recurrent state**. Prefix caching for these models is implemented differently than for pure Transformers, so tuning experience from pure-attention models does not carry over. How to tell: check whether the model config has parameters like mamba / SSM / recurrent.

Frameworks usually provide a retention-policy parameter. **In many cases the default is fine** — manually specifying retention intervals can actually break the boundary alignment the framework has already set up.

### External offloading (LMCache)

Offload KV to CPU RAM or disk, trading cheap capacity for VRAM:

```bash
# Conceptual example; the exact parameter name depends on your version
-e CACHE_MODE=lmcache
```

| Pros | Cons |
|---|---|
| Supports durable resumption of very long sessions | Every hit requires a CPU/disk read |
| Dramatically reduces VRAM pressure | Under high concurrency it can become the new bottleneck |

**Note**: video/image requests usually cannot reuse text prefix caches, so offloading strategies offer limited benefit for multimodal requests.

## 6.4 Long context: is DCP worth it

DCP (decode context parallelism) splits the sequence dimension of the KV cache across multiple GPUs, letting a single request use more context than one GPU could hold.

| Prerequisite | How DCP fares |
|---|---|
| NVLink / high-speed RDMA between GPUs | Viable; an important tool for long context |
| **PCIe only between GPUs** | **Limited benefit** — the communication of passing KV around the ring can cancel out the parallelism gains |
| Few GPUs (2–4) | Low parallelism, few ring hops; relatively viable |
| Many GPUs | Ring hops multiply and traffic approaches the full KV size; not worthwhile over PCIe |

**Practical recommendation**:

```
When you need more context, try these in order:
1. Drop KV to FP8                      ← lowest cost
2. Shorten prompts / fewer concurrency slots  ← trade concurrency for length
3. Add GPUs and extend capacity with TP/PP
4. Only then consider DCP              ← least certain payoff over PCIe
```

If you do use DCP, note that it usually must divide the TP degree evenly, and performance at one parallelism degree cannot be extrapolated to another — **a measured throughput number for one configuration does not apply to another**.

**Basis: derivable + measured** (the phenomenon of DCP communication becoming the bottleneck over PCIe appears in multiple public write-ups)

## 6.5 Checklist

Confirm each item before deploying:

- [ ] Computed the KV requirement for your target context using the formula in 6.1
- [ ] Confirmed the actual KV precision configuration (BF16 / FP8 / mixed FP8)
- [ ] `gpu-memory-utilization` leaves enough workspace
- [ ] If the model has SSM / linear-attention components, confirmed prefix caching is a natively supported path in the framework
- [ ] Evaluated DCP's actual benefit for your long-context scenario rather than enabling it by default
- [ ] Measured the actual impact of the concurrency slot count on VRAM

## 6.6 Measurement template

```markdown
### Configuration
- Model / layers / KV heads / head_dim:
- KV precision:
- gpu-memory-utilization:
- Concurrency slots:
- Prefix caching / external offloading: on / off

### Capacity
| Configuration | Available KV tokens | Max runnable context | Measured concurrency limit |
|---|---|---|---|

### Prefix caching effect
| Scenario | TTFT before | TTFT after | Improvement |
|---|---|---|---|
```

Framework startup logs usually **report the total number of available KV tokens directly** — this is the easiest verification there is. Note the number first, then cross-check it against the formula in 6.1.

**Basis: open**

---

Previous chapter: [05 Speculative decoding](05-speculative.md) | Back to [README](../README.md)
