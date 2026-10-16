# 02 Parallelism

> Adding GPUs does not necessarily make things faster. This chapter explains when it makes things slower, and how to route around it.

## 2.1 What each of the five parallelism schemes splits

| Scheme | Split dimension | Communication pattern | Traffic |
|---|---|---|---|
| **TP** tensor parallelism | Each layer's weight matrices | One all-reduce per layer | **Largest**, occurs every layer |
| **PP** pipeline parallelism | Layers split into stages | Activations passed at stage boundaries | Small |
| **DP** data parallelism | Full replicas, each serving independently | **Zero communication** at inference time | None |
| **EP** expert parallelism | MoE experts spread across GPUs | all-to-all | Medium to high |
| **DCP** decode context parallelism | The sequence dimension of the KV cache | KV blocks passed around a ring | Grows with parallelism degree |

The key distinction: **TP's communication happens inside every layer**, while PP only communicates at stage boundaries. The deeper the model, the more TP's cumulative communication cost adds up.

**Basis: derivable**

## 2.2 First principle: if it fits on one GPU, don't split it

If the model (including the KV cache for your target context length) fits on a single GPU, **running several independent replicas** is almost always faster than tensor parallelism:

- No communication between replicas at all
- Throughput scales linearly with the number of replicas
- If one replica goes down, the others keep serving

**Measured reference** (4x RTX 3090, no NVLink, AWQ-quantized Qwen3.6-35B-A3B, vLLM 0.21.0, 8K context, 128 concurrent requests):

| Configuration | Aggregate throughput |
|---|---|
| Two instances, each TP=2 | **1437 tok/s** |
| Single instance, TP=4 | 610 tok/s |

Same hardware, same model, a **2.4x gap**.

> ⚠️ This is a single measurement under a single set of conditions (consumer GPUs, 8K context, one specific model). The directional conclusion — "replicas beat splitting across GPUs" — transfers, but the specific numbers do not apply to other hardware or models.

**Basis: measured** (source: vllm-topology-bench, as compiled by third-party technical blogs)

## 2.3 Second principle: TP's boundary is the PCIe Switch

When the model does not fit on one GPU, you have to split it. Then:

```
Does it fit within the GPUs under one PCIe Switch?
├─ Yes  → TP = number of GPUs under that switch
└─ No   → use a TP × PP combination: keep TP inside the switch, let PP cross switches
```

**Why**: every TP step performs an all-reduce. If that step crosses a `SYS` boundary (see [01 Hardware & Topology](01-hardware.md)), it has to traverse the inter-CPU interconnect and latency spikes.

Starting from the example topology in chapter 01 (GPU0/1/2 at `PIX`, GPU3 at `SYS`):

| Configuration | Verdict |
|---|---|
| TP=4 | ❌ Every all-reduce step hits the slowest link |
| TP=3 | ✅ Stays entirely within the switch |
| TP=2 × PP=2 | ✅ TP stays on the fast links; PP only passes activations at stage boundaries |
| Two TP=2 replicas | ✅ Provided the model fits on 2 GPUs |

## 2.4 When "adding GPUs makes it slower"

If any of the following holds, your parallelism has overshot:

- **TP spans more than a PCIe Switch / NUMA domain** — the gains get eaten by communication
- **The model fit on a single GPU to begin with** — adding GPUs introduces communication that never needed to exist
- **Very low concurrency** — communication overhead cannot be amortized across batches, and per-request latency actually rises
- **Forcing TP onto an MoE model's experts** — use EP instead

Diagnostic order: check `nvidia-smi topo -m` first, then the concurrency level, and only then suspect the framework.

## 2.5 MoE models: prefer EP

In MoE models the expert weights dominate. Splitting with TP slices every expert apart and forces an all-reduce every layer; with EP, different GPUs hold different experts and the only communication is a single all-to-all at routing time.

Rules of thumb:

| Situation | Recommendation |
|---|---|
| Total expert count divisible by the GPU count | EP is often the better answer |
| Expert count not divisible by the GPU count | EP creates load imbalance; fall back to TP |
| All experts fit on a single GPU | Just run replicas, don't split |

**Basis: derivable** (for framework implementation details, check the documentation of your specific vLLM / SGLang version)

## 2.6 Decision flow

```
1. Do the model + KV cache fit on one GPU?
   → Yes: run N replicas (DP), done
   → No: continue

2. Work out how many GPUs you need
   → See the VRAM budget formula in chapter 06

3. Are these GPUs under the same PCIe Switch?
   → Yes: TP = GPU count
   → No: TP = GPUs within the switch, fill the rest with PP

4. Is it an MoE model?
   → Yes: evaluate whether EP is feasible (does the expert count divide evenly?)
   → No: done

5. Verify by measurement
   → Record with the templates in benchmarks/ and compare 2-3 candidate configurations
```

## 2.7 Flag quick reference

| Framework | TP | PP | DP | EP |
|---|---|---|---|---|
| vLLM | `--tensor-parallel-size` | `--pipeline-parallel-size` | `--data-parallel-size` | `--enable-expert-parallel` |
| Default | 1 | 1 | 1 | off |

> Flag names and defaults change between framework versions; always check the official documentation for the version you are running.

**Basis: open** (measured comparisons of different parallel configurations on your machine are welcome — please submit them)

---

Previous chapter: [01 Hardware & Topology](01-hardware.md) | Next chapter: [03 Communication](03-communication.md)
