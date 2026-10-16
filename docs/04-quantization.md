# 04 Quantization

> Quantization is a trade between VRAM and output quality. This chapter explains where the cost of that trade actually lands, and how to verify you didn't lose more than you gained.

## 4.1 Format comparison

The table below describes only the relative size of the **weights**; actual VRAM usage also includes the KV cache, activations, and runtime overhead.

| Format | Weight quantization | Activation quantization | Approx. weight share | Hardware requirement |
|---|---|---|---|---|
| BF16 / FP16 | 16 bit | 16 bit | 100% (baseline) | Everything |
| FP8 (E4M3/E5M2) | 8 bit | 8 bit | ~50% | Hopper and newer |
| MXFP8 | 8 bit (microscaling) | 8 bit | ~50% | Native on Blackwell |
| AWQ / GPTQ | 4 bit | 16 bit | ~30% | Anything with the matching kernels |
| W4A8 | 4 bit | 8 bit | ~30% | Depends on kernel support |
| NVFP4 | 4 bit | 4 bit | ~25% | Native on Blackwell |

**The trend**: fewer bits save more VRAM, but the more aggressively activations are quantized, the greater the risk to output stability. `W4A16` (4-bit weights, 16-bit activations) is usually the most conservative of the aggressive options.

**Basis: derivable**

## 4.2 MoE models: protect the router first

MoE models carry a specific risk: **when the router picks the wrong expert, the output goes off the rails entirely**, and this failure is not masked by the "average error" of weight quantization — it is discrete and discontinuous.

Published research (Local Inference Lab's quantization report on GLM-5.3-Flash) specifically compared:

- Three activation precisions: **W4A4 / W4A8 / W4A16**
- The effect of keeping **FP32 router weights**
- The metrics of interest were **greedy checksum stability** (does the same input reliably produce the same output) and **the speed cost**

Its conclusion points one way: the more aggressively activations are quantized, the harder output determinism becomes to guarantee for MoE; keeping the router weights at higher precision is a very cost-effective step.

> See the original report for the concrete numbers; this project does not republish unverified figures.

**Basis: measured** (source: local-inference-lab/rtx6kpro, the `checksum-stability-moe-activations` report)

**Practical recommendations**:

| Priority | Action |
|---|---|
| 1 | Keep router weights in FP32, or at least BF16 |
| 2 | Expert weights can be quantized more aggressively |
| 3 | Start activations at A16, and only lower it after confirming no loss |

## 4.3 How to verify quantization didn't cost you

Quantization quality cannot be judged by "does it run". Three levels of verification, from shallow to deep:

| Level | Method | Question it answers |
|---|---|---|
| Distribution level | **KLD** (KL divergence) | How far the post-quantization output distribution has drifted from the original model |
| Behavior level | **Repeat / mutated-prefix checks, checksum stability** | Does the same input reliably produce the same output |
| Task level | **Real task benchmarks** | Did quality drop in the scenario you actually care about |

**How to read KLD correctly**: it is a regression and quality-control tool, **not a complete quality metric**. A good KLD does not mean good task performance, but a worsening KLD almost certainly means something is wrong. Pair it with concrete probes such as long-context decoding, code completion, and acceptance-rate checks.

**One check that is easy to overlook**: **output determinism**. Run the same prompt twice with greedy decoding; the outputs should be byte-identical. If they diverge after quantization, something in the numerical path has broken — and this is especially common on MoE models.

**Basis: derivable + measured** (the KLD methodology follows the published approach of local-inference-lab/rtx6kpro)

## 4.4 Selection decision flow

```
1. Does the hardware support a native 4-bit format (Blackwell)?
   ├─ Yes → NVFP4 is worth evaluating, but starting at W4A8 is safer
   └─ No  → use AWQ/GPTQ (W4A16), keep activations at 16 bit

2. Is it an MoE model?
   ├─ Yes → do not quantize the router weights along with everything else
   └─ No  → standard strategy

3. Is VRAM sufficient?
   ├─ Yes → prefer raising KV precision or context length over more aggressive weight quantization
   └─ No  → then lower the weight precision

4. Verify
   → KLD + determinism checks + your own task benchmarks
```

**One rule of thumb**: if you are only slightly short on VRAM, adjust `gpu-memory-utilization` or shorten the context rather than changing the quantization format. A change of quantization format is global in effect and costs far more than tuning a few parameters.

## 4.5 Measurement template

```markdown
### Quantization configuration
- Baseline model:
- Quantization format (weights/activations):
- Router precision:
- Quantization tool and version:

### KLD
| Comparison | KLD | Sample size | Context length |
|---|---|---|---|

### Determinism
- Greedy repeat output identical: yes / no
- Mutated-prefix test: pass / fail

### Task benchmarks
| Task | Baseline score | Quantized score | Change |
|---|---|---|---|

### Speed cost
| Configuration | C1 output | C8 aggregate |
|---|---|---|
```

**Basis: open**

---

Previous chapter: [03 Communication](03-communication.md) | Next chapter: [05 Speculative decoding](05-speculative.md)
