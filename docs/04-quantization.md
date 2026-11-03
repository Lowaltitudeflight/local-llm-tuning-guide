# 04 Quantization

> Quantization is a trade between VRAM and output quality. This chapter explains where the cost of that trade actually lands, and how to verify you didn't lose more than you gained.

## 4.1 Format comparison

The table below describes only the relative size of the **weights**; actual VRAM usage also includes the KV cache, activations, and runtime overhead.

| Format | Weight quantization | Activation quantization | Approx. weight share | Hardware requirement |
|---|---|---|---|---|
| BF16 / FP16 | 16 bit | 16 bit | 100% (baseline) | Everything |
| FP8 (E4M3/E5M2) | 8 bit | 8 bit | ~50% | Ada (SM89) and newer — A100 (SM80) has no FP8 tensor cores |
| MXFP8 | 8 bit (microscaling) | 8 bit | ~50% | Native on Blackwell |
| AWQ / GPTQ | 4 bit | 16 bit | ~30% | Anything with the matching kernels |
| W4A8 | 4 bit | 8 bit | ~30% | Depends on kernel support |
| NVFP4 | 4 bit | 4 bit | ~25% | Native on Blackwell |

**The trend**: fewer bits save more VRAM, but the more aggressively activations are quantized, the greater the risk to output stability. `W4A16` (4-bit weights, 16-bit activations) is usually the most conservative of the aggressive options.

**Basis: derivable**

## 4.2 MoE models: activations first, router second

MoE models carry a specific risk: **when the router picks the wrong expert, the output goes off the rails entirely**, and this failure is not masked by the "average error" of weight quantization — it is discrete and discontinuous.

Local Inference Lab's quantization report on GLM-5.3-Flash isolates the two levers people usually conflate. The metric is **greedy checksum stability**: the same prompt, temperature 0, does the answer hash stay the same.

| Change | Greedy checksum failure rate |
|---|---|
| QAD checkpoint, W4A4 activations (framework default) | **5.9%** (41/700) |
| QAD checkpoint, **W4A16 activations** (`VLLM_B12X_MOE_FP4_FORCE_A16=1`) | **0%** (0/500) |
| QAD checkpoint, W4A16 + **FP32 router** | 0% (n=500) — router not needed here, and no measurable runtime cost |
| pre-QAD checkpoint, W4A16, BF16 router | 19.5% |
| pre-QAD checkpoint, W4A16, **FP32 router** | **4.0%** (n=200) |

Two conclusions, and the order matters:

1. **Activation precision is the primary lever.** Moving from W4A4 to W4A16 alone took the failure rate from 5.9% to 0. If you serve an MoE model with 4-bit activations and see non-deterministic output, raise activations to A16 *first* — keeping a high-precision router while leaving activations at A4 does not fix it.
2. **FP32 router weights are a free secondary safeguard, but checkpoint-dependent.** On the QAD checkpoint they changed nothing (0/500 either way); on the pre-QAD checkpoint they cut 19.5% → 4.0%. They add no measurable throughput cost: the router-weight multiply moves from the FC2 epilogue into the FP32 top-k sum, and the arithmetic is the same.

**Basis: measured** (source: local-inference-lab/rtx6kpro, `models/glm-5.3-flash/checksum-stability-moe-activations.md`; failure counts are the report's own n)

**Practical recommendations**:

| Priority | Action |
|---|---|
| 1 | Keep activations at **A16** unless a checksum-stability test says otherwise |
| 2 | Only lower activation precision after running that test on your checkpoint |
| 3 | Keep router weights in FP32 or BF16 — no measurable cost, and it is the difference on pre-QAD checkpoints |
| 4 | Expert weights can be quantized more aggressively than either |

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
