# 05 Speculative Decoding

> Speculative decoding is often treated as "free speedup". On multi-GPU PCIe machines it is not free, and **the more you guess, the slower it can get**.

## 5.1 How it works and what it costs

The basic flow of speculative decoding:

1. A small, fast **draft** component guesses k tokens in a row
2. The **target** model verifies all k tokens in parallel with a single forward pass
3. Accept the longest correct prefix and restart from the first mistake

The key insight: **verifying k tokens in one forward pass is far cheaper than running k forward passes serially** (decode is memory-bandwidth-bound, and batching barely adds any time).

But that cheapness comes with conditions:

| Cost source | Explanation |
|---|---|
| Draft cost | The draft model itself takes VRAM and compute |
| Wasted verification | Whatever was guessed wrong is pure wasted compute |
| **Communication overhead** | On multiple GPUs, both draft and verification results must be synchronized across cards |
| **VRAM squeeze** | VRAM taken by the draft reduces what is left for the KV cache |

The first two apply in every scenario; the last two are amplified in the multi-GPU PCIe setting.

**Basis: derivable**

## 5.2 Comparing the main approaches

| Approach | Draft source | Extra VRAM | Characteristics |
|---|---|---|---|
| **MTP** | The model's own prediction heads | Minimal | No extra model needed; built in at training time |
| **DFlash / DFlash2** | A separate draft model | Significant | Stronger drafting, but requires an extra download and VRAM |
| **EAGLE family** | A separate small model | Moderate | General-purpose, mature ecosystem |
| **Medusa** | Multiple prediction heads | Small | Similar in spirit to MTP |

**The core difference**: MTP-style drafts "grow on the model itself", so their marginal cost is nearly zero; DFlash-style approaches attach a separate model, and the marginal cost includes VRAM, bandwidth, and one extra load.

**Basis: derivable**

## 5.3 The counterintuitive measurement: longer drafts ≠ faster

Published measurements (4x RTX PRO 6000 **Max-Q**, no NVLink, TP4, DCP1, NVFP4-quantized MoE model GLM-5.3-Flash, 4096-token prefill budget, 32 concurrent slots, FP8 KV cache, temperature 1 / top-p 0.95, five warmed-up 30-second windows per cell):

| Mode | Draft tokens | C1 single-concurrency | C8 aggregate | 32K prefill |
|---|---|---|---|---|
| MTP3 | 3 | **281.3 tok/s** | **897.6 tok/s** | 13,816 tok/s |
| DFlash2 K7 | 7 | 229.0 tok/s | 702.6 tok/s | 13,955 tok/s |

**Going from a draft of 3 to a draft of 7 dropped single-concurrency output by 18.6% and aggregate throughput by 21.7%.**

The verifier rate tells the same story: 112.14 steps/s for MTP3 versus 87.56 steps/s for DFlash2.

Meanwhile, **prefill was nearly identical between the two** (13,816 vs 13,955, a 1% difference) — exactly as expected, since speculative decoding only affects the decode phase.

**Why**:

1. **DFlash2 attaches a separate draft model**; the extra VRAM footprint squeezes the KV cache, and the extra weight reads compete for the already-scarce memory bandwidth
2. **Guessing 7 at a time wastes more when verification fails** — at a fixed acceptance rate, a longer draft means more discarded compute
3. **On machines without NVLink**, the cross-GPU synchronization of draft and verification competes with the main model's all-reduce for the same PCIe link

> ⚠️ **Note on run-to-run variance**: the source itself reports that a separate set of five C1 checks measured 265.7–277.0 tok/s for MTP3 and 212.9–225.2 tok/s for DFlash2. In other words, these numbers carry roughly ±5% of run-to-run variance; **do not use them for fine-grained version comparisons**. DFlash2's low value (212.9) even falls below the 219.7 recorded for its previous version.

**Basis: measured** (source: local-inference-lab/rtx6kpro, the GLM-5.3-Flash model page)

## 5.4 How to choose the draft length

| Situation | Recommendation |
|---|---|
| First-time setup | **Start at 2–3** and measure the acceptance rate first |
| High acceptance rate (>0.7) with bandwidth to spare | You can try up to 5–7 |
| No NVLink, multi-GPU TP | Stay around 3; the gains from lengthening get eaten easily |
| Draft requires a separate model | Work out the VRAM headroom first, then decide if it's worth it |
| Single GPU or NVLink cluster | You can be more aggressive |

**How to judge**: the model's startup log usually reports the acceptance rate. If the acceptance rate is low while the draft is long, you are mostly doing wasted work.

## 5.5 A practical order of trade-offs

When you want more speed, try these in order, from cheapest to most expensive:

```
1. Confirm the baseline: measure once with speculative decoding off (--mode off or similar)
2. Enable MTP-style (built into the model) with draft = 3
3. Tune the draft length: measure 2 / 3 / 5 in separate rounds
4. Still want more → first check whether the KV cache / quantization can be improved (see chapters 04 and 06)
5. Only then consider attaching a separate draft model (DFlash / EAGLE)
```

Step 4 is the one people skip. **Many of the gains people chase by adding a draft model are actually smaller than what you get from getting KV precision or the parallel configuration right.**

## 5.6 Measurement template

```markdown
### Configuration
- Target model / quantization:
- Parallel configuration (TP/PP/DCP):
- Draft scheme (MTP / DFlash / EAGLE):
- Draft token count:
- Client command:

### Results
| Draft length | Acceptance rate | C1 output | C8 aggregate | 32K prefill |
|---|---|---|---|---|
| 0 (baseline) | — |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 5 |  |  |  |  |

### Notes
- Number of warm-up windows:
- Duration of each measurement:
- Number of run-to-run repetitions (recommend ≥3):
```

**Basis: open**

---

Previous chapter: [04 Quantization](04-quantization.md) | Next chapter: [06 KV cache](06-kv-cache.md)
