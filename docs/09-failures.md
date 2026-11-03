# 09 Failed configurations

Most tuning guides only publish what worked. That is the less useful half: when a configuration fails, the failure is usually silent — the service starts, answers, and is just slower than it should be, or it crashes with an error that points at the wrong layer.

This chapter is the negative half of the library: configurations that **do not work**, with the symptom, the actual cause, and the verdict. Entries marked **measured** were hit on the 8×72 reference box; the rest are derivable from framework/kernel mechanics.

## 9.1 How to read an entry

| Column | Meaning |
|---|---|
| **Symptom** | What you see first — the error line or the suspicious number |
| **Actual cause** | Where the problem really is, once diagnosed |
| **Verdict** | `never` (no parameter fixes it) / `avoid` (works but costs more than it gains) / `conditional` (works under stated conditions) |

## 9.2 Measured on the 8×72 box

| ID | Configuration | Symptom | Actual cause | Verdict |
|---|---|---|---|---|
| F-01 | MiniMax-M3.1-Flash NVFP4 on SM120 (RTX PRO 5000/6000) | Startup fails; after working around the prefill backend, decode indexer rejects the card | Kernel source hard-codes `arch not in ((10,0),(10,3))` — MSA sparse-attention decode indexer only accepts SM100/SM103 | `never` on that kernel build. Use the NVFP4 + Marlin path, or wait for an sm_120 port |
| F-02 | Qwen3.5-family model in a MiniMax dev image (triton 3.9) | Starts, then crashes in the linear-attention kernel | The image's triton removed `tl.make_block_ptr`; the model kernel still calls it | `never` on that image. Use the mainline image (triton 3.8) |
| F-03 | `bench_serving` inside a container | Tokenizer can't read the local path; random dataset generation hangs | Tokenizer needs the model dir mounted; dataset generation fetches config from the network | `conditional` — mount the model dir and pass `-e HF_ENDPOINT=...` |
| F-04 | TP=8 with `--enable-dp-attention --dp 8` on a **dense** model | 37.7 tok/s at c1 vs 75.3 for naive TP8 | DP-attention adds a second all-to-all schedule; dense models have nothing to shard across it | `avoid` for dense. It is a MoE tool |
| F-05 | TP4 pinned to all 8 GPUs' worth of topology (no `CUDA_VISIBLE_DEVICES`) | TP4 lands across the SYS boundary, 20-30% slower than expected | The scheduler picks arbitrary cards, not the four under one NUMA node | `avoid`. Pin the group: `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| F-06 | MoE with W4A4 activations | Output is fluent but non-deterministic; greedy checksum fails 5.9% of the time | Activation precision, not the router | `avoid`. Move to A16 first (0/500 failures); router FP32 is a second, checkpoint-dependent safeguard |

## 9.3 Configurations that do not start at all

| ID | Configuration | Error / symptom | Cause | Verdict |
|---|---|---|---|---|
| F-10 | TP=3 on a model with 4 KV heads | Engine refuses to start | TP must divide attention heads **and** KV heads; 4 % 3 ≠ 0 | `never` |
| F-11 | TP greater than the attention head count | Engine refuses to start | No way to split fewer heads across more ranks | `never` |
| F-12 | `--kv-cache-dtype fp8` on A100 (SM80) | Fails or silently falls back to fp16 | FP8 KV requires SM≥8.9; A100 has no FP8 tensor cores | `never` on A100. Use fp16 KV |
| F-13 | NVFP4 weights on non-Blackwell | Kernel not found / unsupported quantization | NVFP4 needs Blackwell-class tensor cores | `never`. Fall back to AWQ/GPTQ or FP8 |
| F-14 | AWQ weights with `--dtype bfloat16` | Start fails or precision is silently coerced | vLLM's AWQ kernels only declare fp16 activations | `never`. Use `--dtype float16` (or omit) |
| F-15 | 72B BF16 on 4 × 72 GB with `--gpu-memory-utilization 0.99` | OOM at startup even though weights "fit" | 145 GB of weights + per-card workspace exceeds 4 × 72 × 0.99 | `never` at that size. 8 cards, or quantize |
| F-16 | `--max-model-len` larger than what the KV pool can hold | `max seq len is larger than the maximum number of tokens that can be stored in KV cache` | The declared context cannot be served by the remaining VRAM | `never` for that pair. Lower the context or lower KV precision |

## 9.4 Configurations that run but mislead

These are the most expensive class: nothing crashes, the numbers look good, and the conclusion is wrong.

| ID | Configuration | What looks right | What is actually wrong |
|---|---|---|---|
| F-20 | Every request uses the same prompt | TTFT looks excellent | Prefix cache serves the prefill; you measured the cache, not the model |
| F-21 | One run per concurrency level | Throughput curve is smooth | No variance, no error bars; a ±5% startup-to-startup difference can flip the ranking |
| F-22 | Reporting throughput without latency | "c64 is the best configuration" | c64 may have 1.6 s of queueing before the first token; goodput may be lower than c8 |
| F-23 | Comparing engines/quantizations at different `--max-tokens` or concurrency levels | "AWQ is 2× faster" | The comparison changed more than one variable |
| F-24 | Benchmarking from the same machine as the server | Simpler setup | The load generator competes for CPU/PCIe and pollutes TTFT and P99 |
| F-25 | `measured` KV/token figures that are actually `2 × layers × kv_heads × head_dim × bytes` | A precise-looking number | It is a derivation, not a measurement; label it `derived` and keep the formula visible |

## 9.5 Adding an entry

Open an issue with:
1. the exact command line (image digest, framework version),
2. the first error line (not the whole log),
3. `nvidia-smi topo -m`,
4. what you tried that also failed.

A "does not fit" result is an entry too — the negative column of the [4×72 recipe](../recipes/4x-72gb-pcie.md) collects those.
