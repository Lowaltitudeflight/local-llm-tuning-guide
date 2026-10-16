# Measurement Methodology

> This project's value lies not in its conclusions, but in **whether those conclusions are reproducible**. That is why measurement discipline matters more than measurement results.

## Why this document exists

The most common problem in Chinese-language technical content is: numbers are given, but conditions are not. As a result, the same claim can be true on one machine and false on another, and the reader has no way to tell.

This discipline has exactly one goal: **to let another person determine, on their own machine, whether your numbers apply to them.**

## 1. Conditions that must be fixed before measuring

If any one of these is missing, the data cannot be compared.

### Hardware

- [ ] GPU model and count
- [ ] **Power limit** (check with `nvidia-smi -q -d POWER`; low-power variants must be noted)
- [ ] **Clock policy**: locked clocks or automatic (under automatic clocks, results differ with cooling conditions)
- [ ] Interconnect topology (summary of `nvidia-smi topo -m`)
- [ ] CPU / NUMA layout
- [ ] System memory capacity

### Software

- [ ] Driver version
- [ ] CUDA version
- [ ] NCCL version
- [ ] Inference framework version (vLLM / SGLang, etc.)
- [ ] **Exact tag or digest of the container image**
- [ ] Exact revision / commit of the model weights
- [ ] Quantization variant (for community quantizations, give the quantization tool and configuration)

### Configuration

- [ ] Complete server-side launch command (not "something like this" — verbatim)
- [ ] All relevant environment variables
- [ ] Complete client-side command
- [ ] Concurrency level
- [ ] Prompt length (input token count)
- [ ] Generation length (output token count)
- [ ] Sampling parameters (temperature / top_p / whether greedy)

## 2. Which metrics to measure

| Metric | Definition | What it reflects |
|---|---|---|
| **TTFT** | Time from request sent to first token received | Prefill capability |
| **Output throughput** | Single-request decode speed (tok/s) | Decode capability; latency-sensitive scenarios look at this |
| **Aggregate throughput** | Total output under multi-concurrency (tok/s) | Throughput-oriented scenarios look at this |
| **P99 latency** | 99th-percentile end-to-end latency | Serving stability |

**You must distinguish single-concurrency from aggregate**: reporting only one number is seriously misleading. The recommended levels are C1 / C8 (1 concurrent / 8 concurrent), consistent with industry convention.

## 3. Measurement discipline

These three rules determine whether your numbers are usable:

### 1. Warm-up

An inference service has one-time overheads on the first batch of requests: compilation, memory allocation, CUDA graph capture, and so on. **Measure without warming up and you are measuring startup cost, not inference performance.**

Recommendation: discard the first few windows and only aggregate data from the stable ones.

### 2. Fixed windows + repeated runs

| Practice | Explanation |
|---|---|
| Fixed measurement window | e.g. one 30-second window, rather than "run 100 requests" |
| Multiple repetitions | **At least 3 independent launches**, not 3 passes inside the same process |
| Report variance | Give a range (e.g. 265.7–277.0), not just the prettiest number |

Community practice shows that **the same configuration can vary by up to ±5% between launches** — if you do not repeat the measurement, the "version improvement" you observe may just be noise.

### 3. Change one variable at a time

When comparing two configurations, keep everything except the target parameter identical. Change the quantization and the parallelism at the same time, and attribution becomes impossible.

## 4. What you must report

Use this template to submit data (see [CONTRIBUTING.md](../CONTRIBUTING.md)):

```markdown
### Hardware
- GPU: model × count
- Power limit:
- Clock policy:
- Topology summary:
- CPU / NUMA:

### Software
- Driver / CUDA / NCCL:
- Framework + version:
- Image tag:
- Model revision / quantization:

### Server side
(complete command, pasted verbatim)

### Client side
(complete command, pasted verbatim)

### Workload
- Concurrency: C1 / C8
- Input length:
- Output length:
- Sampling parameters:

### Results
| Configuration | C1 output | C8 aggregate | 32K prefill | Notes |
|---|---|---|---|---|

### Measurement details
- Number of warm-up windows:
- Window duration:
- Number of independent launch repetitions:
- Observed variance range:
```

## 5. A necessary attitude

**Negative results are just as valuable.**

"I tried TP=8 on eight 4090s and throughput was only 60% of TP=2" — data like that saves whoever comes next days of trial and error, and is worth no less than a pretty speedup number.

This project welcomes two kinds of submissions: **reproducible performance data** and **reproducible failure reports**. The only thing we do not accept is numbers without conditions.
