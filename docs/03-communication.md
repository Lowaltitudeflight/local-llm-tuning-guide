# 03 Communication

> TP's communication happens at every layer. This chapter is about spending as little time on it over PCIe as possible.

## 3.1 Two kinds of all-reduce and when each applies

The two-stage variant (commonly called two-shot) splits the all-reduce into a reduce-scatter followed by an all-gather:

| | Two-stage (reduce-scatter + all-gather) | One-shot |
|---|---|---|
| Total traffic | Roughly independent of the rank count | Rises linearly with the rank count |
| Synchronization rounds | Two | One |
| Best case | **Large messages**, bandwidth-bound | **Small messages**, latency-bound |
| On PCIe | Still viable for large messages | Usually better during decode |

The heart of this trade-off is **whether latency or bandwidth dominates**:

- **Prefill**: thousands of tokens processed at once, so all-reduce messages are large → bandwidth-bound → two-stage still pays off
- **Decode**: very few tokens per step, so messages are small but synchronization is extremely frequent → latency-bound → one-shot wins

So the right question is not "which algorithm is better" but "how large are the messages in this phase".

**Basis: derivable**

## 3.2 What real projects do

Two points from public PCIe optimization practice (Local Inference Lab's GLM-5.3-Flash configuration) are worth studying:

1. **Two-stage all-reduce disabled by default** — its configuration documentation states explicitly that "Two-shot all-reduce is off"; the target configuration takes the one-shot path.
2. **A message-size threshold for switching modes** — its configuration exposes a parameter like `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE`, which shows this is not a one-size-fits-all choice but one that changes with message size.

Point 2 is more instructive than point 1: **the takeaway is not an algorithm, it's a threshold.**

> That project targets 4 PCIe GPUs (no NVLink) running an NVFP4-quantized MoE model at TP=4. Its configuration is a reasonable starting reference for the 4-GPU PCIe scenario.

**Basis: measured** (source: the public configuration files of local-inference-lab/rtx6kpro)

## 3.3 NCCL environment variables

These are the variables most often worth tuning on PCIe machines. **Establish a baseline measurement before changing anything**, otherwise you cannot tell whether a change helped.

| Variable | What it does | Typical use |
|---|---|---|
| `NCCL_P2P_LEVEL` | Highest topology level at which P2P transfers are allowed (`PIX`/`PXB`/`PHB`/`SYS`) | For cross-NUMA GPU pairs, restrict to `PIX` to avoid the slow path |
| `NCCL_P2P_DISABLE` | Disable GPU-to-GPU P2P entirely | A diagnostic tool when P2P is suspected of being unstable |
| `NCCL_ALGO` | Select the collective algorithm (`Ring` / `Tree`) | With few GPUs, `Tree` can reduce latency |
| `NCCL_PROTO` | Transport protocol (`LL` / `LL128` / `Simple`) | `LL` often wins for small messages, `Simple` for large ones |
| `NCCL_MIN_NCHANNELS` | Minimum number of channels | Raise it when bandwidth utilization is too low |
| `NCCL_BUFFSIZE` | Chunk buffer size | Match it to your message sizes |
| `NCCL_ASYNC_ERROR_HANDLING` | Asynchronous error handling | Temporarily disable when debugging hangs; **do not use in production** |

**General troubleshooting order**:

```bash
# 1. Confirm the topology before touching anything (see chapter 01)
nvidia-smi topo -m

# 2. Establish a baseline with nccl-tests
all_reduce_perf -b 8 -e 128M -f 2 -g 4

# 3. Change one variable at a time, then re-measure
# 4. Record: the change + before/after numbers + hardware and software versions
```

**Basis: derivable** (for NCCL variable semantics, the official documentation is authoritative)

## 3.4 Interaction with speculative decoding

Speculative decoding **amplifies communication pressure**:

- Candidate tokens produced by the draft phase must be verified in sync across GPUs
- The longer the draft, the more data each verification round moves
- If the draft model is itself TP-sharded, its all-reduces compete with the main model's traffic for the same PCIe link

This is why speculative decoding pays off less on PCIe machines than on NVLink machines, and why draft lengths need to be more conservative there. See [05 Speculative decoding](05-speculative.md) for details.

**Basis: derivable**

## 3.5 Measurement template

```markdown
### Configuration
- GPU / count:
- Topology (topo -m summary):
- Parallel configuration (TP/PP):
- NCCL version:
- Relevant environment variables:

### nccl-tests all_reduce_perf results
| Message size | Algorithm/protocol | Time | Bandwidth |
|---|---|---|---|

### Inference-side comparison
| Configuration | C1 output | C8 aggregate | Notes |
|---|---|---|---|
```

**Basis: open**

---

Previous chapter: [02 Parallelism](02-parallelism.md) | Next chapter: [04 Quantization](04-quantization.md)
