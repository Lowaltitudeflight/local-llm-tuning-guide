# Recipe: 4 × 96 GB, PCIe interconnect

Target hardware: 4 cards with 96 GB of VRAM (RTX PRO 6000 Blackwell class), no NVLink, connected via PCIe 5.0 x16.

## Applicability boundaries

| Works | Does not work |
|---|---|
| Single-machine deployment with 384 GB total VRAM | Scenarios that require scaling across nodes |
| Small-to-medium MoE (quantized) | Unquantized very large dense models |
| Regular serving with 32K–128K context | Ultra-long scenarios requiring 1M context (see [06 KV Cache](../docs/06-kv-cache.md)) |

## Topology prerequisites

**Run this first, then decide the parameters:**

```bash
nvidia-smi topo -m
```

Decision rules (see [02 Parallelism](../docs/02-parallelism.md)):

| Observation | How to set TP |
|---|---|
| All four cards pairwise `PIX` / `PXB`, all within one PCIe Switch group | TP=4 |
| Four cards split across two Switches | TP=2 × PP=2 |
| Any `SYS` (cross-CPU-socket) | Split at Switch boundaries; do not let TP cross them |

## Launch configuration

### Case A: all four cards in the same group (TP=4)

```bash
vllm serve <your-model> \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

| Parameter | Why it is set this way |
|---|---|
| `--tensor-parallel-size 4` | Use all four cards, all on the fast links |
| `--gpu-memory-utilization 0.90` | Leave headroom for workspace; **do not jump straight to 0.95** — burst peaks can fail |
| `--max-model-len 32768` | Conservative starting point; get it running first, then scale up using the formula in 6.1 |
| `--max-num-seqs 32` | Moderate-concurrency starting point |
| `--kv-cache-dtype fp8` | The cheapest VRAM-saving lever |
| `--enable-prefix-caching` | Significant benefit when prefixes repeat |
| `--enable-chunked-prefill` | Keeps long prompts from blocking decoding |

### Case B: split across two Switches (TP=2 × PP=2)

```bash
vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

No extra parameters needed — let PP carry the cross-Switch communication. It only passes activations at layer boundaries, at far lower cost than TP's per-layer all-reduce.

## Known limitations and open questions

| Item | Status |
|---|---|
| The parameters above are a structurally sound starting point | **Derivable** |
| Optimal values of this recipe on your specific hardware | **Open** |
| Whether NCCL environment variables need tuning | **Open** (see [03 Communication](../docs/03-communication.md)) |
| Whether speculative decoding brings a benefit | **Open** (start with draft=3, see [05 Speculative Decoding](../docs/05-speculative.md)) |

**Special note**: if your cards are the low-power versions (Max-Q class, ~300 W), the power limit may become the bottleneck before VRAM does. Please note the power limit and cooling conditions in your measurement record.

## After you measure

Please record following [METHODOLOGY.md](../benchmarks/METHODOLOGY.md) and submit your results to [benchmarks/](../benchmarks/). The value of the recipe library comes from real data, not guesswork.
