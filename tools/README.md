# Tools

Two front ends over the same set of decisions.

| File | What it is | When to use it |
|---|---|---|
| [`planner.py`](planner.py) | Command-line planner. Standard library only, no dependencies. | On a delivery host, in CI, or anywhere without a browser |
| [`config-generator.html`](config-generator.html) | Single-file offline web page. No build step, no network. | Customer networks that block external resources; handing a file to someone |

Both answer the same three questions — interconnect, memory, strategy — and
both refuse to invent numbers. They differ only in how you reach them.

## planner.py

```bash
# 8x A800, Qwen2.5-72B, tuned for throughput
python tools/planner.py --gpus A800-80 --count 8 --model qwen2.5-72b \
    --ctx 8192 --goal throughput

# 4x RTX 4090, a 7B model: expect replicas, not tensor parallelism
python tools/planner.py --gpus RTX4090-24 --count 4 --model qwen2.5-7b

# GPUs split across two PCIe switches, two cards per switch
python tools/planner.py --gpus L40S-48 --count 4 --params 70 \
    --interconnect pcie-cross-switch --group 2

# machine-readable output for scripting
python tools/planner.py --gpus H800-80 --count 8 --model qwen2.5-72b --json

# hardware table and built-in models
python tools/planner.py --list-gpus
```

### What it decides

1. **Interconnect** — reads the real GPU-to-GPU bandwidth (A800 and H800 are
   cut to 400 GB/s, versus 600 and 900 for A100 and H100) and derives how far
   tensor parallelism can go before the link starts taxing it.
2. **Memory** — weights at the chosen quantisation, plus KV cache at the
   requested context and concurrency, against usable VRAM per GPU.
3. **Strategy** — TP / PP / replica counts, restricted to tensor-parallel sizes
   that actually divide the model's attention and KV head counts. A TP value
   the engine rejects is not a slow configuration, it is a crash.

### Evidence grades

Every conclusion is tagged, and the tag travels with the output:

| Tag | Meaning |
|---|---|
| `[measured]` | Publicly verifiable measurement; source recorded next to it |
| `[derivable]` | Follows from hardware specs or framework mechanics |
| `[open]` | Direction known, public data missing; the cell is left empty |

If the KV geometry of your model is not in the built-in table, the planner says
so and tells you how to measure it, rather than guessing a number that would
silently decide whether the model "fits".

Run the checks with:

```bash
python -m unittest discover tests -v
```
