# Tools

Two front ends over the same set of decisions.

| File | What it is | When to use it |
|---|---|---|
| [`planner.py`](planner.py) | Command-line planner. Standard library only, no dependencies. | On a delivery host, in CI, or anywhere without a browser |
| [`config-generator.html`](config-generator.html) | Single-file offline web page. No build step, no network. | Customer networks that block external resources; handing a file to someone |
| [`detect_topology.py`](detect_topology.py) | Collects GPU topology, power limits, and NUMA layout as JSON. | Before choosing parallelism; also produces the hardware block of a recipe |
| [`check_repo.py`](check_repo.py) | CI check: bilingual parity, relative links, benchmark schema. | Every push, locally or in GitHub Actions |

Both answer the same three questions — interconnect, memory, strategy — and
both refuse to invent numbers. They differ only in how you reach them.

The "no network" claim of `config-generator.html` is enforced, not assumed:
`tests/test_offline_html.py` fails the build the moment an external resource
reference creeps in.

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

# attach the measured data from benchmarks/data to the output
python tools/planner.py --gpus RTXPRO5000-72 --count 8 --model qwen2.5-32b \
    --benchmarks benchmarks/data

# just list what the benchmark directory holds
python tools/planner.py --benchmarks benchmarks/data --list-benchmarks

# hardware table and built-in models
python tools/planner.py --list-gpus
python tools/planner.py --list-models
```

The built-in model table now includes Qwen3.8-27B, Qwen3-235B-A22B,
MiniMax-M3.1-Flash, and DeepSeek-V4.1-Flash. The last one uses MLA, so its
KV/token is a measured override rather than the GQA formula; pass
`--kv-bytes-per-token` for your own build.

On `--gpus RTXPRO5000-72`, the planner warns when the selected model needs an
SM120 kernel workaround (MiniMax-M3.1-Flash, DeepSeek-V4.1-Flash). The warning
also appears in `--json` output under `warnings`.

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

## detect_topology.py

```bash
python tools/detect_topology.py            # human-readable summary
python tools/detect_topology.py --json     # paste into a recipe / CI
python tools/detect_topology.py --planner  # --gpus ... --count ... --group ...
```

It runs `nvidia-smi --query-gpu`, `nvidia-smi topo -m`, `numactl -H`, and
`lspci`, then reports the PCIe/NUMA groups, whether NVLink is present, power
limits, and a planner argument hint. Everything is read-only and offline.

## check_repo.py

```bash
python tools/check_repo.py
```

Fails when a local Markdown link points at a missing file, an English chapter
lacks its Chinese counterpart (or vice versa), or a benchmark JSON does not
carry `lltg.benchmark.v1` and the fields the planner needs. CI runs it on every
push and pull request.

## bench/run_matrix.py

The benchmark runner lives in [`../bench/run_matrix.py`](../bench/run_matrix.py).
It wraps `vllm bench serve`, runs the C1/C8/C32/C64 + 32K + ShareGPT matrix,
and writes `summary.json` / `summary.md` with variance across repeated runs.
