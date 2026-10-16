# Machine-readable data format

`benchmarks/data/*.json` holds structured measurement data. The tables in `benchmarks/README.md` are backed by these files: **tables are for humans, JSON is for tools**.

`tools/planner.py --benchmarks benchmarks/data` reads this directory and feeds existing measured data directly into its planning output — the data is not just documentation; it enters the decision chain.

## Rules

1. **The schema field must be `lltg.benchmark.v1`**, otherwise the planner refuses to load the file (it would rather error out than guess the format).
2. **Unknown fields get `null`, not guessed values.** This discipline is consistent with [METHODOLOGY.md](../METHODOLOGY.md): blank is more valuable than fabricated.
3. `kind` has exactly two values:
   - `first-party` — measured on their own machine by this project or a contributor (must satisfy METHODOLOGY's repetition and variance requirements)
   - `external-reference` — publicly verifiable third-party measurements (source and non-transferable parts must be stated)
4. `variance` is required for `first-party` (a range across at least 3 independent launches); fill it for external data when available.
5. `transferable` / `not_transferable` are required: state explicitly which conclusions transfer and which do not.

## Field reference

| Field | Meaning |
|---|---|
| `id` | Stable identifier matching the file name (kebab-case) |
| `kind` | `first-party` or `external-reference` |
| `hardware` | GPU model × count, VRAM, power limit, interconnect, `nvidia-smi topo -m` summary, CPU/NUMA |
| `software` | Driver / CUDA / NCCL / framework and versions / image tag / model revision |
| `model` | Model name, quantization method and tool, revision |
| `workload` | Concurrency levels, input/output lengths, sampling parameters |
| `results[]` | One row per configuration; metric names use `*_c1_*` (single concurrency) / `*_c8_*` (aggregate) to distinguish |
| `variance` | Range across multiple independent launches, per metric |
| `methodology` | One-line summary of window length, warm-up, and repetition count |
| `source` | Source project and URL (required for external-reference) |
| `transferable` / `not_transferable` | Transfer boundaries |
| `submitted_by` | GitHub username (for first-party) |
| `date` | Measurement date (YYYY-MM-DD; write null if unknown, do not guess) |

## How to submit

Submit via the [benchmark report issue template](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml) and a maintainer will curate it into the library; or open a PR that adds a file following this schema directly (see [CONTRIBUTING.md](../../CONTRIBUTING.md)).
