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
| `software` | Driver / CUDA / NCCL / framework and versions / image **digest** (not just tag) / model revision |
| `software.image_digest` | `repo@sha256:...`; a mutable tag does not identify a build |
| `software.clocks` | Clock policy during the run (locked / default / unknown). "Measured" claims are not reproducible without it |
| `model` | Model name, quantization method and tool, revision |
| `workload` | Concurrency levels, input/output lengths, sampling parameters, prompt source or seed |
| `commands.server` | The exact server launch command (placeholders are not reproducible) |
| `commands.client` | The exact load-generator command and its version |
| `results[]` | One row per configuration; metric names use `*_c1_*` (single concurrency) / `*_c8_*` (aggregate) to distinguish |
| `variance` | Range across multiple independent launches, per metric |
| `methodology` | One-line summary of window length, warm-up, and repetition count |
| `source` | Source project and URL (required for external-reference) |
| `transferable` / `not_transferable` | Transfer boundaries |
| `submitted_by` | GitHub username (for first-party) |
| `date` | Measurement date (YYYY-MM-DD; write null if unknown, do not guess) |

## Measured vs derived

Two numbers that look identical in a table can have different standing:

- **measured** — produced by a run, with `variance`, `commands.*` and `software.clocks` filled in.
- **derived** — computed from a model config or hardware spec (`2 × layers × kv_heads × head_dim × bytes` for KV/token, `params × bytes` for weights). It is still useful, but it must be labelled `derived` and the formula must be visible, because it cannot disagree with reality — it only disagrees with a wrong config.

If a figure was back-computed from a startup log rather than observed directly, write `derived-from-log` and say which line. Files in this directory that predate the `commands`/`clocks` fields are known to be incomplete on this axis; treat their `measured` label as "run on hardware" rather than "fully specified".

## How to submit

Submit via the [benchmark report issue template](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml) and a maintainer will curate it into the library; or open a PR that adds a file following this schema directly (see [CONTRIBUTING.md](../../CONTRIBUTING.md)).
