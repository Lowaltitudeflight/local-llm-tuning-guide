# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions use
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-03

### Fixed

- **MODEL_MATRIX KV/token was wrong for every fp8 row** (and for Qwen3-8B).
  The fp16/fp8 columns differed by 4× where the precision only accounts for 2×:
  Qwen3.8-27B and Qwen3-32B were 64 KiB instead of 128 KiB, Qwen3-235B-A22B
  64 instead of 94, Qwen3-8B 128 instead of 144. A derivation block now shows
  the geometry, and `tools/check_model_matrix.py` re-derives the published rows
  in CI.
- **`tools/planner.py` treated every known PCIe SKU as "interconnect unknown"**,
  capping TP at 2 — so `--gpus RTXPRO5000-72 --count 8` contradicted the
  guide's own TP8 measurement. PCIe SKUs now have their own branch; `--group`
  caps TP at one switch, and the ceiling is otherwise the card count.
- **RTX 3090 was listed as driver-restricted for P2P.** That restriction is a
  RTX 40/50 consumer-card property; 3090 is Ampere and supports P2P.
- **The MoE conclusion named the wrong lever.** The cited report shows W4A4 →
  W4A16 activation precision taking checksum failures from 5.9% to 0; FP32
  router weights are free but checkpoint-dependent (0/500 on QAD, 19.5% → 4.0%
  pre-QAD). The chapter now says "activations first, router second".
- **"TP8 beats TP4" dropped the 2× hardware caveat.** TP8 used 8 cards vs TP4's
  4; per GPU, TP4 is 1.7× better. The 89% label was also C8, not C1 (85.3%).
- **`--enable-chunked-prefill` two-shot "threshold" claim** was inferred from a
  parameter name in a config whose value is `off`.
- **Topology order** put NODE after SYS; NODE is same-socket and is the fast
  path for a 4-GPU group.
- **TP=3 recommendation** replaced with TP=2 × PP=2: TP must divide heads and
  KV heads, and the documented example card pair is already on `PXB`.
- **FP8 hardware floor** corrected from "Hopper and newer" to Ada (SM89)+.
- **`gpu-memory-utilization`** described as a total budget, not "the last three
  items" (weights are inside it, which is why 0.95 OOMs).
- **tok/s/W had two denominators** (1.16 vs 1.69 for the same run). The table
  now shows measured-draw and limit-normalised side by side.

### Added

- `docs/09-failures.md` (+ zh-CN): configurations that do not start, and ones
  that run but mislead.
- vLLM ↔ SGLang flag mapping in `docs/02`, with the three traps that make a
  copied config wrong.
- Latency-vs-concurrency table and SLO-based goodput guidance in
  `benchmarks/README.md`.
- `tools/check_model_matrix.py`; heading-anchor validation and benchmark-page
  parity in `tools/check_repo.py`.
- SCHEMA: `commands.server`, `commands.client`, `software.clocks`,
  `software.image_digest`, plus a measured-vs-derived section.

## [0.2.0] - 2026-11-02

### Added

- **8×72 GB recipe, Case C**: MiniMax-M3.1-Flash NVFP4 now runs on SM120 via
  three read-only patches plus explicit flags; measured C1–C64, cache-on/off,
  and 1M-context numbers.
- **`benchmarks/data/minimax-m31-flash-8x72-sm120.json`**: first-party data
  for the above (9,680 requests, 0 failures; 1M cold request 517.7 s).
- **4×72 GB recipe** (`recipes/4x-72gb-pcie.md`): what fits on four 72 GB
  cards, what does not, and why (including the negative results).
- **`MODEL_MATRIX.md`**: weights, KV/token, minimum 72 GB card count, and
  verified configurations per model.
- **`tools/detect_topology.py`**: one-command topology, power, and NUMA
  collection with a planner hint.
- **`bench/run_matrix.py`**: C1/C8/C32/C64 + 32K + ShareGPT runner that writes
  a variance summary.
- **`tools/check_repo.py`**: bilingual parity, relative-link, and benchmark
  schema checks; wired into CI.
- **`docs/07-troubleshooting.md`** and **`docs/08-power-cost.md`** plus Chinese
  counterparts.
- `CITATION.cff`, `CHANGELOG.md`, `DATA_LICENSE.md`, `pyproject.toml`,
  `requirements-dev.txt`, and `.pre-commit-config.yaml`.

### Changed

- README quick decision table now says “start with TP within the switch, then
  measure” instead of treating the switch boundary as absolute; links the
  dense-27B counterexample.
- Planner model table adds Qwen3.8-27B, Qwen3-235B-A22B, MiniMax-M3.1-Flash,
  and DeepSeek-V4.1-Flash; `--list-models` and an SM120 warning were added.
- Fixed three broken relative links in the Chinese recipes and added the
  required `submitted_by` field to existing first-party benchmark JSON files.

### Gitee

- The Gitee mirror is Chinese-primary (`README.md` Chinese, `README.en.md`
  English); GitHub remains English-primary with `README.zh-CN.md`.

## [0.1.0] - 2026-10-23

### Added

- Six chapters: hardware/topology, parallelism, communication, quantization,
  speculative decoding, KV cache (English + Chinese).
- 4×24, 4×96, 8×24, 8×72 recipes (English + Chinese).
- Planner CLI and offline configuration generator.
- First-party benchmarks: Qwen3.8-27B 8×72 parallelism, single-card BF16
  across three Blackwell SKUs, GLM-5.3-Flash FP8 TP8+EP8.
- Measurement methodology, benchmark schema, issue templates, CI.
