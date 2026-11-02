# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions use
[Semantic Versioning](https://semver.org/).

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
