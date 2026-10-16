# Configuration Recipes

Ready-to-use configurations, organized by hardware type. **Every recipe states its applicability boundaries and known limitations** — do not copy them across hardware types verbatim.

## Naming convention

```
<gpu-count>x-<vram>-<interconnect>-<notes>.md
Examples: 4x-96gb-pcie.md
          8x-24gb-pcie.md
```

## Available recipes

| Recipe | Hardware | Status |
|---|---|---|
| [4x-24gb-pcie](4x-24gb-pcie.md) | 4 × 24 GB cards, PCIe interconnect (RTX 4090 / 3090 class) | Reference starting point |
| [4x-96gb-pcie](4x-96gb-pcie.md) | 4 × 96 GB cards, PCIe interconnect (RTX PRO 6000 class) | Reference starting point |
| [8x-24gb-pcie](8x-24gb-pcie.md) | 8 × 24 GB cards, PCIe interconnect (RTX 4090 class) | Reference starting point |

## What a recipe contains

Each recipe file has four sections:

1. **Applicability boundaries** — when it works, and when it does not
2. **Topology prerequisites** — the connectivity you need to confirm with `nvidia-smi topo -m`
3. **Launch configuration** — complete, copy-pasteable commands, with a one-line rationale per parameter
4. **Known limitations and open questions** — explicitly stating which claims are inference and which require you to measure

## Important disclaimer

The recipes in this project are **structured starting points**, not item-by-item validated optima.

- Parameters are derived from public engineering practice and the decision rules in [02 Parallelism](../docs/02-parallelism.md)
- **No guarantee that they are optimal on your specific hardware** — PCIe topology, cooling, power policy, and model version all affect the outcome
- After using one, measure it following [benchmarks/METHODOLOGY.md](../benchmarks/METHODOLOGY.md) and submit your results back

**Basis: derivable** (the recipes themselves); the specific numbers need to be confirmed by your own measurements.
