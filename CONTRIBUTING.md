# Contributing Guide

## The two most welcome kinds of contributions

### 1. Reproducible measured data

This is the most valuable contribution to this project. Please follow [benchmarks/METHODOLOGY.md](benchmarks/METHODOLOGY.md) strictly.

Before submitting, please confirm:

- [ ] Complete server-side and client-side commands (verbatim, not "something like this")
- [ ] Hardware, clocks, power limit
- [ ] Exact versions of driver / CUDA / NCCL / framework / image tag
- [ ] At least 3 independent-launch repetitions, with the variance range
- [ ] Input/output lengths and sampling parameters

**Negative results are equally welcome.** "I tried TP=8 on eight 4090s and throughput was only 60% of TP=2" — data like that saves others days of trial and error, and is worth no less than a pretty speedup number.

### 2. Correcting errors

If you find a conclusion that does not hold, a number that was misread, or a recommendation with a counterexample on specific hardware — please open an issue with your evidence attached.

**This project is willing to be proven wrong.** The precondition is reproducible evidence.

## Content standards

### Conclusions must carry a basis tag

| Tag | When to use |
|---|---|
| **Measured** | There is a publicly verifiable measurement, with conditions and source stated |
| **Derivable** | Derived directly from hardware specs or framework mechanics, independent of any specific measurement |
| **Open** | The direction is clear but data is lacking |

### No numbers without a basis

This is a hard rule. Leave it blank rather than making it up. **An empty cell is far more valuable than a fake number.**

### Citations must name their source

When citing someone else's data, state clearly:

- Source project / repository
- The original measurement conditions (hardware, software, workload)
- **Which conclusions transfer and which do not**

### Terminology conventions

- Expand abbreviations on first use
- Commands, environment variables, paths, and log excerpts are not translated
- Keep specific parameter names as-is (e.g. `--tensor-parallel-size`)

### Wording

- Write judgments, not hedged ambiguity
- Acknowledge uncertainty, and say where the uncertainty lies
- Avoid words like "as everyone knows" and "obviously"

## Submission process

1. Fork this repository
2. Create a branch: `data/<hardware>-<date>` or `fix/<topic>`
3. Commit, with a commit message explaining what changed and on what basis
4. Open a Pull Request, with the measurement conditions in the description

## About recipes (recipes/)

Recipes are **structured starting points**, not validated optima. When adding a recipe, please:

- State the applicability boundaries (when it must not be used)
- State the topology prerequisites
- Give a one-line rationale for every parameter
- Explicitly list "known limitations" and "open questions"

## License

Contributed content will be published under the [MIT](LICENSE) license.
