# 01 Hardware & Topology

> Before you touch a single parameter, understand how your GPUs are actually wired together. Get this step wrong and every setting after it is wrong too.

## 1.1 How big is the gap

| Interconnect | Typical bidirectional bandwidth | Where you see it |
|---|---|---|
| PCIe 4.0 x16 | ~64 GB/s | Previous-gen platforms, consumer motherboards |
| PCIe 5.0 x16 | ~128 GB/s | Current mainstream workstations and servers |
| NVLink 3 (A100) | ~600 GB/s | A100 SXM |
| NVLink 4 (H100) | ~900 GB/s | H100 SXM |
| NVLink 5 (B200) | ~1800 GB/s | B200 / GB200 |

That is a **7x to 14x** difference. This is why every recipe copied from an NVLink cluster falls apart on machines without NVLink.

**Basis: derivable** (all figures are vendor-published specifications)

## 1.2 The key point: PCIe is not homogeneous either

Four PCIe cards can have completely different topologies. From fastest to slowest:

| `nvidia-smi topo -m` label | Meaning | Relative speed |
|---|---|---|
| `NV#` | NVLink, the number is the link count | Fastest |
| `PIX` | Through a **single** PCIe Switch | Fast |
| `PXB` | Through **multiple** PCIe Switches | Medium |
| `PHB` | Through a PCIe Host Bridge (fanned out by the CPU) | Slow |
| `SYS` | Across the inter-CPU SMP interconnect (crosses NUMA) | Slowest |
| `NODE` | Within the same NUMA node | — |

**What this means for inference**: every all-reduce step of TP travels along these paths. If a pair of cards is connected via `SYS`, that traffic has to leave one CPU, cross QPI/UPI, and enter the other CPU — latency can be several times that of `PIX`.

**Basis: derivable**

## 1.3 Run these three commands first

```bash
# 1. How the GPUs are connected to each other — the most important one
nvidia-smi topo -m

# 2. Which PCIe slot each card sits in, at what generation and width
nvidia-smi -q | grep -A2 "GPU Link Info"

# 3. NUMA layout (how many CPUs, which cards each one owns)
lscpu | grep -i numa
numactl --hardware
```

The output of `nvidia-smi topo -m` is a matrix; each cell shows the connection type between two cards:

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      PIX     PXB     SYS     0-31            0
GPU1    PIX      X      PXB     SYS     0-31            0
GPU2    PXB     PXB      X      SYS     0-31            0
GPU3    SYS     SYS     SYS      X      32-63           1
```

In this example, GPU0/1/2 sit behind the same set of PCIe Switches (`PIX`/`PXB`), while GPU3 hangs off the other CPU (`SYS`). **The conclusion falls out immediately: TP=4 would cross the slowest link, so consider TP=3 (staying within the switch) or a 2x2 arrangement.**

## 1.4 What the topology decides

| What you observe | Configuration implication |
|---|---|
| All `NV#` | A normal NVLink cluster; follow the standard advice |
| All `PIX` | Four cards on one switch; TP=4 is viable, but watch the all-reduce pattern |
| A mix of `PXB` / `PHB` | Scale TP cautiously, consider splitting with PP |
| Any `SYS` | **Do not let TP cross the SYS boundary** — that is the performance cliff |

## 1.5 Measurement template

This project does not publish unverified bandwidth numbers. Use the template below to measure and submit your results (see [CONTRIBUTING.md](../CONTRIBUTING.md)):

```markdown
### Hardware
- GPU model / count:
- GPU interconnect (topo -m summary):
- CPU / NUMA domains:
- Motherboard / PCIe Switch:
- PCIe generation and width:

### Measurements
| GPU pair | topo label | all-reduce 32MB time | measured bandwidth |
|---|---|---|---|
| 0-1 |  |  |  |
| 0-3 |  |  |  |

### Tools and versions
- NCCL version:
- CUDA version:
- Measurement command:
```

Recommended measurement tools: `nccl-tests` (`all_reduce_perf`), `nvbandwidth`.

**Basis: open**

---

Next chapter: [02 Parallelism](02-parallelism.md)
