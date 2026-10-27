#!/usr/bin/env python3
"""planner.py - parallel-strategy planner for locally hosted LLM serving.

Hardware in, launch command out. The same job as ``tools/config-generator.html``,
but as a command-line tool so that it can run on an air-gapped delivery host,
be unit-tested, and be wired into CI.

It answers three questions, in order:

1. **Interconnect** - what is the real bandwidth between the GPUs, and how far
   can tensor parallelism be pushed before it starts paying for itself?
2. **Memory** - do weights plus KV cache fit, and at what TP / PP do they fit?
3. **Strategy** - given the two answers above, what is the least surprising
   TP / PP / DP combination, and why?

Every printed conclusion carries an evidence grade, because a recommendation
without a basis is exactly what this project exists to avoid:

* ``measured``  - publicly verifiable measurement, source recorded next to it
* ``derivable`` - follows from hardware specs or framework mechanics alone
* ``open``      - direction is clear, public data is missing; the cell stays empty

This tool deliberately refuses to invent numbers. When the KV footprint of a
model is unknown, it says so and tells you how to measure it.

Usage
-----
    python tools/planner.py --gpus A800 --count 8 --model qwen2.5-72b
    python tools/planner.py --gpus RTX4090 --count 4 --params 7 --quant fp16
    python tools/planner.py --gpus H800 --count 8 --model qwen2.5-72b \
        --ctx 32768 --goal throughput --json
    python tools/planner.py --gpus RTXPRO5000-72 --count 8 --model qwen2.5-32b \
        --benchmarks benchmarks/data
    python tools/planner.py --list-gpus

Standard library only. No dependencies, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Evidence grades
# --------------------------------------------------------------------------

MEASURED = "measured"
DERIVABLE = "derivable"
OPEN = "open"

# --------------------------------------------------------------------------
# Hardware table
# --------------------------------------------------------------------------
# ``link_bw_gbs`` is bidirectional GPU-to-GPU bandwidth in GB/s.
# ``scaling`` is the observed per-GPU efficiency in a fully populated node
# (1.0 would be perfect linear scaling). Sources are recorded so that the
# number can be re-checked rather than trusted.
#
# A800 / H800 are the export-controlled variants: identical compute and HBM to
# A100 / H100, with NVLink bandwidth cut to 400 GB/s. Their scaling figures are
# therefore worse than the cards they are derived from, which is precisely the
# situation this repository is about.
#
# H20 is listed with ``link`` = "unknown" on purpose: public sources disagree
# (some list 900 GB/s NVLink, some list a PCIe attachment). Rather than pick a
# side, the field is left open and the planner asks you to look at
# ``nvidia-smi topo -m`` on the actual machine.

GPU_SPECS: Dict[str, Dict[str, object]] = {
    "A100-80": {
        "vram_gb": 80, "link": "nvlink", "link_bw_gbs": 600,
        "scaling": 0.90, "consumer": False,
        "note": "NVLink 3.0, 600 GB/s bidirectional",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.90x per-GPU efficiency)",
    },
    "A800-80": {
        "vram_gb": 80, "link": "nvlink", "link_bw_gbs": 400,
        "scaling": 0.88, "consumer": False,
        "note": "NVLink cut to 400 GB/s (A100 is 600); compute and HBM unchanged",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.88x per-GPU efficiency)",
    },
    "H100-80": {
        "vram_gb": 80, "link": "nvlink", "link_bw_gbs": 900,
        "scaling": 0.92, "consumer": False,
        "note": "NVLink 4.0, 900 GB/s bidirectional",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.92x per-GPU efficiency)",
    },
    "H800-80": {
        "vram_gb": 80, "link": "nvlink", "link_bw_gbs": 400,
        "scaling": 0.88, "consumer": False,
        "note": "NVLink cut to 400 GB/s (H100 is 900); compute and HBM unchanged",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.88x per-GPU efficiency)",
    },
    "H20-96": {
        "vram_gb": 96, "link": "unknown", "link_bw_gbs": None,
        "scaling": None, "consumer": False,
        "note": "public sources disagree: some list 900 GB/s NVLink, some a PCIe "
                "attachment. Check 'nvidia-smi topo -m' on your machine and pass "
                "--interconnect explicitly.",
        "grade": OPEN,
        "source": None,
    },
    "L40S-48": {
        "vram_gb": 48, "link": "pcie", "link_bw_gbs": 64,
        "scaling": 0.75, "consumer": False,
        "note": "PCIe Gen4 x16 (~64 GB/s bidirectional)",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.75x per-GPU efficiency)",
    },
    "A40-48": {
        "vram_gb": 48, "link": "pcie", "link_bw_gbs": 64,
        "scaling": 0.75, "consumer": False,
        "note": "PCIe Gen4 x16 (~64 GB/s bidirectional)",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.75x per-GPU efficiency)",
    },
    "RTX4090-24": {
        "vram_gb": 24, "link": "pcie", "link_bw_gbs": 64,
        "scaling": 0.75, "consumer": True,
        "note": "PCIe Gen4 x16; peer-to-peer access is restricted by the driver, "
                "so NCCL falls back to host memory",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.75x per-GPU efficiency)",
    },
    "RTX3090-24": {
        "vram_gb": 24, "link": "pcie", "link_bw_gbs": 64,
        "scaling": 0.75, "consumer": True,
        "note": "PCIe Gen4 x16; peer-to-peer access is restricted by the driver",
        "grade": MEASURED,
        "source": "public multi-GPU scaling catalogue (0.75x per-GPU efficiency)",
    },
    "RTXPRO5000-72": {
        "vram_gb": 72, "link": "pcie", "link_bw_gbs": 128,
        "scaling": 0.78, "consumer": False,
        "note": "RTX PRO 5000 Blackwell, 72 GB GDDR7 (73,415 MiB reported), "
                "PCIe 5.0 x16. First-party measured 2026-10-26: 300 W power "
                "wall is the per-GPU throughput ceiling — single-stream decode "
                "already draws ~300 W; c32 and c64 aggregate throughput are "
                "identical on one GPU. Dual-socket boards are 4+4 across NUMA, "
                "cross-group links are SYS.",
        "grade": MEASURED,
        "source": "first-party snapshot + single-card benchmark "
                  "(benchmarks/data/qwen38-27b-bf16-singlecard-three-machines.json)",
    },
    "RTX5090-96": {
        "vram_gb": 96, "link": "pcie", "link_bw_gbs": 128,
        "scaling": 0.75, "consumer": True,
        "note": "Modded RTX 5090 with 96 GB GDDR7 (aftermarket 96 GB variant), "
                "PCIe, 575 W power limit, nominal ~1792 GB/s memory bandwidth. "
                "Used as the A-machine in the first-party three-machine study; "
                "peer-to-peer behavior may differ from stock 5090.",
        "grade": DERIVABLE,
        "source": "first-party hardware inventory; bandwidth from public "
                  "RTX 5090 specifications",
    },
    "RTXPRO6000-96": {
        "vram_gb": 96, "link": "pcie", "link_bw_gbs": 128,
        "scaling": 0.78, "consumer": False,
        "note": "RTX PRO 6000 Blackwell, 96 GB GDDR7, PCIe 5.0 x16. Server "
                "Edition SKUs run a 600 W power limit and measured ~1597 GB/s "
                "memory bandwidth; Max-Q SKUs run ~300 W and hit the power "
                "wall before the VRAM wall.",
        "grade": MEASURED,
        "source": "first-party three-machine study: Server Edition 600 W / "
                  "~1597 GB/s measured; Max-Q power limit from vendor specs",
    },
}

# Generic fallbacks for machines that are not in the table above.
GENERIC_SPECS: Dict[str, Dict[str, object]] = {
    "nvlink": {"link": "nvlink", "link_bw_gbs": 400, "scaling": 0.88, "consumer": False},
    "pcie": {"link": "pcie", "link_bw_gbs": 64, "scaling": 0.75, "consumer": False},
    "pcie5": {"link": "pcie", "link_bw_gbs": 128, "scaling": 0.78, "consumer": False},
}

# --------------------------------------------------------------------------
# Quantisation table
# --------------------------------------------------------------------------
# ``bytes_per_param`` is the storage cost of one weight. Activation and KV
# precision are separate knobs and are NOT covered by this number.

QUANT_SPECS: Dict[str, Dict[str, object]] = {
    "fp16": {"bytes": 2.0, "note": "baseline; no quality risk"},
    "bf16": {"bytes": 2.0, "note": "baseline; no quality risk"},
    "fp8": {"bytes": 1.0, "note": "hardware dependent (Hopper / Ada); measure quality on your own data"},
    "int8": {"bytes": 1.0, "note": "check calibration coverage before trusting quality"},
    "awq": {"bytes": 0.55, "note": "4-bit weights plus metadata; keep the MoE router at higher precision"},
    "gptq": {"bytes": 0.55, "note": "4-bit weights plus metadata; keep the MoE router at higher precision"},
}

# --------------------------------------------------------------------------
# KV cache footprint
# --------------------------------------------------------------------------
# Bytes of KV cache per token at FP16, derived from published architectures:
#   kv_bytes_per_token = layers * kv_heads * head_dim * 2 (K and V) * 2 (fp16)
#
# These are geometry calculations, not measurements - hence ``derivable``.
# If your model is not listed, pass --kv-bytes-per-token; the tool will not
# guess for you.

# name -> parameter count (B), attention heads, KV heads, KV bytes/token @fp16
#
# ``heads`` and ``kv_heads`` are not decoration: tensor parallelism splits the
# attention heads across GPUs, so a TP value that does not divide both counts
# is not a slow configuration, it is an invalid one. The engine refuses to
# start. This is why the planner only ever proposes divisors.

MODEL_ARCH: Dict[str, Dict[str, float]] = {
    "qwen2.5-7b": {"params": 7, "heads": 28, "kv_heads": 4,
                   "kv_bytes_fp16": 28 * 4 * 128 * 2 * 2},       # 57 344 B
    "qwen2.5-14b": {"params": 14, "heads": 40, "kv_heads": 8,
                    "kv_bytes_fp16": 48 * 8 * 128 * 2 * 2},      # 196 608 B
    "qwen2.5-32b": {"params": 32, "heads": 64, "kv_heads": 8,
                    "kv_bytes_fp16": 64 * 8 * 128 * 2 * 2},      # 262 144 B
    "qwen2.5-72b": {"params": 72, "heads": 64, "kv_heads": 8,
                    "kv_bytes_fp16": 80 * 8 * 128 * 2 * 2},      # 327 680 B
    "llama-3-8b": {"params": 8, "heads": 32, "kv_heads": 8,
                   "kv_bytes_fp16": 32 * 8 * 128 * 2 * 2},       # 131 072 B
    "llama-3-70b": {"params": 70, "heads": 64, "kv_heads": 8,
                    "kv_bytes_fp16": 80 * 8 * 128 * 2 * 2},      # 327 680 B
    "llama-2-7b": {"params": 7, "heads": 32, "kv_heads": 32,
                   "kv_bytes_fp16": 32 * 32 * 128 * 2 * 2},      # 524 288 B
    "llama-2-13b": {"params": 13, "heads": 40, "kv_heads": 40,
                    "kv_bytes_fp16": 40 * 40 * 128 * 2 * 2},     # 819 200 B
    "llama-2-70b": {"params": 70, "heads": 64, "kv_heads": 8,
                    "kv_bytes_fp16": 80 * 8 * 128 * 2 * 2},      # 327 680 B
}

KV_BYTES_PER_TOKEN_FP16: Dict[str, int] = {
    name: int(arch["kv_bytes_fp16"]) for name, arch in MODEL_ARCH.items()
}

# Bytes of KV cache per token by dtype, relative to FP16.
KV_DTYPE_FACTOR: Dict[str, float] = {"fp16": 1.0, "bf16": 1.0, "fp8": 0.5}

# Framework and workspace overhead that never belongs to weights or KV:
# CUDA graphs, activations, fragmentation headroom.
FRAMEWORK_OVERHEAD_GB = 2.0

# Default concurrency slots by goal. These are starting points, not optima.
GOAL_SEQS: Dict[str, Dict[str, int]] = {
    "latency": {"single": 8, "replica": 32},
    "balanced": {"single": 32, "replica": 64},
    "throughput": {"single": 64, "replica": 128},
}

GB = 1024 ** 3


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Conclusion:
    """A single recommendation together with the basis it rests on."""

    claim: str
    grade: str
    reason: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"claim": self.claim, "grade": self.grade, "reason": self.reason}


@dataclass
class Plan:
    gpu: str
    count: int
    vram_gb: float
    model: str
    params_b: float
    quant: str
    ctx: int
    goal: str
    utilization: float
    tp: int = 1
    pp: int = 1
    dp: int = 1
    seqs: int = 32
    kv_dtype: str = "fp16"
    conclusions: List[Conclusion] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    memory: Optional[Dict[str, float]] = None
    benchmarks: Optional[List[Dict[str, object]]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "input": {
                "gpu": self.gpu, "count": self.count, "vram_gb": self.vram_gb,
                "model": self.model, "params_b": self.params_b, "quant": self.quant,
                "ctx": self.ctx, "goal": self.goal,
                "gpu_memory_utilization": self.utilization,
            },
            "recommendation": {
                "tensor_parallel": self.tp,
                "pipeline_parallel": self.pp,
                "replicas": self.dp,
                "max_num_seqs": self.seqs,
                "kv_cache_dtype": self.kv_dtype,
                "command": render_command(self),
            },
            "memory_gb": self.memory,
            "conclusions": [c.as_dict() for c in self.conclusions],
            "warnings": self.warnings,
            "measured_data": [e for e in (self.benchmarks or [])],
        }


# --------------------------------------------------------------------------
# Resolution helpers
# --------------------------------------------------------------------------

def resolve_gpu(gpu: str, vram_override: Optional[float]) -> Dict[str, object]:
    """Look up a GPU, falling back to generic interconnect classes."""
    key = gpu.strip()
    if key in GPU_SPECS:
        spec = dict(GPU_SPECS[key])
    elif key.lower() in GENERIC_SPECS:
        spec = dict(GENERIC_SPECS[key.lower()])
        spec.setdefault("vram_gb", vram_override or 24)
        spec["note"] = f"generic {key} profile; fill in real numbers from your machine"
        spec["grade"] = OPEN
        spec["source"] = None
        spec["consumer"] = False
    else:
        raise SystemExit(
            f"unknown GPU '{gpu}'. Known values: {', '.join(sorted(GPU_SPECS))} "
            f"or generic classes {', '.join(sorted(GENERIC_SPECS))}. "
            f"Use --list-gpus for the full table."
        )
    if vram_override is not None:
        spec["vram_gb"] = vram_override
    return spec


def resolve_kv_bytes(model: str, params_b: float, override: Optional[float],
                     kv_dtype: str) -> Tuple[Optional[float], str]:
    """Return (bytes of KV per token, evidence grade). None means unknown."""
    if override is not None:
        return override, MEASURED if override > 0 else OPEN
    key = model.strip().lower()
    if key in KV_BYTES_PER_TOKEN_FP16:
        factor = KV_DTYPE_FACTOR.get(kv_dtype, 1.0)
        return KV_BYTES_PER_TOKEN_FP16[key] * factor, DERIVABLE
    # Unknown geometry. Refuse to guess: an invented KV number would silently
    # decide whether the model "fits", which is the one thing that must not be
    # wrong.
    return None, OPEN


def tp_ceiling(spec: Dict[str, object], group: int, count: int,
               interconnect_override: Optional[str]) -> Tuple[int, Conclusion]:
    """How far tensor parallelism can go before the interconnect fights back."""
    link = interconnect_override or str(spec.get("link", "pcie"))
    consumer = bool(spec.get("consumer"))
    bw = spec.get("link_bw_gbs")

    if link == "nvlink":
        ceiling = min(count, 8)
        reason = (
            f"NVLink at {bw} GB/s bidirectional; TP can span the whole node "
            f"(per-GPU efficiency {spec.get('scaling')})."
        )
        grade = str(spec.get("grade", MEASURED))
    elif link == "pcie-same-switch":
        ceiling = min(count, group if group else count, 4)
        reason = (
            "All GPUs sit under one PCIe switch. TP stays inside that switch; "
            "beyond 4-way the all-reduce cost usually exceeds the gain."
        )
        grade = DERIVABLE
    elif link == "pcie-cross-switch":
        ceiling = max(1, min(count, group if group else 1))
        reason = (
            "GPUs span PCIe switches or CPU sockets. TP is capped at the group "
            "size; anything beyond that must go through pipeline parallelism."
        )
        grade = DERIVABLE
    else:
        ceiling = min(count, 2)
        reason = (
            "Interconnect is unknown. TP is capped at 2 as a conservative "
            "default - run 'nvidia-smi topo -m' and pass --interconnect."
        )
        grade = OPEN

    if consumer:
        ceiling = min(ceiling, 2)
        reason += (
            " Consumer cards additionally have driver-restricted P2P access, so "
            "NCCL routes through host memory; TP beyond 2 rarely pays off."
        )
        grade = MEASURED if grade == MEASURED else grade

    return max(1, ceiling), Conclusion(f"TP ceiling = {max(1, ceiling)}", grade, reason)


# --------------------------------------------------------------------------
# Core planning
# --------------------------------------------------------------------------

def plan(args: argparse.Namespace) -> Plan:
    spec = resolve_gpu(args.gpus, args.vram)
    count = max(1, args.count)
    vram = float(spec["vram_gb"])
    quant = args.quant.lower()
    if quant not in QUANT_SPECS:
        raise SystemExit(
            f"unknown quantisation '{args.quant}'. "
            f"Known values: {', '.join(sorted(QUANT_SPECS))}"
        )
    bytes_per_param = float(QUANT_SPECS[quant]["bytes"])

    params_b = args.params if args.params is not None else guess_params(args.model)
    if params_b is None:
        raise SystemExit(
            f"model '{args.model}' is not in the built-in table. "
            f"Pass --params <billions> explicitly."
        )

    kv_dtype = "fp8" if args.ctx >= 65536 else args.kv_dtype
    kv_bytes, kv_grade = resolve_kv_bytes(args.model, params_b, args.kv_bytes_per_token, kv_dtype)

    seqs = args.concurrency if args.concurrency else (
        GOAL_SEQS[args.goal]["replica"] if count > 1 else GOAL_SEQS[args.goal]["single"]
    )

    p = Plan(
        gpu=args.gpus, count=count, vram_gb=vram, model=args.model,
        params_b=params_b, quant=quant, ctx=args.ctx, goal=args.goal,
        utilization=args.util, seqs=seqs, kv_dtype=kv_dtype,
    )
    if getattr(args, "benchmarks", None):
        p.benchmarks = load_benchmarks(args.benchmarks)

    # --- memory budget -----------------------------------------------------
    usable_per_gpu = vram * args.util - FRAMEWORK_OVERHEAD_GB
    weights_total_gb = params_b * bytes_per_param
    kv_total_gb = (
        (kv_bytes * args.ctx * seqs) / GB if kv_bytes is not None else None
    )
    p.memory = {
        "vram_per_gpu_gb": round(vram, 2),
        "usable_per_gpu_gb": round(usable_per_gpu, 2),
        "weights_total_gb": round(weights_total_gb, 2),
        "kv_total_gb": round(kv_total_gb, 2) if kv_total_gb is not None else None,
        "framework_overhead_gb": FRAMEWORK_OVERHEAD_GB,
    }

    p.conclusions.append(Conclusion(
        f"Weights ≈ {weights_total_gb:.1f} GB at {quant}",
        DERIVABLE,
        f"{params_b:.0f}B parameters × {bytes_per_param} bytes per parameter. "
        f"{QUANT_SPECS[quant]['note']}.",
    ))

    if kv_total_gb is None:
        p.conclusions.append(Conclusion(
            "KV cache footprint: unknown",
            OPEN,
            "The architecture is not in the built-in table and this project does "
            "not invent numbers. Measure it, then pass --kv-bytes-per-token: "
            "run one request and read the 'GPU KV cache size' line in the vLLM "
            "startup log, divided by max-num-seqs and max-model-len.",
        ))
        kv_total_gb = 0.0
    else:
        p.conclusions.append(Conclusion(
            f"KV cache ≈ {kv_total_gb:.1f} GB at ctx={args.ctx}, seqs={seqs}",
            kv_grade,
            f"{kv_bytes:.0f} bytes/token × {args.ctx} tokens × {seqs} slots, "
            f"kv dtype {kv_dtype}.",
        ))

    # --- interconnect ------------------------------------------------------
    ceiling, ceiling_conclusion = tp_ceiling(spec, args.group, count, args.interconnect)
    p.conclusions.append(ceiling_conclusion)
    if str(spec.get("link")) == "unknown" or spec.get("scaling") is None:
        p.warnings.append(
            f"{args.gpus}: {spec.get('note')} Run 'nvidia-smi topo -m' on the "
            f"real machine and pass --interconnect to override."
        )

    # --- strategy ----------------------------------------------------------
    fits_single = (weights_total_gb + kv_total_gb) <= usable_per_gpu

    if fits_single:
        p.dp = count
        p.tp = p.pp = 1
        p.conclusions.append(Conclusion(
            f"Use {count} independent replicas instead of tensor parallelism",
            MEASURED,
            "Weights plus KV fit on one GPU, so each card can hold a complete "
            "copy. Replicas exchange nothing, so throughput scales close to "
            "linearly, while TP all-reduces at every layer and hands most of "
            "the gain back to the interconnect.",
        ))
    else:
        # Smallest TP that both fits in memory and is a legal split of the
        # attention heads. Legal first, then large enough - a TP value the
        # engine rejects is worse than a slightly oversized one.
        need_total = weights_total_gb + kv_total_gb
        arch = MODEL_ARCH.get(args.model.strip().lower())
        heads = arch["heads"] if arch else None
        kv_heads = arch["kv_heads"] if arch else None
        candidates = valid_tp_values(heads, kv_heads, count, ceiling)

        if arch is None:
            p.warnings.append(
                "Architecture not in the built-in table, so legal TP values "
                "fall back to powers of two. Check that your model's head "
                "count is divisible by the proposed TP."
            )

        fitting = [t for t in candidates if need_total / t <= usable_per_gpu]
        if fitting:
            p.tp = fitting[0]
        else:
            # Nothing fits by splitting tensors alone.
            p.tp = candidates[-1] if candidates else 1
        remaining = max(1, count // p.tp)

        if need_total / p.tp > usable_per_gpu:
            # TP alone (capped by the interconnect) is not enough: add PP.
            pp_needed = 1
            while pp_needed <= remaining and need_total / (p.tp * pp_needed) > usable_per_gpu:
                pp_needed += 1
            p.pp = max(1, min(pp_needed, remaining))
            p.conclusions.append(Conclusion(
                f"TP={p.tp} is capped by the interconnect, so add PP={p.pp}",
                DERIVABLE,
                "Pipeline parallelism splits layers instead of tensors: it "
                "passes activations at stage boundaries only, which is far "
                "cheaper over a slow link than a per-layer all-reduce.",
            ))
            p.warnings.append(
                "PP only helps throughput under enough concurrency - it does "
                "nothing for single-request latency. If your traffic is a "
                "handful of concurrent users, cut context or quantise instead."
            )
        p.dp = max(1, count // (p.tp * p.pp))

        if fitting:
            fit_note = (
                f"TP={p.tp} is the smallest legal split where weights plus KV "
                f"fit in {usable_per_gpu:.1f} GB."
            )
        else:
            fit_note = (
                f"No legal TP value up to the interconnect ceiling of {ceiling} "
                f"makes it fit; pipeline parallelism is taking up the remainder."
            )
        p.conclusions.append(Conclusion(
            f"TP={p.tp}, PP={p.pp}, replicas={p.dp}",
            DERIVABLE,
            f"Weights plus KV need {need_total:.1f} GB; one GPU offers "
            f"{usable_per_gpu:.1f} GB usable at --gpu-memory-utilization "
            f"{args.util:.2f}. " + fit_note,
        ))

        if p.dp > 1:
            p.conclusions.append(Conclusion(
                f"Remaining GPUs form {p.dp} replicas of the TP={p.tp} group",
                DERIVABLE,
                "Once the model fits, extra GPUs are worth more as independent "
                "replicas than as a wider TP group.",
            ))

    # --- KV dtype ----------------------------------------------------------
    if kv_dtype == "fp8" and args.kv_dtype != "fp8":
        p.conclusions.append(Conclusion(
            "KV cache dtype forced to fp8",
            DERIVABLE,
            f"Context is {args.ctx} tokens; at that length KV dominates the "
            "footprint and fp8 is the cheapest saving available.",
        ))

    # --- standing warnings -------------------------------------------------
    if p.dp > 1:
        p.warnings.append(
            "Replicas are separate vLLM processes. Pin each with "
            "CUDA_VISIBLE_DEVICES, give each its own --port, and put a load "
            "balancer in front."
        )
    if bool(spec.get("consumer")) and p.tp > 1:
        p.warnings.append(
            "Consumer cards have driver-restricted P2P access; NCCL will fall "
            "back to host memory and TP may be slower than you expect. Verify "
            "with a real benchmark before committing to this layout."
        )
    if kv_bytes is None:
        p.warnings.append(
            "KV footprint was unknown, so the fit check covers weights only. "
            "Treat the result as a lower bound."
        )
    p.warnings.append(
        "This is a structurally sound starting point, not an optimum. Measure "
        "it on the real machine and submit the numbers back - see "
        "benchmarks/METHODOLOGY.md."
    )

    return p


def guess_params(model: str) -> Optional[float]:
    """Infer parameter count from a model name; None when it is not known."""
    arch = MODEL_ARCH.get(model.strip().lower())
    return float(arch["params"]) if arch else None


# --------------------------------------------------------------------------
# Measured benchmark data (benchmarks/data/*.json)
# --------------------------------------------------------------------------

BENCHMARK_SCHEMA = "lltg.benchmark.v1"


def load_benchmarks(path: str) -> List[Dict[str, object]]:
    """Load benchmark entries from a JSON file or a directory of them.

    Refuses anything that does not carry the expected schema tag. Silently
    parsing a foreign file would put unvetted numbers next to the plan,
    which is the one thing this project must not do.
    """
    import glob
    import os

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
        if not files:
            raise SystemExit(f"no *.json files found in '{path}'")
    else:
        files = [path]

    entries: List[Dict[str, object]] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{f}: not valid JSON ({exc})") from exc
        if not isinstance(data, dict) or data.get("schema") != BENCHMARK_SCHEMA:
            raise SystemExit(
                f"{f}: expected schema '{BENCHMARK_SCHEMA}', got "
                f"'{data.get('schema') if isinstance(data, dict) else type(data).__name__}'. "
                f"See benchmarks/data/SCHEMA.md."
            )
        data["_file"] = os.path.basename(f)
        entries.append(data)
    return entries


def render_benchmarks(entries: List[Dict[str, object]]) -> str:
    out: List[str] = ["MEASURED DATA (loaded via --benchmarks)", ""]
    for e in entries:
        hw = e.get("hardware", {})
        model = e.get("model", {})
        src = e.get("source", {})
        out.append(f"  [{e.get('kind', '?')}] {e.get('title', e.get('id', '?'))}")
        out.append(f"      hardware : {hw.get('gpus', '?')} - {hw.get('interconnect', '?')}")
        model_line = f"      model    : {model.get('name', '?')}"
        if model.get("quantization"):
            model_line += f", {model.get('quantization')}"
        out.append(model_line)
        if e.get("transferable"):
            out.append(f"      transfers: {e['transferable']}")
        if src.get("project"):
            out.append(f"      source   : {src['project']}")
        out.append(f"      file     : {e.get('_file')}")
        out.append("")
    out.append(
        "These entries describe THEIR test conditions. They anchor the plan "
        "above; they do not replace a measurement on your machine."
    )
    return "\n".join(out)


def valid_tp_values(heads: Optional[float], kv_heads: Optional[float],
                    count: int, ceiling: int) -> List[int]:
    """Tensor-parallel sizes the engine will actually accept.

    TP splits attention heads across GPUs, so both ``num_attention_heads`` and
    ``num_key_value_heads`` must be divisible by it. When the architecture is
    unknown we fall back to powers of two, which is what the engines are
    optimised for, and say so.
    """
    upper = max(1, min(count, ceiling))
    if heads and kv_heads:
        h, k = int(heads), int(kv_heads)
        return [t for t in range(1, upper + 1) if h % t == 0 and k % t == 0]
    return [t for t in range(1, upper + 1) if (t & (t - 1)) == 0]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_command(p: Plan) -> str:
    lines: List[str] = []
    if p.dp > 1 and p.tp == 1 and p.pp == 1:
        lines.append(f"# {p.dp} independent replicas, one process per GPU")
        lines.append("CUDA_VISIBLE_DEVICES=0 vllm serve <your-model> \\")
    elif p.dp > 1:
        lines.append(f"# {p.dp} replicas, each using TP={p.tp}"
                     + (f" PP={p.pp}" if p.pp > 1 else ""))
        lines.append("CUDA_VISIBLE_DEVICES=0,1 vllm serve <your-model> \\")
    else:
        lines.append("vllm serve <your-model> \\")
    if p.tp > 1:
        lines.append(f"  --tensor-parallel-size {p.tp} \\")
    if p.pp > 1:
        lines.append(f"  --pipeline-parallel-size {p.pp} \\")
    lines.append(f"  --gpu-memory-utilization {p.utilization:.2f} \\")
    lines.append(f"  --max-model-len {p.ctx} \\")
    lines.append(f"  --max-num-seqs {p.seqs} \\")
    if p.kv_dtype == "fp8":
        lines.append("  --kv-cache-dtype fp8 \\")
    lines.append("  --enable-prefix-caching \\")
    lines.append("  --enable-chunked-prefill \\")
    lines.append("  --host 0.0.0.0 --port 8000")
    if p.dp > 1:
        lines.append("")
        lines.append("# repeat for each replica: shift CUDA_VISIBLE_DEVICES, "
                     "increment --port")
    return "\n".join(lines)


GRADE_MARK = {MEASURED: "[measured]", DERIVABLE: "[derivable]", OPEN: "[open]"}


def render_text(p: Plan) -> str:
    out: List[str] = []
    out.append("=" * 68)
    out.append("  Local LLM parallelism planner")
    out.append("=" * 68)
    out.append("")
    out.append("INPUT")
    out.append(f"  GPUs            : {p.count} x {p.gpu} ({p.vram_gb:.0f} GB each)")
    out.append(f"  Model           : {p.model} ({p.params_b:.0f}B, {p.quant})")
    out.append(f"  Context / goal  : {p.ctx} tokens / {p.goal}")
    out.append(f"  Utilisation     : {p.utilization:.2f}")
    out.append("")
    out.append("MEMORY BUDGET (GB)")
    m = p.memory or {}
    out.append(f"  VRAM per GPU          : {m.get('vram_per_gpu_gb', 0):>8.1f}")
    out.append(f"  Usable per GPU        : {m.get('usable_per_gpu_gb', 0):>8.1f}")
    out.append(f"  Weights (total)       : {m.get('weights_total_gb', 0):>8.1f}")
    kv = m.get("kv_total_gb")
    out.append(f"  KV cache (total)      : {('%8.1f' % kv) if kv is not None else ' unknown'}")
    out.append("")
    out.append("CONCLUSIONS")
    for c in p.conclusions:
        out.append(f"  {GRADE_MARK.get(c.grade, '[?]')} {c.claim}")
        if c.reason:
            out.append(f"      {c.reason}")
    out.append("")
    out.append("RECOMMENDED LAYOUT")
    out.append(f"  TP (tensor)   : {p.tp}")
    out.append(f"  PP (pipeline) : {p.pp}")
    out.append(f"  Replicas      : {p.dp}")
    out.append(f"  max-num-seqs  : {p.seqs}")
    out.append(f"  kv-cache-dtype: {p.kv_dtype}")
    out.append("")
    out.append("LAUNCH COMMAND")
    for line in render_command(p).split("\n"):
        out.append("  " + line)
    out.append("")
    if p.warnings:
        out.append("WATCH OUT")
        for w in p.warnings:
            out.append(f"  - {w}")
        out.append("")
    out.append("Grades: [measured] = verifiable measurement  "
               "[derivable] = from specs  [open] = data missing, left blank")
    return "\n".join(out)


def render_gpu_table() -> str:
    lines = ["Known GPUs (pass with --gpus):", ""]
    lines.append(f"  {'name':<14}{'VRAM':>6}{'link':>10}{'bw GB/s':>9}{'scale':>7}  note")
    lines.append("  " + "-" * 92)
    for name, s in sorted(GPU_SPECS.items()):
        bw = s.get("link_bw_gbs")
        sc = s.get("scaling")
        lines.append(
            f"  {name:<14}{str(s['vram_gb']):>6}{str(s['link']):>10}"
            f"{(f'{bw:.0f}' if bw else '?'):>9}"
            f"{(f'{sc:.2f}' if sc else '?'):>7}  {s['note']}"
        )
    lines.append("")
    lines.append("Generic classes: nvlink, pcie (Gen4), pcie5 (Gen5)")
    lines.append("Built-in models: " + ", ".join(sorted(KV_BYTES_PER_TOKEN_FP16)))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="planner.py",
        description="Recommend a TP / PP / replica layout for local LLM serving.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every conclusion carries an evidence grade. This tool does not "
               "invent numbers; unknown values are reported as [open].",
    )
    ap.add_argument("--gpus", default="A800-80", help="GPU model, e.g. A800-80, H800-80, RTX4090-24, pcie")
    ap.add_argument("--count", type=int, default=8, help="number of GPUs in the node")
    ap.add_argument("--vram", type=float, default=None, help="override VRAM per GPU (GB)")
    ap.add_argument("--model", default="qwen2.5-72b", help="model name (built-in table) or any label with --params")
    ap.add_argument("--params", type=float, default=None, help="parameter count in billions (overrides --model)")
    ap.add_argument("--quant", default="fp16", help="fp16 | bf16 | fp8 | int8 | awq | gptq")
    ap.add_argument("--ctx", type=int, default=8192, help="target context length (max-model-len)")
    ap.add_argument("--concurrency", type=int, default=None, help="max-num-seqs; default derived from --goal")
    ap.add_argument("--goal", default="balanced", help="latency | balanced | throughput")
    ap.add_argument("--util", type=float, default=0.90, help="--gpu-memory-utilization")
    ap.add_argument("--group", type=int, default=0,
                    help="GPUs per PCIe switch / NUMA group (for cross-switch topologies)")
    ap.add_argument("--interconnect", default=None,
                    help="override detected link: nvlink | pcie-same-switch | pcie-cross-switch")
    ap.add_argument("--kv-dtype", default="fp16", help="fp16 | fp8 KV cache precision")
    ap.add_argument("--kv-bytes-per-token", type=float, default=None,
                    help="measured KV bytes per token; overrides the built-in table")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--list-gpus", action="store_true", help="print the hardware table and exit")
    ap.add_argument("--benchmarks", default=None,
                    help="path to a benchmark JSON file or a directory of them "
                         "(e.g. benchmarks/data); loaded entries are attached "
                         "to the plan as measured data")
    ap.add_argument("--list-benchmarks", action="store_true",
                    help="with --benchmarks: list the entries and exit")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_gpus:
        print(render_gpu_table())
        return 0
    if args.goal not in GOAL_SEQS:
        raise SystemExit(f"unknown goal '{args.goal}'. Known values: {', '.join(GOAL_SEQS)}")
    if not 0.5 <= args.util <= 0.98:
        raise SystemExit("--util must be between 0.50 and 0.98")

    benchmarks: Optional[List[Dict[str, object]]] = None
    if args.list_benchmarks:
        if not args.benchmarks:
            raise SystemExit("--list-benchmarks requires --benchmarks <path>")
        benchmarks = load_benchmarks(args.benchmarks)
        for e in benchmarks:
            hw = e.get("hardware", {})
            print(f"[{e.get('kind', '?')}] {e.get('id', '?')}  "
                  f"({hw.get('gpus', '?')})  {e.get('title', '')}")
        return 0

    p = plan(args)
    if args.json:
        print(json.dumps(p.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_text(p))
        if benchmarks:
            print()
            print(render_benchmarks(benchmarks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
