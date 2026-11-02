#!/usr/bin/env python3
"""detect_topology.py - collect the hardware facts the planner needs.

Runs a small set of read-only commands (nvidia-smi, numactl, lspci) and emits
one JSON object that can be pasted into a recipe or fed to planner.py. No
network, standard library only.

Usage
-----
    python tools/detect_topology.py
    python tools/detect_topology.py --json > topology.json
    python tools/detect_topology.py --planner
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

SCHEMA = "lltg.topology.v1"


def run(cmd: List[str]) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout or ""
    except Exception:
        return ""


def _num(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value.upper() in ("N/A", "[N/A]", "NOT SUPPORTED"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_gpus(text: str) -> List[Dict[str, object]]:
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx = _num(parts[0])
        if idx is None:
            continue
        gpus.append({
            "index": int(idx),
            "name": parts[1],
            "memory_total_mib": _num(parts[2]),
            "power_limit_w": _num(parts[3]),
            "power_max_w": _num(parts[4]),
        })
    return gpus


def parse_topo(text: str) -> Dict[str, object]:
    """Parse `nvidia-smi topo -m` into a link map and connectivity groups."""
    rows: List[List[str]] = []
    gpu_count = 0
    for line in text.splitlines():
        if line.strip().startswith("GPU0") and "CPU Affinity" in line:
            header = line.split()
            gpu_count = sum(1 for h in header if re.fullmatch(r"GPU\d+", h))
            continue
        if re.match(r"^GPU\d+\s", line):
            rows.append(line.split())
    links: Dict[str, str] = {}
    for i, row in enumerate(rows[:gpu_count]):
        for j in range(gpu_count):
            if i == j:
                continue
            # row: GPUi [X] link0 link1 ...; columns start at index 1
            if j + 1 < len(row):
                links[f"{i}-{j}"] = row[j + 1]
    # Union GPUs connected by anything other than SYS (SYS crosses CPU sockets).
    parent = list(range(gpu_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    has_nvlink = False
    for key, link in links.items():
        a, b = (int(x) for x in key.split("-"))
        if link.startswith("NV"):
            has_nvlink = True
        if link != "SYS":
            union(a, b)
    groups: Dict[int, List[int]] = {}
    for i in range(gpu_count):
        groups.setdefault(find(i), []).append(i)
    group_list = sorted((sorted(v) for v in groups.values()), key=lambda g: g[0])
    cross_group = len(group_list) > 1
    return {
        "gpu_count": gpu_count,
        "links": links,
        "groups": group_list,
        "cross_group": cross_group,
        "has_nvlink": has_nvlink,
    }


def parse_numa(text: str) -> List[Dict[str, object]]:
    nodes: Dict[int, Dict[str, object]] = {}
    for line in text.splitlines():
        m = re.match(r"node (\d+) cpus:\s*(.*)", line)
        if m:
            nodes.setdefault(int(m.group(1)), {})["cpus"] = m.group(2).strip()
        m = re.match(r"node (\d+) size:\s*(\d+)\s*MB", line)
        if m:
            nodes.setdefault(int(m.group(1)), {})["size_mb"] = int(m.group(2))
    return [dict(node=k, **v) for k, v in sorted(nodes.items())]


def summarize(gpus: List[Dict[str, object]], topo: Dict[str, object],
              numa: List[Dict[str, object]]) -> Dict[str, object]:
    total_vram = sum(g.get("memory_total_mib") or 0 for g in gpus) / 1024
    limits = [g.get("power_limit_w") for g in gpus if g.get("power_limit_w")]
    if topo.get("has_nvlink"):
        interconnect = "nvlink"
    elif topo.get("cross_group"):
        interconnect = "pcie-cross-group"
    else:
        interconnect = "pcie-same-group"
    return {
        "gpu_count": len(gpus),
        "vram_gb_total": round(total_vram, 1),
        "power_limit_w_min": min(limits) if limits else None,
        "power_limit_w_max": max(limits) if limits else None,
        "interconnect": interconnect,
        "numa_nodes": len(numa),
        "groups": topo.get("groups", []),
    }


def warnings(gpus: List[Dict[str, object]], topo: Dict[str, object]) -> List[str]:
    out = []
    if topo.get("cross_group"):
        out.append("GPUs span multiple PCIe/NUMA groups; keep TP inside one group by default.")
    if not topo.get("has_nvlink"):
        out.append("No NVLink detected; PCIe all-reduce costs dominate multi-GPU scaling.")
    limits = [g.get("power_limit_w") for g in gpus if g.get("power_limit_w")]
    if limits and max(limits) <= 320:
        out.append("Power limit <= 320 W per GPU: expect a throughput ceiling at high concurrency.")
    return out


def planner_hint(gpus: List[Dict[str, object]], topo: Dict[str, object]) -> Dict[str, object]:
    name = gpus[0]["name"] if gpus else ""
    key = ""
    m = re.search(r"RTX PRO (\d+)", str(name), re.I)
    if m:
        key = f"RTXPRO{m.group(1)}-72"
    elif "4090" in str(name):
        key = "RTX4090-24"
    elif "5090" in str(name):
        key = "RTX5090-96"
    groups = topo.get("groups") or []
    return {
        "gpus": key or None,
        "count": len(gpus),
        "group": len(groups[0]) if groups else 0,
    }


def collect() -> Dict[str, object]:
    gpu_text = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,power.limit,power.max_limit",
        "--format=csv,noheader,nounits",
    ])
    topo_text = run(["nvidia-smi", "topo", "-m"])
    numa_text = run(["numactl", "-H"])
    lspci_text = run(["lspci"])
    gpus = parse_gpus(gpu_text)
    topo = parse_topo(topo_text)
    numa = parse_numa(numa_text)
    return {
        "schema": SCHEMA,
        "gpus": gpus,
        "topology": topo,
        "numa": numa,
        "lspci_nvidia": [line.strip() for line in lspci_text.splitlines() if "nvidia" in line.lower()][:16],
        "summary": summarize(gpus, topo, numa),
        "warnings": warnings(gpus, topo),
        "planner": planner_hint(gpus, topo),
    }


def render(data: Dict[str, object]) -> str:
    s = data["summary"]
    lines = [
        f"GPUs: {s['gpu_count']}  VRAM: {s['vram_gb_total']} GB  "
        f"power: {s['power_limit_w_min']}-{s['power_limit_w_max']} W",
        f"interconnect: {s['interconnect']}  NUMA nodes: {s['numa_nodes']}  "
        f"groups: {s['groups']}",
    ]
    for g in data["gpus"]:
        lines.append(f"  GPU{g['index']}: {g['name']}  "
                     f"{(g['memory_total_mib'] or 0)/1024:.1f} GB  "
                     f"{g['power_limit_w']} W")
    for w in data["warnings"]:
        lines.append("WARNING: " + w)
    p = data["planner"]
    if p.get("gpus"):
        lines.append(f"planner hint: --gpus {p['gpus']} --count {p['count']} "
                     f"--group {p['group']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Collect GPU topology facts as JSON.")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--planner", action="store_true", help="print a planner.py argument hint")
    args = ap.parse_args(argv)
    data = collect()
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.planner:
        p = data["planner"]
        print(f"--gpus {p.get('gpus') or 'pcie'} --count {p['count']} --group {p['group']}")
    else:
        print(render(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
