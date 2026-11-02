#!/usr/bin/env python3
"""run_matrix.py - run the standard C1/C8/C32/C64 + 32K + ShareGPT matrix.

Wraps `vllm bench serve` so the METHODOLOGY template becomes one command. It
does not start the model server; point --base-url at a running OpenAI-compatible
endpoint.

Usage
-----
    python bench/run_matrix.py --base-url http://127.0.0.1:8000 \
        --model qwen3.8-27b --tokenizer /models/Qwen3.8-27B --out results/qwen

    # run the client from a container (mounts the tokenizer and output dir)
    python bench/run_matrix.py --runner docker \
        --docker-image vllm/vllm-openai:v0.32.0 \
        --mount /models:/models:ro --mount $PWD/results:/results \
        --model /models/Qwen3.8-27B --tokenizer /models/Qwen3.8-27B \
        --base-url http://127.0.0.1:8000 --out results/qwen

    python bench/run_matrix.py --dry-run ...
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional

LABELS = ["c1", "c8", "c32", "c64", "ctx32k_c1", "ctx32k_c4", "sharegpt_c64"]

MATRIX = {
    "c1": (1024, 256, 1, 200, 10, "random"),
    "c8": (1024, 256, 8, 200, 10, "random"),
    "c32": (1024, 256, 32, 320, 10, "random"),
    "c64": (1024, 256, 64, 640, 10, "random"),
    "ctx32k_c1": (32768, 256, 1, 20, 2, "random"),
    "ctx32k_c4": (32768, 256, 4, 40, 2, "random"),
    "sharegpt_c64": (0, 0, 64, 1000, 10, "sharegpt"),
}


def build_command(args: argparse.Namespace, label: str, rep: int) -> List[str]:
    inlen, outlen, conc, prompts, warm, dataset = MATRIX[label]
    result_dir = os.path.join(args.result_dir, f"{label}-r{rep}")
    bench = [
        "bench", "serve",
        "--backend", "openai",
        "--base-url", args.base_url,
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--num-prompts", str(prompts),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--num-warmups", str(warm),
        "--ignore-eos",
        "--temperature", "0",
        "--seed", str(args.seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--label", f"{label}-r{rep}",
        "--save-result",
        "--result-dir", result_dir,
    ]
    if dataset == "sharegpt":
        bench += ["--dataset-name", "sharegpt", "--dataset-path", args.sharegpt]
    else:
        bench += ["--dataset-name", "random",
                  "--random-input-len", str(inlen),
                  "--random-output-len", str(outlen)]
    if args.runner == "docker":
        cmd = ["docker", "run", "--rm", "--network", "host"]
        for mount in args.mount:
            cmd += ["-v", mount]
        cmd += ["--entrypoint", "vllm", args.docker_image]
        return cmd + bench
    return ["vllm"] + bench


def find_result(result_dir: str) -> Optional[Dict[str, object]]:
    if not os.path.isdir(result_dir):
        return None
    for name in os.listdir(result_dir):
        if name.endswith(".json"):
            try:
                with open(os.path.join(result_dir, name)) as fh:
                    return json.load(fh)
            except Exception:
                return None
    return None


def summarize(rows: List[Dict[str, object]], out_dir: str) -> Dict[str, object]:
    by_label: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        label = str(row.get("label", "")).split("-r")[0]
        by_label.setdefault(label, []).append(row)
    summary = {}
    for label, runs in by_label.items():
        def vals(key: str) -> List[float]:
            return [float(r.get(key) or 0) for r in runs]
        out = vals("output_throughput")
        summary[label] = {
            "runs": len(runs),
            "output_throughput_mean": round(statistics.mean(out), 2) if out else 0,
            "output_throughput_min": round(min(out), 2) if out else 0,
            "output_throughput_max": round(max(out), 2) if out else 0,
            "ttft_p50_mean": round(statistics.mean(vals("median_ttft_ms")), 1),
            "tpot_p50_mean": round(statistics.mean(vals("median_tpot_ms")), 1),
            "completed": sum(int(r.get("completed") or 0) for r in runs),
            "failed": sum(int(r.get("failed") or 0) for r in runs),
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    lines = ["| label | runs | output tok/s (mean) | min | max | TTFT p50 | TPOT p50 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        if label not in summary:
            continue
        s = summary[label]
        lines.append(f"| {label} | {s['runs']} | {s['output_throughput_mean']:.1f} | "
                     f"{s['output_throughput_min']:.1f} | {s['output_throughput_max']:.1f} | "
                     f"{s['ttft_p50_mean']:.0f} ms | {s['tpot_p50_mean']:.1f} ms |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the standard LLM benchmark matrix.")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--out", default=None, help="summary output directory (default: result-dir)")
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sharegpt", default="/trace/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--runner", choices=["local", "docker"], default="local")
    ap.add_argument("--docker-image", default="vllm/vllm-openai:v0.32.0")
    ap.add_argument("--mount", action="append", default=[], help="host:container[:ro]")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    unknown = [x for x in labels if x not in MATRIX]
    if unknown:
        raise SystemExit(f"unknown label(s): {', '.join(unknown)}")
    os.makedirs(args.result_dir, exist_ok=True)
    out_dir = args.out or args.result_dir
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for rep in range(1, args.reps + 1):
        for label in labels:
            cmd = build_command(args, label, rep)
            print("$ " + " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"WARNING: {label} r{rep} exited {rc}", file=sys.stderr)
            result = find_result(os.path.join(args.result_dir, f"{label}-r{rep}"))
            if result:
                rows.append(result)
    if not args.dry_run and rows:
        summary = summarize(rows, out_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
