#!/usr/bin/env python3
"""Tests for bench/run_matrix.py command construction and summarization."""

import argparse
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

import run_matrix as rm  # noqa: E402


def ns(**kw):
    base = dict(
        base_url="http://127.0.0.1:8000",
        model="m",
        tokenizer="/models/m",
        result_dir="/tmp/r",
        runner="local",
        mount=[],
        docker_image="img",
        sharegpt="/trace/sharegpt.json",
        seed=42,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class TestCommands(unittest.TestCase):
    def test_random_command(self):
        cmd = rm.build_command(ns(), "c8", 2)
        self.assertEqual(cmd[0], "vllm")
        self.assertIn("--max-concurrency", cmd)
        self.assertEqual(cmd[cmd.index("--max-concurrency") + 1], "8")
        self.assertNotIn("--dataset-path", cmd)

    def test_sharegpt_command(self):
        cmd = rm.build_command(ns(), "sharegpt_c64", 1)
        self.assertEqual(cmd[cmd.index("--dataset-name") + 1], "sharegpt")
        self.assertEqual(cmd[cmd.index("--dataset-path") + 1], "/trace/sharegpt.json")

    def test_docker_command(self):
        cmd = rm.build_command(ns(runner="docker", mount=["/m:/models:ro"]), "c1", 1)
        self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
        self.assertIn("/m:/models:ro", cmd)
        self.assertEqual(cmd[cmd.index("--entrypoint") + 1], "vllm")

    def test_summarize(self):
        rows = [
            {"label": "c1-r1", "output_throughput": 90, "median_ttft_ms": 200,
             "median_tpot_ms": 10, "completed": 10, "failed": 0},
            {"label": "c1-r2", "output_throughput": 92, "median_ttft_ms": 210,
             "median_tpot_ms": 11, "completed": 10, "failed": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = rm.summarize(rows, tmp)
            self.assertEqual(summary["c1"]["runs"], 2)
            self.assertEqual(summary["c1"]["output_throughput_min"], 90)
            self.assertTrue(os.path.exists(os.path.join(tmp, "summary.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "summary.md")))


if __name__ == "__main__":
    unittest.main()
