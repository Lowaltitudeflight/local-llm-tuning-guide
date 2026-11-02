#!/usr/bin/env python3
"""Tests for the model table additions and SM120 warning."""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import planner  # noqa: E402


class TestModelTable(unittest.TestCase):
    def test_new_models_present(self):
        for name in ("qwen3.8-27b", "qwen3-235b-a22b",
                     "minimax-m3.1-flash", "deepseek-v4.1-flash"):
            self.assertIn(name, planner.MODEL_ARCH)
            self.assertGreater(planner.MODEL_ARCH[name]["kv_bytes_fp16"], 0)

    def test_list_models(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = planner.main(["--list-models"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("minimax-m3.1-flash", out)
        self.assertIn("qwen3.8-27b", out)

    def test_sm120_warning_in_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = planner.main([
                "--gpus", "RTXPRO5000-72", "--count", "8",
                "--model", "minimax-m3.1-flash", "--quant", "fp8",
                "--ctx", "131072", "--json",
            ])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertTrue(any("SM120" in w for w in data.get("warnings", [])))


if __name__ == "__main__":
    unittest.main()
