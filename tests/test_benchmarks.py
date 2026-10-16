#!/usr/bin/env python3
"""Tests for the --benchmarks data loading in tools/planner.py.

The loader must be strict on purpose: an entry that does not carry the
expected schema tag must be rejected loudly, because a silently accepted
foreign file would put unvetted numbers next to the plan.

Run with:  python -m unittest discover tests
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import planner  # noqa: E402

VALID_ENTRY = {
    "schema": "lltg.benchmark.v1",
    "id": "test-entry",
    "kind": "external-reference",
    "title": "Replicas beat TP when the model fits",
    "hardware": {"gpus": "4 x RTX 3090", "interconnect": "PCIe Gen4"},
    "software": {"framework": "vLLM", "framework_version": "0.21.0"},
    "model": {"name": "test-35b-a3b", "quantization": "AWQ"},
    "workload": {"concurrency": [128]},
    "results": [
        {"config": "2x TP=2", "metrics": {"aggregate_toks": 1437}},
        {"config": "1x TP=4", "metrics": {"aggregate_toks": 610}},
    ],
    "variance": {"note": "single run as published"},
    "source": {"project": "test project", "url": None},
    "transferable": "replicas beat TP here",
    "not_transferable": "the numbers",
}


def make_dir(files):
    tmp = tempfile.TemporaryDirectory()
    for name, content in files.items():
        with open(os.path.join(tmp.name, name), "w", encoding="utf-8") as fh:
            fh.write(content)
    return tmp


class TestLoadBenchmarks(unittest.TestCase):
    def test_loads_valid_directory(self):
        with make_dir({"a.json": json.dumps(VALID_ENTRY)}) as tmp:
            entries = planner.load_benchmarks(tmp)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "test-entry")
        self.assertEqual(entries[0]["_file"], "a.json")

    def test_rejects_wrong_schema(self):
        bad = dict(VALID_ENTRY, schema="some-other-schema/v9")
        with make_dir({"bad.json": json.dumps(bad)}) as tmp:
            with self.assertRaises(SystemExit):
                planner.load_benchmarks(tmp)

    def test_rejects_invalid_json(self):
        with make_dir({"bad.json": "{not json"}) as tmp:
            with self.assertRaises(SystemExit):
                planner.load_benchmarks(tmp)

    def test_empty_directory_is_an_error(self):
        with make_dir({}) as tmp:
            with self.assertRaises(SystemExit):
                planner.load_benchmarks(tmp)

    def test_single_file_path_works(self):
        with make_dir({"one.json": json.dumps(VALID_ENTRY)}) as tmp:
            entries = planner.load_benchmarks(os.path.join(tmp, "one.json"))
        self.assertEqual(len(entries), 1)


class TestBenchmarksInPlan(unittest.TestCase):
    def test_plan_carries_loaded_entries(self):
        with make_dir({"a.json": json.dumps(VALID_ENTRY)}) as tmp:
            p = planner.plan(planner.build_parser().parse_args(
                ["--gpus", "RTX3090-24", "--count", "4", "--params", "35",
                 "--benchmarks", tmp]))
        self.assertEqual(len(p.benchmarks), 1)
        d = p.as_dict()
        self.assertEqual(d["measured_data"][0]["id"], "test-entry")

    def test_render_includes_title_and_transferable(self):
        with make_dir({"a.json": json.dumps(VALID_ENTRY)}) as tmp:
            entries = planner.load_benchmarks(tmp)
        text = planner.render_benchmarks(entries)
        self.assertIn("Replicas beat TP when the model fits", text)
        self.assertIn("replicas beat TP here", text)
        self.assertIn("external-reference", text)

    def test_list_benchmarks_prints_entries(self):
        with make_dir({"a.json": json.dumps(VALID_ENTRY)}) as tmp:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = planner.main(["--gpus", "RTX3090-24", "--count", "4",
                                   "--params", "35", "--benchmarks", tmp,
                                   "--list-benchmarks"])
        self.assertEqual(rc, 0)
        self.assertIn("test-entry", buf.getvalue())

    def test_list_benchmarks_without_path_fails(self):
        with self.assertRaises(SystemExit):
            planner.main(["--list-benchmarks"])


class TestRealDataFiles(unittest.TestCase):
    """The shipped benchmarks/data entries must load with the real loader."""

    def test_shipped_data_loads(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data")
        entries = planner.load_benchmarks(data_dir)
        ids = {e["id"] for e in entries}
        self.assertIn("rtx6000-maxq-mtp-vs-dflash", ids)
        self.assertIn("rtx3090-replicas-vs-tp", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
