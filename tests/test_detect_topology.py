#!/usr/bin/env python3
"""Tests for tools/detect_topology.py parsing (no NVIDIA hardware needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import detect_topology as dt  # noqa: E402

TOPO = """\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNODE\tSYS\tSYS\t0-95\t0
GPU1\tNODE\t X \tSYS\tSYS\t0-95\t0
GPU2\tSYS\tSYS\t X \tNODE\t96-191\t1
GPU3\tSYS\tSYS\tNODE\t X \t96-191\t1
"""

GPUS = """0, NVIDIA RTX PRO 5000 72GB, 73415, 300.00, 300.00
1, NVIDIA RTX PRO 5000 72GB, 73415, 300.00, 300.00
"""

NUMA = """available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3
node 0 size: 773000 MB
node 1 cpus: 4 5 6 7
node 1 size: 773000 MB
"""


class TestTopology(unittest.TestCase):
    def test_parse_gpus(self):
        gpus = dt.parse_gpus(GPUS)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["name"], "NVIDIA RTX PRO 5000 72GB")
        self.assertEqual(gpus[0]["power_limit_w"], 300.0)

    def test_parse_groups(self):
        topo = dt.parse_topo(TOPO)
        self.assertEqual(topo["gpu_count"], 4)
        self.assertEqual(topo["groups"], [[0, 1], [2, 3]])
        self.assertTrue(topo["cross_group"])
        self.assertFalse(topo["has_nvlink"])
        self.assertEqual(topo["links"]["0-2"], "SYS")
        self.assertEqual(topo["links"]["2-3"], "NODE")

    def test_numa(self):
        numa = dt.parse_numa(NUMA)
        self.assertEqual(len(numa), 2)
        self.assertEqual(numa[0]["size_mb"], 773000)

    def test_summary_and_warnings(self):
        data = {
            "gpus": dt.parse_gpus(GPUS),
            "topology": dt.parse_topo(TOPO),
            "numa": dt.parse_numa(NUMA),
        }
        s = dt.summarize(data["gpus"], data["topology"], data["numa"])
        self.assertEqual(s["gpu_count"], 2)
        self.assertEqual(s["interconnect"], "pcie-cross-group")
        w = dt.warnings(data["gpus"], data["topology"])
        self.assertTrue(any("TP inside one group" in x for x in w))
        self.assertTrue(any("No NVLink" in x for x in w))


if __name__ == "__main__":
    unittest.main()
