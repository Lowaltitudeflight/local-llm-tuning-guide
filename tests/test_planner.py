#!/usr/bin/env python3
"""Unit tests for tools/planner.py.

Tests the three things the planner must never get wrong:

1. A model that fits on one GPU must be planned as replicas, never as TP.
2. TP must never exceed the ceiling the interconnect allows.
3. Unknown KV geometry must be reported as [open], never guessed.

Standard library only - run with:  python -m unittest discover tests
or:  python tests/test_planner.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import planner  # noqa: E402


def run(argv):
    return planner.plan(planner.build_parser().parse_args(argv))


class TestReplicaPreference(unittest.TestCase):
    """If it fits on one GPU, replicas beat tensor parallelism."""

    def test_small_model_on_many_gpus_becomes_replicas(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--model", "qwen2.5-7b",
                 "--quant", "fp16", "--ctx", "8192"])
        self.assertEqual(p.tp, 1)
        self.assertEqual(p.pp, 1)
        self.assertEqual(p.dp, 8)

    def test_replica_conclusion_is_graded_measured(self):
        p = run(["--gpus", "H800-80", "--count", "8", "--model", "qwen2.5-7b"])
        grades = [c.grade for c in p.conclusions]
        self.assertIn(planner.MEASURED, grades)


class TestInterconnectCeiling(unittest.TestCase):
    """TP is bounded by the link, not by the number of GPUs installed."""

    def test_consumer_cards_cap_tp_at_two(self):
        p = run(["--gpus", "RTX4090-24", "--count", "4", "--params", "70",
                 "--quant", "fp16"])
        self.assertLessEqual(p.tp, 2)

    def test_cross_switch_caps_tp_at_group_size(self):
        p = run(["--gpus", "L40S-48", "--count", "8", "--params", "70",
                 "--interconnect", "pcie-cross-switch", "--group", "2"])
        self.assertLessEqual(p.tp, 2)

    def test_nvlink_allows_wider_tp(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--params", "70",
                 "--quant", "fp16"])
        self.assertGreater(p.tp, 2)

    def test_tp_never_exceeds_gpu_count(self):
        for gpu in ("A800-80", "H800-80", "RTX4090-24", "L40S-48"):
            p = run(["--gpus", gpu, "--count", "4", "--params", "70"])
            self.assertLessEqual(p.tp * p.pp, 4)


class TestMemoryBudget(unittest.TestCase):
    """Weights and KV drive how far the model must be split."""

    def test_weights_scale_with_quantisation(self):
        fp16 = run(["--gpus", "A800-80", "--count", "8", "--params", "70", "--quant", "fp16"])
        awq = run(["--gpus", "A800-80", "--count", "8", "--params", "70", "--quant", "awq"])
        self.assertAlmostEqual(fp16.memory["weights_total_gb"], 140.0, places=1)
        self.assertLess(awq.memory["weights_total_gb"], fp16.memory["weights_total_gb"])

    def test_quantisation_can_remove_the_need_to_split(self):
        # Modest context and concurrency, so that KV does not dominate and the
        # effect of weight quantisation is visible on its own.
        base = ["--gpus", "A800-80", "--count", "8", "--params", "70",
                "--ctx", "4096", "--concurrency", "8"]
        heavy = run(base + ["--quant", "fp16"])
        light = run(base + ["--quant", "awq"])
        self.assertGreater(heavy.tp, 1)
        self.assertEqual(light.tp, 1)

    def test_kv_dominates_at_long_context(self):
        # At 8k context and 64 slots the KV cache alone is ~172 GB for a 70B
        # class model - larger than the weights. The planner must split for it.
        short = run(["--gpus", "A800-80", "--count", "8", "--params", "70",
                     "--quant", "awq", "--ctx", "4096", "--concurrency", "8"])
        long = run(["--gpus", "A800-80", "--count", "8", "--params", "70",
                    "--quant", "awq", "--ctx", "8192", "--concurrency", "64"])
        self.assertEqual(short.tp, 1)
        self.assertGreater(long.tp, 1)

    def test_longer_context_forces_fp8_kv(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--model", "qwen2.5-72b",
                 "--ctx", "131072"])
        self.assertEqual(p.kv_dtype, "fp8")

    def test_framework_overhead_is_reserved(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--model", "qwen2.5-7b", "--util", "0.90"])
        self.assertAlmostEqual(
            p.memory["usable_per_gpu_gb"], 80 * 0.90 - planner.FRAMEWORK_OVERHEAD_GB, places=2
        )


class TestNoInventedNumbers(unittest.TestCase):
    """The whole point of the project: an empty cell beats a fabricated one."""

    def test_unknown_model_leaves_kv_open(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--model", "totally-unknown-13b",
                 "--params", "13"])
        open_claims = [c for c in p.conclusions if c.grade == planner.OPEN]
        self.assertTrue(open_claims, "unknown KV geometry must be reported as [open]")
        self.assertTrue(any("KV" in c.claim for c in open_claims))

    def test_measured_kv_input_is_accepted(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--model", "totally-unknown-13b",
                 "--params", "13", "--kv-bytes-per-token", "120000"])
        self.assertIsNotNone(p.memory["kv_total_gb"])
        self.assertGreater(p.memory["kv_total_gb"], 0)

    def test_h20_interconnect_stays_open(self):
        p = run(["--gpus", "H20-96", "--count", "8", "--params", "70"])
        self.assertTrue(any("topo" in w or "interconnect" in w for w in p.warnings))


class TestLegalTensorParallelSizes(unittest.TestCase):
    """TP must divide both head counts, or the engine refuses to start."""

    def test_tp_divides_attention_heads(self):
        # Qwen2.5-72B: 64 attention heads, 8 KV heads -> TP in {1,2,4,8}.
        for gpu, count in (("A800-80", 8), ("H800-80", 8), ("RTX4090-24", 4)):
            p = run(["--gpus", gpu, "--count", str(count), "--model", "qwen2.5-72b"])
            self.assertEqual(64 % p.tp, 0, f"{gpu}: TP={p.tp} does not divide 64 heads")
            self.assertEqual(8 % p.tp, 0, f"{gpu}: TP={p.tp} does not divide 8 KV heads")

    def test_tp_stays_within_gpu_count_and_ceiling(self):
        p = run(["--gpus", "RTX4090-24", "--count", "4", "--model", "qwen2.5-72b"])
        self.assertLessEqual(p.tp, 4)

    def test_replicas_do_not_need_head_divisibility(self):
        # Replicas run a whole model per GPU, so head counts are irrelevant.
        p = run(["--gpus", "A800-80", "--count", "8", "--model", "qwen2.5-7b"])
        self.assertEqual(p.tp, 1)
        self.assertEqual(p.dp, 8)


class TestCommandRendering(unittest.TestCase):
    """The output has to be something you can paste into a shell."""

    def test_command_includes_parallel_flags(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--params", "70", "--quant", "fp16"])
        cmd = planner.render_command(p)
        self.assertIn("vllm serve", cmd)
        self.assertIn(f"--tensor-parallel-size {p.tp}", cmd)
        self.assertIn("--gpu-memory-utilization", cmd)

    def test_replica_layout_mentions_cuda_visible_devices(self):
        p = run(["--gpus", "A800-80", "--count", "8", "--model", "qwen2.5-7b"])
        self.assertIn("CUDA_VISIBLE_DEVICES", planner.render_command(p))


if __name__ == "__main__":
    unittest.main(verbosity=2)
