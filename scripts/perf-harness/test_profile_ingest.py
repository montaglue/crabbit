#!/usr/bin/env python3
"""Unit tests for profile_ingest.py (docs/PROFILE-FEEDBACK-PLAN.md gates).

Run: python3 scripts/perf-harness/test_profile_ingest.py
Also wired into `cargo test -p backend-tests` (tests/profile_ingest.rs).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_ingest  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
BLOCKMAP = os.path.join(FIXTURES, "synthetic.blockmap.json")
PERF_SCRIPT = os.path.join(FIXTURES, "synthetic.perf-script.txt")
BLOCKMAP_OPS = os.path.join(FIXTURES, "synthetic-ops.blockmap.json")


def run_ingest(argv):
    return profile_ingest.main(argv)


class IngestFixtureTest(unittest.TestCase):
    def ingest(self, perf_scripts, blockmaps=None, want_op_costs=False,
               extra_argv=None):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "profile.json")
            argv = []
            for blockmap in blockmaps or [BLOCKMAP]:
                argv += ["--blockmap", blockmap]
            for script in perf_scripts:
                argv += ["--perf-script", script]
            argv += ["-o", out]
            argv += extra_argv or []
            self.assertEqual(run_ingest(argv), 0)
            with open(out) as handle:
                profile = json.load(handle)
            op_costs_path = os.path.join(tmp, "profile.op_costs.json")
            if want_op_costs:
                self.assertTrue(
                    os.path.exists(op_costs_path),
                    "op-attributed blockmap must produce op_costs sidecar",
                )
                with open(op_costs_path) as handle:
                    return profile, json.load(handle)
            self.assertFalse(
                os.path.exists(op_costs_path),
                "op-less blockmap must not produce an op_costs sidecar",
            )
            return profile

    def test_synthetic_fixture_normalizes_like_spectral(self):
        profile = self.ingest([PERF_SCRIPT])
        # hot_loop: entry (id 0) 2 samples, id 1 gets 1, id 2 gets 8; the
        # id -1 range, the out-of-range offset, the unknown symbols are
        # dropped. Frequencies are indexed by RA order (not layout order).
        # Smoothed (alpha=1): (n+1)/(entry+1) with entry=2,1,8 samples.
        self.assertEqual(profile["hot_loop"], [1.0, 2 / 3, 3.0])
        # _helper matches helper via underscore stripping; entry-only.
        self.assertEqual(profile["helper"], [1.0])
        # some_other_fn is not in the blockmap: absent, not an error.
        self.assertEqual(sorted(profile), ["helper", "hot_loop"])

    def test_multiple_runs_merge_additively(self):
        with tempfile.TemporaryDirectory() as tmp:
            second = os.path.join(tmp, "run2.txt")
            with open(second, "w") as handle:
                # A second run that only hits the loop body (id 2, range
                # 16..40) shifts the ratio: entry stays 2, id 2 goes 8 -> 12.
                handle.write("\t1000 hot_loop+0x10\n" * 4)
            profile = self.ingest([PERF_SCRIPT, second])
        self.assertEqual(profile["hot_loop"], [1.0, 2 / 3, 13 / 3])

    def test_entry_is_forced_to_at_least_one_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "run.txt")
            with open(script, "w") as handle:
                # Samples only in the loop body: the unsampled entry is
                # forced to one sample; smoothing floors the unsampled
                # blocks at the detection threshold instead of 0.0 (a
                # literal zero marks uses there as free to spill), and the
                # vector is rescaled so freq[entry] == 1.0 exactly — the
                # contract shared with the spectral model (see normalize()).
                handle.write("\t1000 hot_loop+0x10\n" * 5)
            profile = self.ingest([script])
        self.assertEqual(profile["hot_loop"], [1.0, 1.0, 6.0])

    def test_smooth_zero_restores_raw_counts(self):
        # `--smooth 0` is the documented A/B switch back to unsmoothed
        # frequencies (entry forced to >= 1 sample, unsampled blocks 0.0).
        profile = self.ingest([PERF_SCRIPT], extra_argv=["--smooth", "0"])
        self.assertEqual(profile["hot_loop"], [1.0, 0.5, 4.0])
        self.assertEqual(profile["helper"], [1.0])

    def test_root_names_cover_rust_roots(self):
        # Keep ROOT_NAMES in sync with `pub mod roots` in
        # crates/pliron-ll/src/passes/aarch64/opmap.rs (ISEL_ABI..MIDEND).
        self.assertEqual(
            profile_ingest.ROOT_NAMES,
            {
                -1: "isel:abi",
                -2: "regalloc",
                -3: "frame",
                -4: "placement",
                -5: "unattributed",
                -6: "midend",
            },
        )
        for root in range(-6, 0):
            self.assertNotIn("root:", profile_ingest.lifted_key(root))

    def test_unsampled_symbols_are_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "run.txt")
            with open(script, "w") as handle:
                handle.write("\t1000 helper+0x0\n")
            profile = self.ingest([script])
        self.assertNotIn("hot_loop", profile)
        self.assertEqual(profile["helper"], [1.0])

    def test_multiple_blockmaps_and_duplicate_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = os.path.join(tmp, "other.blockmap.json")
            with open(other, "w") as handle:
                json.dump(
                    {
                        # Duplicate of the fixture's hot_loop: first one wins.
                        "hot_loop": [{"id": 0, "start": 0, "end": 8}],
                        "second_obj_fn": [
                            {"id": 0, "start": 0, "end": 8},
                            {"id": 1, "start": 8, "end": 24},
                        ],
                    },
                    handle,
                )
            script = os.path.join(tmp, "run.txt")
            with open(script, "w") as handle:
                handle.write("\t1000 second_obj_fn+0x8\n")
                handle.write("\t1000 second_obj_fn+0x10\n")
                handle.write("\t1000 hot_loop+0x10\n")
            profile = self.ingest([script], blockmaps=[BLOCKMAP, other])
        self.assertEqual(profile["second_obj_fn"], [1.0, 3.0])
        # hot_loop kept the fixture's 3-block shape (offset 0x10 = id 2).
        self.assertEqual(profile["hot_loop"], [1.0, 1.0, 2.0])

    def test_demangled_symbols_with_spaces_still_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            blockmap = os.path.join(tmp, "b.blockmap.json")
            with open(blockmap, "w") as handle:
                json.dump(
                    {"spaced fn<like this>": [{"id": 0, "start": 0, "end": 32}]},
                    handle,
                )
            script = os.path.join(tmp, "run.txt")
            with open(script, "w") as handle:
                handle.write("\t1000 spaced fn<like this>+0x4\n")
            profile = self.ingest([script], blockmaps=[blockmap])
        self.assertEqual(profile["spaced fn<like this>"], [1.0])


class HelperTest(unittest.TestCase):
    def test_match_symbol_underscore_both_directions(self):
        blockmaps = {"plain": {}, "_prefixed": {}}
        self.assertEqual(profile_ingest.match_symbol(blockmaps, "plain"), "plain")
        self.assertEqual(profile_ingest.match_symbol(blockmaps, "_plain"), "plain")
        self.assertEqual(
            profile_ingest.match_symbol(blockmaps, "prefixed"), "_prefixed"
        )
        self.assertIsNone(profile_ingest.match_symbol(blockmaps, "absent"))

    def test_resolve_ip(self):
        table = [(0x1000, "a"), (0x1040, "b")]
        self.assertEqual(profile_ingest.resolve_ip(table, 0x1004), ("a", 4))
        self.assertEqual(profile_ingest.resolve_ip(table, 0x1040), ("b", 0))
        self.assertIsNone(profile_ingest.resolve_ip(table, 0xFFF))




class OpCostTest(IngestFixtureTest):
    def test_op_costs_machine_and_lifted(self):
        profile, op_costs = self.ingest(
            [PERF_SCRIPT], blockmaps=[BLOCKMAP_OPS], want_op_costs=True
        )
        # Block-level output is unchanged by the op annotations.
        self.assertEqual(profile["hot_loop"], [1.0, 2 / 3, 3.0])
        hot = op_costs["hot_loop"]
        # Samples land per instruction range (see the perf-script fixture:
        # offsets 0x0,0x8 -> ops 0,1; 0x10..0x1c + more in block id 2 ->
        # ops 2,3,4; block id 1 -> op 5; the id -1 block's op 6 IS
        # op-attributed even though its block samples are dropped).
        machine = hot["machine"]
        self.assertEqual(sum(machine.values()), 12)
        self.assertIn("2", machine)
        # Lifted: op ids aggregate (ops 2+3 both derive from llvm id 3);
        # negative roots render by name.
        lifted = hot["lifted"]
        self.assertEqual(
            lifted["3"], machine.get("2", 0) + machine.get("3", 0)
        )
        self.assertIn("isel:abi", lifted)
        self.assertIn("frame", lifted)
        self.assertIn("placement", lifted)
        self.assertEqual(sum(lifted.values()), sum(machine.values()))
        # helper has no ops array -> block-level only, absent here.
        self.assertNotIn("helper", op_costs)


if __name__ == "__main__":  # pragma: no cover - re-run friendly
    unittest.main()
