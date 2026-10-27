"""CPU-only input, policy, numerical, and evaluation-boundary contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run as pilot


class PilotContracts(unittest.TestCase):
    def setUp(self):
        self.cfg, self.base = pilot.load_config()

    def test_committed_inputs_align_without_parsing_validation_targets(self):
        docs, parts, baselines, manifest = pilot.load_unlabeled_inputs()
        self.assertEqual(len(docs), 2494)
        self.assertEqual(manifest["documents"], 2494)
        for name in pilot.PARTITIONS:
            self.assertEqual(baselines[name]["fit"], parts[name]["fit"])
            self.assertEqual(baselines[name]["pred"], parts[name]["pred"])
            self.assertEqual(len(baselines[name]["probabilities"]), len(parts[name]["pred"]))

    def test_target_reader_parses_only_requested_ids(self):
        targets = pilot.read_targets_only([0, 1])
        self.assertEqual(set(targets), {0, 1})
        self.assertTrue(set(targets.values()) <= {1, 2, 3, 4, 5})

    def test_baseline_priority_and_matched_budget(self):
        p = np.asarray([[.8, .1, .05, .03, .02], [.1, .1, .2, .3, .3],
                        [.4, .1, .2, .2, .1]])
        incumbent = p.copy()
        selected = (1 - incumbent[:, 0]) > .55
        priority = pilot.selected_priority(selected, pilot.gain(incumbent))
        self.assertEqual(np.flatnonzero(priority > 0).tolist(), [1, 2])
        budget = pilot.matched_budget_mask(1 - p[:, 0], np.asarray(["c", "b", "a"]), 2)
        self.assertEqual(int(budget.sum()), 2)
        self.assertEqual(pilot.official(np.asarray([1, 5, 4]), priority, np.asarray(["a", "b", "c"]))["selected"], 2)

    def test_five_class_ce_normalizes_small_float32_row_error(self):
        p = np.asarray([[.7, .1, .1, .06, .0400001],
                        [.1, .2, .2, .2, .3000001]], dtype=np.float32)
        result = pilot.probability_metrics(np.asarray([1, 5]), p)
        self.assertTrue(np.isfinite(result["five_class_logloss"]))
        self.assertTrue(np.isfinite(result["read_auc"]))
        self.assertGreaterEqual(result["read_brier"], 0)
        self.assertLess(result["probability_row_sum_max_error"], 1e-5)

    def test_confusion_has_all_five_labels_and_class_metrics(self):
        y = np.asarray([1, 2, 3, 4, 5])
        p = np.eye(5, dtype=float) * .8 + .04
        matrix, classes = pilot.confusion_and_classes(y, p)
        self.assertEqual(np.asarray(matrix).shape, (5, 5))
        self.assertEqual([r["label"] for r in classes], [1, 2, 3, 4, 5])
        self.assertTrue(all(r["precision"] == r["recall"] == 1.0 for r in classes))

    def test_evaluator_checks_both_partitions_before_reading_any_prediction_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = pilot.runtime_snapshot()
            pilot.write_json(root / "runtime_lock.json", {
                "runtime": snapshot, "config_sha256": pilot.sha(pilot.HERE / "config.json")})
            with patch.object(pilot, "read_targets_only", side_effect=AssertionError("prediction labels read too early")):
                with self.assertRaises(FileNotFoundError):
                    pilot.evaluate(root)

    def test_fit_only_membership_contains_no_prediction_ids(self):
        docs, parts, _, _ = pilot.load_unlabeled_inputs()
        for name in pilot.PARTITIONS:
            members, fit_labels = pilot.build_members(parts[name], docs, self.cfg)
            self.assertEqual(set(members["train"]), set(parts[name]["fit"]))
            self.assertEqual(set(members["validation"]), set(parts[name]["pred"]))
            self.assertTrue(set(members["train_probes"]) <= set(members["train"]))
            self.assertEqual(set(fit_labels), set(members["train"]))
            self.assertTrue(set(fit_labels.values()) == {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main(verbosity=2)
