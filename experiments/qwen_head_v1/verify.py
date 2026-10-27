"""CPU contracts for the head fit and experiment selection boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("qwen_head_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HeadContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(422)
        self.docs = {}
        self.targets = {}
        self.indices = list(range(180))
        hidden = []
        for i in self.indices:
            group, label = i // 5, i % 5 + 1
            self.docs[i] = {"group": f"g{group}"}
            self.targets[i] = label
            row = rng.normal(0, .1, 16)
            row[label - 1] += 2.0
            hidden.append(row)
        self.features = {"indices": np.asarray(self.indices), "hidden": np.asarray(hidden, dtype=np.float32)}
        self.train = list(range(120))
        self.validation = list(range(120, 180))
        self.cfg = {"cv_folds": 3, "lambda_grid": [.01, .1], "max_iterations": 250}

    def tearDown(self):
        self.temp.cleanup()

    def test_regularized_head_is_finite_and_keeps_five_class_order(self):
        y = np.asarray([self.targets[i] for i in self.train])
        x = self.features["hidden"][:120]
        scaler = StandardScaler().fit(x)
        w, b, opt = runner.fit_affine_head(scaler.transform(x), y, .1, 250)
        p = runner.probabilities(self.features["hidden"][120:], scaler, w, b)
        self.assertTrue(opt["success"])
        self.assertEqual(w.shape, (5, 16))
        self.assertEqual(b.shape, (5,))
        self.assertEqual(p.shape, (60, 5))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
        self.assertGreater(float(np.mean(p.argmax(1) + 1 == np.asarray([self.targets[i] for i in self.validation]))), .95)

    def test_penalty_selection_uses_training_groups_only(self):
        selected1, oof1, grid1, folds1 = runner.cv_select(
            self.features, self.targets, self.docs, self.train, self.cfg)
        changed = dict(self.targets)
        for i in self.validation:
            changed[i] = changed[i] % 5 + 1
        selected2, oof2, grid2, folds2 = runner.cv_select(
            self.features, changed, self.docs, self.train, self.cfg)
        self.assertEqual(selected1, selected2)
        self.assertEqual(grid1, grid2)
        self.assertEqual(folds1, folds2)
        np.testing.assert_array_equal(oof1, oof2)
        self.assertEqual(oof1.shape, (len(self.train), 5))

    def test_cv_never_splits_publication_group(self):
        selected, oof, grid, folds = runner.cv_select(
            self.features, self.targets, self.docs, self.train, self.cfg)
        self.assertEqual(len(folds), 3 * len(self.cfg["lambda_grid"]))
        self.assertEqual(len(grid), len(self.cfg["lambda_grid"]))
        self.assertTrue(np.isfinite(oof).all())
        self.assertIn(selected, self.cfg["lambda_grid"])

    def test_policy_priority_uses_nonzero_only_for_selected_rows(self):
        p = np.asarray([[.8, .1, .05, .03, .02], [.1, .1, .2, .3, .3]])
        priority = runner.native_priority(p, .55)
        self.assertEqual(priority[0], 0)
        self.assertGreater(priority[1], 0)
        y = np.asarray([1, 5])
        result = runner.official(y, priority, np.asarray(["1", "2"]))
        self.assertEqual((result["TP"], result["FP"], result["FN"], result["TN"]), (1, 0, 0, 1))

    def test_affine_head_expects_flat_final_hidden_and_emits_five_logits(self):
        state = {"weight": np.zeros((5, 16), dtype=np.float32),
                 "bias": np.zeros(5, dtype=np.float32),
                 "mean": np.zeros(16, dtype=np.float32),
                 "scale": np.ones(16, dtype=np.float32)}
        hidden = torch.zeros((3, 16), dtype=torch.float32)
        logits = runner.head_logits(hidden, {k: torch.as_tensor(v) for k, v in state.items()})
        self.assertEqual(tuple(logits.shape), (3, 5))

    def test_accumulated_penalty_equals_single_update_objective(self):
        ce = np.asarray([.2, .4, .6])
        penalty = .17
        accumulated = np.mean(ce + penalty)
        expected = np.mean(ce) + penalty
        self.assertAlmostEqual(accumulated, expected, places=14)

    def test_paired_group_bootstrap_is_zero_for_identical_arms(self):
        y = np.asarray([1, 5] * 10)
        p = np.tile(np.asarray([[.8, .1, .05, .03, .02], [.1, .1, .2, .3, .3]]), (10, 1))
        groups = [f"g{i}" for i in range(10) for _ in range(2)]
        ids = np.asarray([f"id{i}" for i in range(len(y))])
        result = runner.paired_group_bootstrap(y, p, p.copy(), groups, ids, .55, 12, 7)
        for metric in ("H", "rating_logloss", "read_AUC"):
            self.assertEqual(result[metric]["lower_2_5"], 0.0)
            self.assertEqual(result[metric]["upper_97_5"], 0.0)

    def test_evaluator_refuses_to_read_labels_before_training_completes(self):
        root = self.root / "incomplete"
        (root / "training").mkdir(parents=True)

        class NoRead(dict):
            def __getitem__(self, key):
                raise AssertionError("Evaluator read a target before checking job completion")

        with self.assertRaises(FileNotFoundError):
            runner.evaluate(root, {}, self.docs, NoRead(), {"validation": self.validation})
        self.assertFalse((root / "decision.json").exists())

    def test_evaluator_refuses_to_overwrite_completed_quality_result(self):
        root = self.root / "evaluated"
        root.mkdir()
        marker = root / "decision.json"
        marker.write_text('{"status":"already complete"}\n')
        with self.assertRaises(RuntimeError):
            runner.evaluate(root, {}, self.docs, {}, {})
        self.assertEqual(marker.read_text(), '{"status":"already complete"}\n')


if __name__ == "__main__":
    unittest.main(verbosity=2)
