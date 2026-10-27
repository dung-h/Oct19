#!/usr/bin/env python3
"""CPU contract checks using synthetic prediction artifacts, never GPU evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from common import HERE, digest, fingerprint, load_inputs, write_json


class EvaluationContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        docs, _, parts, base = load_inputs()
        for tag, part in parts.items():
            out = self.root / tag
            out.mkdir()
            lock = dict(fingerprint=fingerprint(), partition=tag, mode="train")
            write_json(out / "runtime_lock.json", lock)
            write_json(out / "COMPLETE.json", dict(status="PREDICTIONS_COMPLETE", lock_sha=digest(lock)))
            rows = [dict(index=i, id=docs[i]["id"], lock_sha=digest(lock), probabilities=p)
                    for i, p in zip(part["pred"], base[tag]["probabilities"])]
            for arm in ("frozen", "trained"):
                (out / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self):
        return subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--output", str(self.root)],
                              capture_output=True, text=True)

    def test_identity_predictions_reproduce_baseline_without_false_gain(self):
        result = self.evaluate()
        self.assertEqual(result.returncode, 0, result.stderr)
        decision = json.loads((self.root / "decision.json").read_text())
        self.assertEqual(decision["mean_H_delta"], 0)
        self.assertEqual(decision["matched_budget_FP_saved"], 0)
        self.assertFalse(decision["continue_confirmation"])
        self.assertFalse(decision["learned_signal"])
        self.assertTrue((self.root / "confusion.csv").exists())
        self.assertTrue((self.root / "label_transitions.csv").exists())

    def test_missing_second_partition_predictions_fail_before_evaluation(self):
        (self.root / "outer1_inner1/trained.jsonl").unlink()
        self.assertNotEqual(self.evaluate().returncode, 0)
        self.assertFalse((self.root / "summary.csv").exists())

    def test_foreign_prediction_lineage_is_rejected(self):
        path = self.root / "outer1_inner0/trained.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["lock_sha"] = "foreign"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertNotEqual(self.evaluate().returncode, 0)
        self.assertFalse((self.root / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
