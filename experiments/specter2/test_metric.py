"""Small checks for the competition's selection, ranking and scoring contract."""
import unittest

import numpy as np

from run_experiment import official_metric, priorities


class CompetitionMetricTest(unittest.TestCase):
    def test_perfect_selection(self):
        self.assertEqual(official_metric([5, 1], [5, 1], ["a", "b"], 1)["score"], 1)

    def test_unselected_worthwhile_items_remain_in_recall_denominator(self):
        result = official_metric([5, 4, 1], [4, 1, 2], ["a", "b", "c"], 3)
        self.assertEqual(result["relevance"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertAlmostEqual(result["score"], 2 / 3)

    def test_id_tie_break_is_text_order(self):
        result = official_metric([5, 1], [3, 3], ["2", "10"], 1)
        self.assertAlmostEqual(result["relevance"], 0.6309297535714575 / 1.6309297535714575)

    def test_empty_and_no_worthwhile_items(self):
        self.assertEqual(official_metric([5], [1], ["1"], 1)["score"], 0)
        self.assertEqual(official_metric([1], [3], ["1"], 1)["score"], 0)

    def test_emitted_priorities_match_independent_evaluator(self):
        rng = np.random.default_rng(42)
        for threshold in [1, 1.5, 2, 3]:
            labels = rng.integers(1, 6, 100)
            predictions = rng.normal(3, 2, 100).round(1)
            ids = np.arange(100).astype(str)
            emitted = priorities(predictions, threshold)
            rows = sorted([(score, identity, label) for score, identity, label in zip(emitted, ids, labels)
                           if score > 0], key=lambda row: (-row[0], row[1]))
            weight = [1 / np.log2(rank + 2) for rank in range(len(rows))]
            q = sum(w * (row[2] - 1) / 4 for w, row in zip(weight, rows)) / sum(weight)
            r = sum(row[2] > 1 for row in rows) / sum(labels > 1)
            h = 2 * q * r / (q + r)
            self.assertAlmostEqual(official_metric(labels, predictions, ids, threshold)["score"], h)


if __name__ == "__main__":
    unittest.main()
