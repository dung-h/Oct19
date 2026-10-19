import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ranking_lib as r


class MetricTests(unittest.TestCase):
    def test_empty_selection(self):
        score = r.official(np.array([1, 5]), np.zeros(2), np.array(['1', '2']))
        self.assertEqual(score['score'], 0.0)
        self.assertEqual(score['coverage'], 0.0)

    def test_text_id_breaks_priority_tie(self):
        score = r.official(np.array([5, 1]), np.array([1., 1.]), np.array(['10', '2']))
        self.assertEqual(score['relevance'], 1 / (1 + 1 / np.log2(3)))

    def test_fixed_gate_preserves_selection(self):
        incumbent = np.array([[.4, .6, 0, 0, 0], [.8, .2, 0, 0, 0]], dtype=float)
        title = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], dtype=float)
        a = r.priority(incumbent, title, .5, 0.) > 0
        b = r.priority(incumbent, title, .5, 1.) > 0
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
