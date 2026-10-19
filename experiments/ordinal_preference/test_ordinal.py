import unittest
import numpy as np
from sklearn.isotonic import isotonic_regression

import ordinal_math as om


class OrdinalTests(unittest.TestCase):
    def test_ground_truth_targets_reconstruct_ratings(self):
        y=np.arange(1,6);q=om.targets(y)
        np.testing.assert_array_equal(q[:,0],y>=2)
        np.testing.assert_array_equal(om.distribution(q),np.eye(5))
        np.testing.assert_allclose(q.mean(1),(y-1)/4)

    def test_pav_matches_independent_isotonic_solver(self):
        raw=np.random.default_rng(42).uniform(size=(100,4))
        actual=om.project(raw)
        expected=np.stack([isotonic_regression(row,increasing=False) for row in raw])
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        np.testing.assert_allclose(actual.sum(1),raw.sum(1),atol=1e-14)
        np.testing.assert_allclose(om.project(actual),actual,atol=1e-14)

    def test_projection_and_gain_identity(self):
        raw=np.array([[.8,.9,.4,.5],[1.,.75,.5,.25],[0.,0.,0.,0.]])
        view=om.ordinal_view(raw)
        np.testing.assert_allclose(view['tails'][0],[.85,.85,.45,.45])
        np.testing.assert_allclose(view['gain'],view['probability']@(np.arange(5)/4))
        np.testing.assert_allclose(view['gain'],raw.mean(1))
        self.assertTrue((view['probability']>=0).all())

    def test_strict_threshold_and_invalid_targets(self):
        view=om.ordinal_view(np.array([[.5,.5,0.,0.],[1.,1.,1.,1.]]))
        np.testing.assert_array_equal(om.priorities(view,.5),[0.,10.])
        with self.assertRaises(AssertionError):om.targets([1,2,999])


if __name__=='__main__':unittest.main()
