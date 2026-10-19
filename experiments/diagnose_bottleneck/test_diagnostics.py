import unittest

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

from run_diagnostics import kernel_predictions, metadata_spaces, select_inner


class DiagnosticChecks(unittest.TestCase):
    def test_custom_kernel_solver_matches_sklearn(self):
        x = np.array([[0., 1.], [1., 0.], [2., 1.], [-1., 2.]])
        z = np.array([[.2, .3], [-1., 0.]])
        y = np.array([1., 2., 5., 3.])
        kf, kp = rbf_kernel(x, gamma=.7), rbf_kernel(z, x, gamma=.7)
        pred, _ = kernel_predictions(kf, kp, y, .1)
        model = KernelRidge(alpha=.1, kernel='precomputed').fit(kf, y-y.mean())
        np.testing.assert_allclose(pred, model.predict(kp)+y.mean(), atol=1e-10)
        shifted, _ = kernel_predictions(kf, kp, y+8, .1)
        np.testing.assert_allclose(shifted, pred+8, atol=1e-10)

    def test_metadata_vocab_and_year_are_fit_on_training_only(self):
        data = pd.DataFrame(dict(venue=['cav', 'cav', 'unseen'],
                                 authors=['Alice, Bob', 'Bob', 'Validation Only'],
                                 year=[2000., 2002., 2050.]))
        spaces = metadata_spaces(data, np.array([0,1]), np.array([2]))
        self.assertEqual(spaces['venue'][0].shape[1], 1)
        self.assertEqual(spaces['venue'][1].nnz, 0)
        self.assertEqual(spaces['authors'][0].shape[1], 2)
        self.assertEqual(spaces['authors'][1].nnz, 0)
        np.testing.assert_allclose(spaces['year'][0].toarray().ravel(), [-.25,.25])
        self.assertAlmostEqual(spaces['year'][1].toarray()[0,0], 12.25)

    def test_outer_validation_labels_do_not_affect_inner_selection(self):
        data = pd.DataFrame({'id':[str(i) for i in range(10)]})
        fit = np.arange(8)
        inner_folds = np.repeat(np.arange(4),2)
        labels = np.array([1.,5.]*5)
        configs = [dict(family='toy', key='c0'), dict(family='toy', key='c1')]
        predictions = {'c0':np.array([1.2,4.5]*4), 'c1':np.array([4.5,1.2]*4)}
        first, _ = select_inner(data, labels, fit, inner_folds, predictions, configs)
        labels[8:] = [5.,1.]
        second, _ = select_inner(data, labels, fit, inner_folds, predictions, configs)
        self.assertEqual(first, second)
        self.assertEqual(first['toy']['key'], 'c0')


if __name__ == '__main__':
    unittest.main()
