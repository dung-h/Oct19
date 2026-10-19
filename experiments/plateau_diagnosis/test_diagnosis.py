import itertools
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
import run_diagnosis as run
from audit_content import claim_spans


class DiagnosisTests(unittest.TestCase):
    def test_verbatim_spans_and_unknown(self):
        text='Background context. We study verification of neural controllers. We propose a new algorithm. We prove the soundness of this algorithm.'
        spans=claim_spans(text)
        self.assertEqual(set(s['aspect'] for s in spans),{'problem','method','result'})
        for span in spans:self.assertEqual(text[span['start']:span['end']],span['evidence'])
        self.assertEqual(claim_spans(''),[])
        self.assertEqual(claim_spans('No explicit scientific claim appears here.'),[])

    def test_fixed_cutoffs(self):
        p=np.array([[.49,.51,0,0,0],[.51,0,0,0,.49]])
        result=run.priorities('catboost',p)
        self.assertGreater(result[0],0)
        self.assertEqual(result[1],0)
        np.testing.assert_allclose(run.priorities('ridge',np.array([1.9,2.,2.1])),[0,0,2.75])

    def test_subset_selection_cannot_use_outer_validation_labels(self):
        with threadpool_limits(2):
            ctx=run.previous.context()
            specs=list(itertools.islice(run.subsets(ctx),7))
            mutated=list(ctx);mutated[2]=ctx[2].copy();mutated[2][specs[0]['val']]=999
            after=list(itertools.islice(run.subsets(tuple(mutated)),7))
            for a,b in zip(specs,after):
                np.testing.assert_array_equal(a['fit'],b['fit'])
                np.testing.assert_array_equal(a['val'],b['val'])


if __name__=='__main__':unittest.main()
