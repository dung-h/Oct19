"""Metamorphic checks for the count representation intervention."""
import unittest

import numpy as np
import pandas as pd

import run_combined as r


class CountTransformation(unittest.TestCase):
    def sample(self):
        frame=pd.DataFrame({'venue':['a','b','c'],'ridge_rating':[1.,2.,3.]})
        for channel in ['retrieval','title','abstract']:
            for scope in ['all','venue']:
                prefix=f'neighbor_{channel}_{scope}'
                frame[prefix+'_references']=[100,200,0]
                frame[prefix+'_support']=[50,20,0]
                frame[prefix+'_class1_nearest']=[.1,.2,0]
        return frame

    def test_common_scale_invariance(self):
        source=self.sample();scaled=source.copy()
        for c in source:
            if c.endswith(('_references','_support')):scaled[c]*=1.5
        pd.testing.assert_frame_equal(r.transform(source,'ORD_count_fraction'),r.transform(scaled,'ORD_count_fraction'))

    def test_zero_reference_and_no_input_mutation(self):
        source=self.sample();snapshot=source.copy(deep=True)
        actual=r.transform(source,'ORD_count_fraction')
        pd.testing.assert_frame_equal(source,snapshot)
        assert np.isfinite(actual.select_dtypes('number').to_numpy()).all()
        assert (actual.filter(like='_support').iloc[2]==0).all()

    def test_unrelated_evidence_preserved(self):
        source=self.sample();actual=r.transform(source,'MC_count_fraction')
        keep=[c for c in source if not c.endswith(('_references','_support'))]
        pd.testing.assert_frame_equal(source[keep],actual[keep])
        pd.testing.assert_frame_equal(actual,r.transform(source,'ORD_count_fraction'))


if __name__=='__main__':unittest.main()
