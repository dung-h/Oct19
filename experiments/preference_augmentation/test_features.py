"""Meaningful invariance and leakage checks, including deliberately changed targets."""
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import features as f


class FeatureTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17);n=60
        self.data=pd.DataFrame(dict(id=[str(i) for i in range(n)],split=['train']*n,
            title=[f'Algorithm {i} for '+('medical diagnosis' if i%2 else 'logic programming') for i in range(n)],
            abstract=['We prove a theorem and evaluate an algorithm.' if i%3 else '' for i in range(n)],
            venue=['a' if i%2 else 'b' for i in range(n)],year=[2020.]*n))
        self.y=np.tile(np.arange(1,6),12);self.groups=np.arange(n)//2
        self.title=normalize(rng.normal(size=(n,8)));self.ret=normalize(rng.normal(size=(n,8)))

    def test_evidence_offsets_and_no_label_dependency(self):
        a,e=f.content_features(self.data)
        other=self.data.assign(Label=999,venue='arbitrary',year=-1)
        b,ee=f.content_features(other)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(e,ee)
        for r in e.itertuples():
            self.assertEqual(self.data.iloc[int(r.id)][r.field][r.start:r.end],r.evidence)
        self.assertEqual(a.abstract_type_unknown.iloc[0],1)

    def test_query_labels_cannot_change_history(self):
        refs=np.arange(40);query=np.arange(40,60)
        a=f.history(self.data,self.y,self.groups,self.title,self.ret,refs,query)
        y=self.y.copy();y[query]=999
        b=f.history(self.data,y,self.groups,self.title,self.ret,refs,query)
        pd.testing.assert_frame_equal(a,b)

    def test_duplicate_group_excluded(self):
        refs=np.arange(39);query=np.array([39])
        a=f.history(self.data,self.y,self.groups,self.title,self.ret,refs,query)
        y=self.y.copy();y[38]=5 if y[38]!=5 else 1
        b=f.history(self.data,y,self.groups,self.title,self.ret,refs,query)
        pd.testing.assert_frame_equal(a,b)

    def test_crossfit_lineage_and_heldout_target_isolation(self):
        a,lineage=f.crossfit_history(self.data,self.y,self.groups,self.title,self.ret,np.arange(60))
        seen=[]
        for row in lineage:
            self.assertTrue(set(self.groups[row['refs']]).isdisjoint(self.groups[row['queries']]))
            seen+=row['queries']
        self.assertEqual(sorted(seen),list(range(60)))
        row=lineage[0];y=self.y.copy();y[row['queries']]=1
        b=f.history(self.data,y,self.groups,self.title,self.ret,row['refs'],row['queries'])
        pd.testing.assert_frame_equal(a.iloc[row['queries']].reset_index(drop=True),b,check_dtype=False)

    def test_official_metric_text_id_ties(self):
        from verify import metric
        result=metric(np.array([1,5]),np.array([1.,1.]),np.array(['2','10']))
        q=1/(1+1/np.log2(3))
        self.assertAlmostEqual(result['relevance'],q)
        self.assertAlmostEqual(result['score'],2*q/(q+1))
        self.assertEqual(result['coverage'],1)
        self.assertEqual(metric(np.array([5]),np.array([0.]),np.array(['0']))['score'],0)

    def test_selection_and_ranking_use_different_quantities(self):
        from run_augmentation import priorities
        p=np.array([[.6,0,0,0,.4],[.3,.7,0,0,0]])
        result=priorities({'family':'both'},p,.5)
        self.assertEqual(result[0],0)
        self.assertAlmostEqual(result[1],1.75)


if __name__=='__main__':unittest.main()
