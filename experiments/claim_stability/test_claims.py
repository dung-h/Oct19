import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_claims as run
import claim_features as cf


class ClaimTests(unittest.TestCase):
    def setUp(self):
        self.data=pd.DataFrame(dict(split=['train','test'],id=['1','2'],
            title=['A student writes software','A tutorial on verification'],
            abstract=['The author is a Ph.D. student. We propose a new symbolic algorithm. We show the algorithm is sound.',
                      'We study verification for education applications. We construct a new invariant.']))

    def test_extraction_label_independent_and_verbatim(self):
        a,e=cf.extract(self.data)
        b,ee=cf.extract(self.data.assign(Label=[999,-5],venue=['arbitrary','none'],year=[-1,9999]))
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(e,ee)
        self.assertIn('We propose a new symbolic algorithm.',a.claims_method.iloc[0])
        for row in e.itertuples():
            source=self.data.loc[self.data.id.eq(row.id),'abstract'].iloc[0]
            self.assertEqual(source[row.start:row.end],row.evidence)

    def test_missing_claims_do_not_fall_back_to_abstract(self):
        d=self.data.copy();d['abstract']=['A background sentence without contribution cues.','']
        a,e=cf.extract(d)
        for role in cf.ASPECTS:self.assertTrue(a['claims_'+role].eq('').all())
        self.assertFalse(e.kind.eq('claims').any())
        self.assertNotEqual(a.prefix_problem.iloc[0],'')

    def test_repairs_are_contextual_and_unknowns_update(self):
        content,_=run.previous.feat.content_features(self.data)
        repaired,_=cf.repair(self.data,content)
        self.assertEqual(content.abstract_domain_education.iloc[0],1)
        self.assertEqual(repaired.abstract_domain_education.iloc[0],0)
        self.assertEqual(repaired.abstract_domain_education.iloc[1],1)
        self.assertEqual(repaired.title_type_tutorial.iloc[1],1)
        self.assertEqual(repaired.title_type_unknown.iloc[1],0)

    def test_svd_fit_is_invariant_to_prediction_rows(self):
        rng=np.random.default_rng(18);v=rng.normal(size=(40,3,384)).astype(np.float32)
        v[39]=0;fit=np.arange(30);pred=np.arange(30,40)
        with threadpool_limits(2):
            a,b,models=cf.project(v,fit,pred)
            altered=v.copy();altered[30:39]=rng.normal(size=(9,3,384))*100
            aa,bb,other=cf.project(altered,fit,pred)
        pd.testing.assert_frame_equal(a,aa)
        for x,y in zip(models,other):np.testing.assert_allclose(x.components_,y.components_,atol=1e-12)
        self.assertTrue(b.iloc[-1].eq(0).all())
        self.assertFalse(np.allclose(b.iloc[:-1],bb.iloc[:-1]))

    def test_repairs_preserve_existing_feature_order(self):
        content,_=run.previous.feat.content_features(self.data)
        fixed,_=cf.repair(self.data,content)
        frame=content.add_prefix('content_');frame.insert(0,'ridge_rating',[2.,3.])
        repaired=run.repair_frame(frame,fixed,np.arange(2))
        self.assertEqual(repaired.columns[:len(frame.columns)].tolist(),frame.columns.tolist())


if __name__=='__main__':unittest.main()
