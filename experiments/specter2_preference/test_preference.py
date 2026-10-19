import unittest

import numpy as np
import pandas as pd
import torch

import modeling as m
import run_preference as run


class PreferenceTests(unittest.TestCase):
    def test_coherent_read_probability_and_gradients(self):
        logits=torch.tensor([[1.,2.,3.,4.,5.],[5.,4.,3.,2.,1.]],requires_grad=True)
        read_logit=torch.logsumexp(logits[:,1:],dim=1)-logits[:,0]
        torch.testing.assert_close(read_logit.sigmoid(),1-logits.softmax(1)[:,0])
        value=m.loss(logits,torch.tensor([5,1]));value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_metric_string_tie_break_and_empty_selection(self):
        y=np.array([1,5]);ids=np.array(['2','10'])
        actual=run.official(y,np.ones(2),ids)
        expected_q=1/(1+1/np.log2(3))
        self.assertAlmostEqual(actual['relevance'],expected_q)
        self.assertEqual(actual['FP'],1)
        self.assertEqual(run.official(y,np.zeros(2),ids)['score'],0.)
        p=np.array([[.5,.5,0.,0.,0.],[0.,0.,0.,0.,1.]])
        np.testing.assert_array_equal(m.priority(p,.5),[0.,10.])

    def test_choice_is_independent_of_outer_labels(self):
        fit=np.arange(8);nums=np.repeat(np.arange(4),2)
        y=np.array([1,5]*5,dtype=float)
        ctx=dict(y=y,data=pd.DataFrame({'id':list(map(str,range(10)))}))
        perfect=np.tile(np.array([[1,0,0,0,0],[0,0,0,0,1.]]),(4,1))
        uniform=np.full((8,5),.2)
        values={'frozen':{1:uniform,2:perfect},'adapted':{1:uniform,2:perfect},'incumbent':{0:uniform}}
        before=run.choose(ctx,fit,nums,values)
        y[8:]=[5,1]
        after=run.choose(ctx,fit,nums,values)
        self.assertEqual(before,after)
        self.assertEqual(before[0]['frozen']['epoch'],2)
        self.assertEqual(before[0]['selection_procedure']['family'],'frozen')


if __name__=='__main__':unittest.main()
