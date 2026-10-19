import unittest
import numpy as np
import torch
import pandas as pd

from run_adapter import hard_candidates,preference_loss,make_priority,choose


class AdapterChecks(unittest.TestCase):
    def test_single_calibration_group_does_not_use_outer_labels(self):
        data=pd.DataFrame({'id':[str(i) for i in range(6)]})
        labels=np.array([1.,5.,1.,5.,1.,5.]);fit=np.arange(4);nums=np.zeros(4,dtype=int)
        values={'frozen':np.array([[.1,.02],[.9,.8],[.1,.02],[.9,.8]])}
        first=choose(data,labels,fit,nums,values)
        labels[4:]=[5.,1.]
        second=choose(data,labels,fit,nums,values)
        self.assertEqual(first,second)
        # Official usefulness comes from the true label5, irrespective of predicted gain0.8.
        self.assertAlmostEqual(first['frozen']['inner_score'],1.)

    def test_mining_uses_only_training_negatives_and_prefers_same_venue(self):
        y=np.array([5.,1.,1.,1.,5.])
        vectors=np.array([[1.,0.],[.7,.7],[1.,0.],[.99,.01],[1.,0.]])
        venues=np.array(['cav','cav','kr','cav','cav'])
        choices=hard_candidates(np.array([0,1,2]),y,vectors,venues)
        np.testing.assert_array_equal(choices[0],[1])
        self.assertNotIn(3,choices[0]);self.assertNotIn(4,choices)

    def test_correct_pair_order_has_lower_loss(self):
        labels=torch.tensor([5.,1.])
        correct=torch.tensor([[2.,2.],[-2.,2.]],requires_grad=True)
        wrong=torch.tensor([[-2.,2.],[2.,2.]],requires_grad=True)
        self.assertLess(float(preference_loss(correct,labels)),float(preference_loss(wrong,labels)))
        loss=preference_loss(correct,labels);loss.backward()
        self.assertTrue(torch.isfinite(correct.grad).all())

    def test_selection_and_gain_ranking_are_separate(self):
        values=np.array([[.9,.2],[.6,.5],[.1,.09]])
        p=make_priority(values,.5)
        np.testing.assert_array_equal(p,[2.,5.,0.])


if __name__=='__main__':unittest.main()
