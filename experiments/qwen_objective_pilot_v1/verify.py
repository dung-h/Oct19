"""Verify fit-only screening artifacts and loss/probability contracts."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax

from screen import ARMS, PARTS, OUT, class_weights, rps


def main():
    y=np.array([1,2,3,4,5])
    perfect=np.eye(5)
    assert np.allclose(rps(perfect,y),0)
    near=np.full((5,5),.025); near[np.arange(5),np.arange(5)]=.9
    near/=near.sum(1,keepdims=True)
    assert rps(near,y).mean() < rps(np.full((5,5),.2),y).mean()
    assert np.allclose(class_weights(y),np.ones(5))
    assert np.isfinite(softmax(np.zeros((3,5)),axis=1)).all()
    reports={p:json.loads((OUT/f'{p}.json').read_text()) for p in PARTS}
    for part,report in reports.items():
        assert report['group_disjoint']
        assert report['n'] in (1494,1497)
        assert set(report['arms'])==set(ARMS)
        for arm,row in report['arms'].items():
            selected=row['selected']
            assert np.isfinite(selected['ce']) and np.isfinite(selected['rps'])
            assert len(selected['fold_ce'])==3
            assert row['selected_lambda'] in (.001,.01,.1,1.,10.)
    decision=json.loads((OUT/'decision.json').read_text())
    assert decision['status']=='SCREEN_COMPLETE'
    assert decision['run_gpu']==any(v!='ce' for v in decision['gpu_candidates'].values())
    print(json.dumps({'status':'VERIFIED','loss_contracts':True,'partitions':list(reports),
                      'run_gpu':decision['run_gpu']}))


if __name__=='__main__': main()
