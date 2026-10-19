#!/usr/bin/env python3
"""Label-oracle diagnostics only. Never produces a submission or trains a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
PRIOR=HERE.parent/'preference_augmentation'/'outputs'


def metric(y,priority,ids):
    ix=np.flatnonzero(priority>0)
    ix=ix[np.lexsort((ids[ix],-priority[ix]))]
    if not len(ix):return 0.
    w=1/np.log2(np.arange(len(ix))+2)
    q=(w*(y[ix]-1)/4).sum()/w.sum()
    recall=(y[ix]>1).sum()/(y>1).sum()
    return 2*q*recall/(q+recall) if q+recall else 0.


def main():
    d=pd.read_csv(OUT/'oof.csv',dtype={'id':str});rows=[];sources={}
    for fold in range(5):
        path=PRIOR/f'outer{fold}_validation.npz';z=np.load(path)
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        a=d.iloc[z['pred']];p=z['both_d4']
        y=a.Label.to_numpy(int);ids=a.id.to_numpy(str)
        gain=p@(np.arange(5)/4);read=1-p[:,0];current=a.MC_full_priority.to_numpy()
        current_h=metric(y,current,ids)
        cutoff_oracle=max(metric(y,np.where(read>t,1+gain,0.),ids) for t in np.r_[-1,np.unique(read)])
        order=np.lexsort((ids,-gain));w=1/np.log2(np.arange(len(y))+2)
        q=np.cumsum(w*(y[order]-1)/4)/np.cumsum(w)
        recall=np.cumsum(y[order]>1)/(y>1).sum()
        h=np.divide(2*q*recall,q+recall,out=np.zeros(len(q)),where=q+recall>0)
        n=int(h.argmax())+1;mask=np.zeros(len(y),bool);mask[order[:n]]=True
        assert abs(metric(y,np.where(mask,1+gain,0.),ids)-h.max())<1e-12
        true_order=np.where(current>0,y.astype(float),0.)
        # Preserve selected label-1 papers with positive priority; do not accidentally exclude them.
        assert np.array_equal(true_order>0,current>0)
        rows.append(dict(outer_fold=fold,current=current_h,
            oracle_read_threshold=cutoff_oracle,oracle_gain_prefix=float(h.max()),
            oracle_true_order_same_selection=metric(y,true_order,ids)))
    frame=pd.DataFrame(rows)
    expected=pd.read_csv(OUT/'fold_metrics.csv').query("family=='MC_full'").sort_values('outer_fold').score
    np.testing.assert_allclose(frame.current,expected,atol=1e-12)
    frame.to_csv(OUT/'remaining_room.csv',index=False)
    info=dict(diagnostic_only=True,uses_outer_labels=True,deployable=False,
        mean_H=frame.drop(columns='outer_fold').mean().to_dict(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oof_sha256=hashlib.sha256((OUT/'oof.csv').read_bytes()).hexdigest(),probability_cache_sha256=sources,
        limitation='Conditional oracle results on reused folds; not a ceiling for all models or decision policies.')
    (OUT/'remaining_room.json').write_text(json.dumps(info,indent=2)+'\n')
    print(frame.to_string(index=False));print(json.dumps(info['mean_H'],indent=2))


if __name__=='__main__':main()
