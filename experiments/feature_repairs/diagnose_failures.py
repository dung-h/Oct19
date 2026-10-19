#!/usr/bin/env python3
"""Post-hoc gate/ranker accounting, diagnostic only; never selects a candidate."""
import json

import numpy as np
import pandas as pd

import run_repairs as r
from verify_repairs import metric


def diagnose():
    assert json.loads((r.OUT/'verification.json').read_text())['status']=='PASSED'
    ctx=r.context();rows=[]
    for fold,_,pred,_ in r.prior.outer_members(ctx['data']):
        choices=json.loads((r.OUT/f'outer{fold}_choices.json').read_text())
        z=np.load(r.OUT/f'outer{fold}_validation_predictions.npz')
        ids=ctx['data'].id.iloc[pred].to_numpy();y=ctx['y'][pred]
        base=choices['reference'];bp=z['reference']
        def crossed(gate,threshold,rank):
            p=np.where(1-gate[:,0]>threshold,10*(rank@(np.arange(5)/4)),0.)
            return metric(y,p,ids)['score']
        hbb=crossed(bp,base['threshold'],bp)
        for policy in r.ARMS[1:]+['selection_procedure']:
            choice=choices[policy];cp=z[choice['arm']]
            haa=crossed(cp,choice['threshold'],cp)
            hab=crossed(cp,choice['threshold'],bp)
            hba=crossed(bp,base['threshold'],cp)
            gate=.5*((hab-hbb)+(haa-hba))
            rank=.5*((hba-hbb)+(haa-hab))
            assert abs(gate+rank-(haa-hbb))<1e-12
            rows.append(dict(outer_fold=fold,policy=policy,selected_arm=choice['arm'],
                base_H=hbb,new_H=haa,new_gate_old_rank_H=hab,old_gate_new_rank_H=hba,
                delta_H=haa-hbb,gate_contribution=gate,rank_contribution=rank))
    frame=pd.DataFrame(rows)
    source=pd.read_csv(r.OUT/'fold_metrics.csv')
    for row in frame.itertuples():
        expected=source[source.outer_fold.eq(row.outer_fold)&source.policy.eq(row.policy)].score.iloc[0]
        assert abs(expected-row.new_H)<1e-12
    frame.to_csv(r.OUT/'gate_rank_accounting_by_fold.csv',index=False)
    summary=frame.groupby('policy',sort=False)[['delta_H','gate_contribution','rank_contribution']].mean()
    summary.to_csv(r.OUT/'gate_rank_accounting.csv')
    return summary


if __name__=='__main__':
    print(diagnose().to_string())
