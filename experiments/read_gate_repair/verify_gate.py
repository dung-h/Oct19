#!/usr/bin/env python3
"""Independent replay of final artifacts and all OOF decision counts."""
from __future__ import annotations

import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import author_features as authors
import gate_lib as g
import run_gate as runner


def main():
    ctx=g.context(); runner.protocol(ctx)
    verified=authors.source_ledger(ctx['data'],pd.read_csv(g.META,dtype={'id':str}))
    saved=pd.read_csv(g.OUT/'metadata_ledger.csv',dtype={'id':str})
    assert saved[['split','id','authors_used','author_source_verified','author_format_uncertain']].equals(
        verified[['split','id','authors_used','author_source_verified','author_format_uncertain']])
    frame=pd.read_csv(g.OUT/'outcomes_by_id.csv',dtype={'id':str})
    metrics=pd.read_csv(g.OUT/'fold_metrics.csv')
    checks=0
    for fold,fit,pred,nums in g.outer_members(ctx):
        ids=ctx['data'].id.iloc[pred].to_numpy(str); y=ctx['y'][pred].astype(int)
        probability=np.load(g.OUT/f'outer{fold}_validation_probabilities.npz')
        inc=g.load_incumbent_outer(fold,fit,pred);title=g.title_probability(f'outer{fold}',fit,pred,runner.title_epoch(f'outer{fold}'))
        replay={'incumbent_deployed':1-inc[:,0],'incumbent_at_050':1-inc[:,0]}
        replay.update({key:probability[key] for key in g.BRANCHES})
        for branch,p in replay.items():
            cutoff=.55 if branch=='incumbent_deployed' else .5
            selected=p>cutoff; priority=g.priority(selected,inc,title)
            part=frame[(frame.outer_fold==fold)&(frame.branch==branch)].reset_index(drop=True)
            assert part.id.tolist()==ids.tolist()
            np.testing.assert_allclose(part.read_probability,p,atol=1e-12)
            np.testing.assert_array_equal(part.selected.to_numpy(bool),selected)
            np.testing.assert_allclose(part.priority,priority,atol=1e-12)
            expected={**g.selection_metrics(y,selected),**g.official(y,priority,ids)}
            row=metrics[(metrics.outer_fold==fold)&(metrics.branch==branch)].iloc[0]
            for key,value in expected.items():assert abs(row[key]-value)<1e-10,(fold,branch,key)
            checks+=1
    summary=pd.read_csv(g.OUT/'summary.csv').set_index('branch')
    for branch,part in frame.groupby('branch'):
        expected={**g.selection_metrics(part.Label,part.selected),**g.official(part.Label.to_numpy(),part.priority.to_numpy(),part.id.to_numpy())}
        for key,value in expected.items():assert abs(summary.loc[branch,key]-value)<1e-10,(branch,key)
    # Rebuild authors for final test then reload both final models and calibrators.
    fit,pred,nums=g.final_partition(ctx); ba,bb,aa,ab,_=runner.pair(ctx,verified,'final_test',fit,pred)
    columns=joblib.load(g.OUT/'final_models'/'columns.joblib'); calibration=joblib.load(g.OUT/'final_calibrators.joblib')
    raw={}
    for family,features,cols in [('binary',bb,columns['base_columns']),('author',pd.concat([bb.reset_index(drop=True),ab.reset_index(drop=True)],axis=1),columns['author_columns'])]:
        clf=CatBoostClassifier();clf.load_model(str(g.OUT/'final_models'/f'{family}_model.cbm'))
        raw[family]=clf.predict_proba(features[cols])[:,1]
    venue=ctx['data'].venue.iloc[pred].fillna('unknown').astype(str).to_numpy();inc=g.load_incumbent_final(ctx,fit,pred);title=g.title_probability('final',fit,pred,runner.title_epoch('final'))
    replay={'incumbent_deployed':1-inc[:,0],'incumbent_at_050':1-inc[:,0]}
    for family in ['binary','author']:
        replay[f'{family}_raw']=raw[family]
        replay[f'{family}_global']=g.apply_calibrator(calibration[family]['global'],raw[family],venue)
        replay[f'{family}_venue']=g.apply_calibrator(calibration[family]['venue'],raw[family],venue)
    for branch,p in replay.items():
        expected_selected=p>(.55 if branch=='incumbent_deployed' else .5)
        detail=pd.read_csv(g.OUT/f'test_predictions_{branch}.csv',dtype={'id':str})
        np.testing.assert_allclose(detail.read_probability,p,atol=1e-10)
        np.testing.assert_array_equal(detail.selected.to_numpy(bool),expected_selected)
        np.testing.assert_allclose(detail.priority,g.priority(expected_selected,inc,title),atol=1e-10)
        submission=pd.read_csv(g.OUT/f'submission_{branch}.csv',dtype={'id':str})
        assert len(submission)==596 and submission.id.is_unique and np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
        checks+=1
    incumbent_sha='8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf'
    assert g.digest(g.INC/'submission_candidate.csv')==incumbent_sha
    result=dict(status='PASSED',outer_branch_metrics_recomputed=checks-8,final_branches_reloaded=8,
                final_author_features_rebuilt=True,incumbent_unchanged=True,uploaded=False)
    (g.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
