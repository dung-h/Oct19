#!/usr/bin/env python3
"""Recompute nested choices/metrics and verify cached checkpoint replay/final model."""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import run_longer as r
m=r.m;prior=r.prior


def metric(y,priority,ids):
    order=np.asarray(sorted(np.flatnonzero(priority>0),key=lambda i:(-priority[i],str(ids[i]))),dtype=int)
    w=1/np.log2(np.arange(len(order))+2)
    q=float(np.average((y[order]-1)/4,weights=w)) if len(order) else 0.
    tp=int((y[order]>1).sum());positive=int((y>1).sum());recall=tp/positive if positive else 0.
    return dict(score=2*q*recall/(q+recall) if q+recall else 0.,relevance=q,coverage=recall,
        TP=tp,FP=len(order)-tp,FN=positive-tp,selected=len(order),
        missed_high=int(((y>=4)&(priority==0)).sum()))


def choices(ctx,fit,nums,prefix):
    values=r.inner(ctx,fit,nums,prefix)
    for k in range(4):
        name=f'{prefix}_inner{k}_adapted'
        fresh=np.load(r.OUT/(name+'.npz'));old=np.load(prior.OUT/(name+'.npz'))
        assert np.array_equal(fresh['fit'],old['fit']) and np.array_equal(fresh['pred'],old['pred'])
        np.testing.assert_allclose(fresh['epoch8'],old['epoch8'],atol=1e-6,rtol=1e-6)
    return r.choose(ctx,fit,nums,values)[0]


def same_choices(a,b):
    assert a.keys()==b.keys()
    for k in a:
        assert a[k]['family']==b[k]['family'] and a[k]['epoch']==b[k]['epoch']
        assert abs(a[k]['threshold']-b[k]['threshold'])<1e-12
        assert abs(a[k]['inner_score']-b[k]['inner_score'])<1e-12


def main():
    torch.set_num_threads(4);m.CHECKPOINTS=r.EPOCHS.copy()
    with threadpool_limits(4):
        ctx=r.context();m.CHECKPOINTS=[1,2,4,8];prior.protocol(ctx);m.CHECKPOINTS=r.EPOCHS.copy();protocol=r.protocol(ctx)
        m.prepare(ctx,prior.OUT)
        original,_=prior.ab.baseline.load_data()
        np.testing.assert_array_equal(ctx['y'][:2494],original.Label.iloc[:2494].astype(int))
        assert np.isnan(ctx['y'][2494:]).all()
        oof=pd.read_csv(r.OUT/'oof.csv',dtype={'id':str});frame=pd.read_csv(r.OUT/'fold_metrics.csv')
        assert oof.id.tolist()==ctx['data'].id.iloc[:2494].tolist()
        checked=0;partition_count=0;seen=[]
        for fold,fit,val,nums in prior.outer_members(ctx):
            saved=choices(ctx,fit,nums,f'outer{fold}')
            same_choices(saved,json.loads((r.OUT/f'outer{fold}_choices.json').read_text()))
            pred,_=r.partition(ctx,fit,val,f'outer{fold}_validation');inc,_=prior.load_incumbent(f'outer{fold}_validation',fit,val)
            z=np.load(r.OUT/f'outer{fold}_validation_adapted.npz')
            assert np.array_equal(z['epochs'],r.EPOCHS)
            old=np.load(prior.OUT/f'outer{fold}_validation_adapted.npz')
            assert np.array_equal(z['fit'],old['fit']) and np.array_equal(z['pred'],old['pred'])
            np.testing.assert_allclose(z['epoch8'],old['epoch8'],atol=1e-6,rtol=1e-6)
            preds={8:pred[8],16:pred[16],0:inc}
            for policy,choice in saved.items():
                fam=choice['family'];prob=preds[{'adapted_8':8,'adapted_16':16,'incumbent':0}[fam]]
                result,priority=r.evaluate(ctx['y'][val],choice,ctx['data'].id.iloc[val].to_numpy(),prob)
                independently=metric(ctx['y'][val],priority,ctx['data'].id.iloc[val].to_numpy())
                for k,v in independently.items():assert abs(result[k]-v)<1e-12
                assert abs(result['AUC']-roc_auc_score(ctx['y'][val]>1,1-prob[:,0]))<1e-12
                row=frame[frame.outer_fold.eq(fold)&frame.policy.eq(policy)].iloc[0]
                for k,v in result.items():assert abs(row[k]-v)<1e-10,(fold,policy,k)
                np.testing.assert_allclose(oof.loc[val,policy+'_priority'],priority,atol=1e-12)
                checked+=1
            seen.extend(val.tolist());partition_count+=1
        assert sorted(seen)==list(range(2494))
        fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
        membership=pd.read_csv(prior.PRIOR/'final_inner_splits.csv',dtype={'id':str})
        final_choices=choices(ctx,fit,membership.inner_fold.to_numpy(),'final')
        same_choices(final_choices,json.loads((r.OUT/'final_choices.json').read_text()))
        pred,_=r.partition(ctx,fit,test,'final_test');choice=final_choices['selection_procedure']
        fresh=np.load(r.OUT/'final_test_adapted.npz')
        old_path=prior.OUT/'final_test_adapted.npz'
        if old_path.exists():
            old=np.load(old_path)
            np.testing.assert_allclose(fresh['epoch8'],old['epoch8'],atol=1e-6,rtol=1e-6)
            final_test_replay='matched archived prior prediction'
        else:
            # The original experiment never wrote a final-test cache. Refit
            # its exact epoch-8 recipe into an isolated verifier directory so
            # the longer run's first-eight-update prediction is still checked.
            replay_dir=r.OUT/'verification_replay';replay_dir.mkdir(exist_ok=True)
            m.CHECKPOINTS=[8]
            replay,_,_=m.fit(ctx,fit,test,'adapted',[8],
                replay_dir/'final_test_adapted',save_dir=replay_dir/'final_test_model')
            m.CHECKPOINTS=r.EPOCHS.copy()
            np.testing.assert_allclose(fresh['epoch8'],replay[8],atol=1e-6,rtol=1e-6)
            final_test_replay='matched independent epoch-8 replay; prior cache absent'
        raw=pred[choice['epoch']] if choice['family'].startswith('adapted') else None
        if raw is not None:
            model=m.reload_predict(ctx,test,r.OUT/'final_model')
            np.testing.assert_allclose(model,raw,atol=1e-7,rtol=1e-7)
            partition_count+=1
        table=pd.read_csv(r.OUT/'submission_candidate.csv',dtype={'id':str})
        assert table.id.tolist()==ctx['template'].id.tolist() and len(table)==596 and table.id.is_unique
        assert np.isfinite(table.priority).all() and (table.priority>=0).all()
        recipe=json.loads((r.OUT/'final_recipe.json').read_text())
        assert recipe['family']==choice['family'] and recipe['epoch']==(choice['epoch'] if choice['family'].startswith('adapted') else None)
        incumbent_hash=m.digest(prior.PRIOR/'submission_candidate.csv')
        result=dict(status='PASSED',outer_policy_metric_rows_recomputed=checked,
            final_inner_choices_recomputed=len(final_choices),outer_partitions_checked=partition_count,
            checkpoint8_replay_matches_archived=True,final_test_epoch8_replay=final_test_replay,
            original_labels_preserved=True,
            final_model_reload=(raw is not None),final_candidate_valid=True,
            incumbent_sha256_unchanged=incumbent_hash=='8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf',uploaded=False)
        (r.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
