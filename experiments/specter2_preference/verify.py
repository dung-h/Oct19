#!/usr/bin/env python3
"""Recompute nested choices/metrics, audit adapter training, and reload final model."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import modeling as m
import run_preference as run


def load_inner(ctx,fit,nums,prefix):
    values={a:{e:np.full((len(fit),5),np.nan) for e in m.CHECKPOINTS} for a in run.ARMS}
    values['incumbent']={0:np.full((len(fit),5),np.nan)}
    count=0
    head_hashes=set()
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask]
        assert set(ctx['groups'][a]).isdisjoint(ctx['groups'][b])
        for arm in run.ARMS:
            path=run.OUT/f'{prefix}_inner{k}_{arm}'
            z=np.load(str(path)+'.npz');info=json.loads(Path(str(path)+'.json').read_text())
            assert np.array_equal(z['fit'],a) and np.array_equal(z['pred'],b)
            assert info['fit']==a.tolist() and info['pred']==b.tolist()
            assert info['seed']==m.SEED and info['epochs']==m.CHECKPOINTS
            assert len(info['trace'])==max(m.CHECKPOINTS)
            assert all(np.isfinite(row['loss']) for row in info['trace'])
            expected_steps=math.ceil(len(a)/m.BATCH)*max(m.CHECKPOINTS)
            if arm=='adapted':
                assert info['trained_encoder_parameters']==894528 and info['adapter_l2_drift']>0
                assert info['nonzero_adapter_gradient_steps']+info['skipped_overflow_steps']==expected_steps
            else:
                assert info['trained_encoder_parameters']==0 and info['adapter_l2_drift']==0
                assert info['nonzero_adapter_gradient_steps']==0
            head_hashes.add(info['initial_head_sha256'])
            for e in m.CHECKPOINTS:
                p=z[f'epoch{e}']
                assert p.shape==(len(b),5) and np.isfinite(p).all() and (p>=0).all()
                np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
                values[arm][e][mask]=p
            count+=1
        values['incumbent'][0][mask],_=run.load_incumbent(f'{prefix}_inner{k}',a,b)
    assert len(head_hashes)==1
    return values,count


def main():
    torch.set_num_threads(4)
    with threadpool_limits(4):
        ctx=run.context();run.protocol(ctx);m.prepare(ctx,run.OUT)
        data=ctx['data'];y=ctx['y']
        original,_=run.ab.baseline.load_data()
        assert np.array_equal(data.id.iloc[:2494],original.id.iloc[:2494])
        np.testing.assert_array_equal(y[:2494],original.Label.iloc[:2494].astype(int))
        assert np.isnan(y[2494:]).all()
        # Re-encode every title using the pinned pretrained model, independently of cache.
        encoder=m.encoder(False);largest=0.
        order=np.argsort([len(x) for x in ctx['tokens']['input_ids']],kind='stable')
        with torch.inference_mode():
            for start in range(0,len(order),m.BATCH):
                ix=order[start:start+m.BATCH]
                with torch.autocast('cuda',dtype=torch.float16):v=m.embed(encoder,ctx,ix).cpu().numpy()
                largest=max(largest,float(np.max(np.abs(v-ctx['vectors'][ix]))))
        assert largest<.002
        del encoder;torch.cuda.empty_cache()
        frame=pd.read_csv(run.OUT/'fold_metrics.csv')
        recall_frame=pd.read_csv(run.OUT/'recall80_fold_metrics.csv')
        oof=pd.read_csv(run.OUT/'oof_predictions.csv',dtype={'id':str})
        assert len(oof)==2494 and oof.id.is_unique and oof.outer_fold.notna().all()
        pd.testing.assert_series_equal(oof.id,data.id.iloc[:2494],check_names=False)
        metrics_checked=0;inner_fits=0;outer_fits=0;seen=[];choices_checked=0
        for fold,fit,val,nums in run.outer_members(ctx):
            assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][val])
            seen.extend(val.tolist())
            values,n=load_inner(ctx,fit,nums,f'outer{fold}');inner_fits+=n
            choices,recall_choices,search=run.choose(ctx,fit,nums,values)
            saved=json.loads((run.OUT/f'outer{fold}_choices.json').read_text())
            assert saved==dict(primary=choices,recall80=recall_choices)
            choices_checked+=len(choices)+len(recall_choices)
            probabilities={};train={}
            for arm in run.ARMS:
                e=choices[arm]['epoch'];path=run.OUT/f'outer{fold}_validation_{arm}'
                z=np.load(str(path)+'.npz');info=json.loads(Path(str(path)+'.json').read_text())
                assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],val)
                assert np.array_equal(z['epochs'],[e]) and len(info['trace'])==e
                assert info['epochs']==[e] and info['arm']==arm
                if arm=='adapted':
                    assert info['adapter_l2_drift']>0
                    assert info['nonzero_adapter_gradient_steps']+info['skipped_overflow_steps']==math.ceil(len(fit)/m.BATCH)*e
                probabilities[arm]=z[f'epoch{e}'];train[arm]=z['train'];outer_fits+=1
            probabilities['incumbent'],train['incumbent']=run.load_incumbent(f'outer{fold}_validation',fit,val,True)
            for family,choice in choices.items():
                a=choice['family']
                actual,p=run.measure(ctx,fit,val,probabilities[a],train[a],choice['threshold'])
                independent=run.ab.evaluate(y[val],p,data.id.iloc[val].to_numpy())
                for key,value in independent.items():assert abs(actual[key]-value)<1e-12
                row=frame[frame.outer_fold.eq(fold)&frame.family.eq(family)].iloc[0]
                for key,value in actual.items():assert abs(row[key]-value)<1e-10,(fold,family,key)
                np.testing.assert_allclose(oof.loc[val,family+'_priority'],p,atol=1e-12)
                assert oof.loc[val,'outer_fold'].eq(fold).all()
                for k in range(5):np.testing.assert_allclose(oof.loc[val,family+f'_p{k+1}'],probabilities[a][:,k],atol=1e-12)
                metrics_checked+=1
            for family,choice in recall_choices.items():
                actual,_=run.measure(ctx,fit,val,probabilities[family],train[family],choice['threshold'])
                p=m.priority(probabilities[family],choice['threshold'])
                independent=run.ab.evaluate(y[val],p,data.id.iloc[val].to_numpy())
                for key,value in independent.items():assert abs(actual[key]-value)<1e-12
                row=recall_frame[recall_frame.outer_fold.eq(fold)&recall_frame.family.eq(family)].iloc[0]
                for key,value in actual.items():assert abs(row[key]-value)<1e-10
                metrics_checked+=1
        assert sorted(seen)==list(range(2494))
        # Incumbent reproduction must match the existing fixed reference exactly.
        old=pd.read_csv(run.CLAIM/'oof_predictions.csv',dtype={'id':str})
        np.testing.assert_allclose(oof.incumbent_priority,old.reference_seed20261018_priority,atol=1e-12)
        fit=np.arange(2494);test=np.arange(2494,len(data))
        membership=pd.read_csv(run.OUT/'final_inner_splits.csv',dtype={'id':str})
        assert membership.id.tolist()==data.id.iloc[:2494].tolist()
        values,n=load_inner(ctx,fit,membership.inner_fold.to_numpy(),'final');inner_fits+=n
        choices,_,_=run.choose(ctx,fit,membership.inner_fold.to_numpy(),values)
        assert choices==json.loads((run.OUT/'final_choices.json').read_text())
        choices_checked+=len(choices)
        challenger=choices['neural_selection']
        saved=np.load(run.OUT/'final_neural.npz')
        assert np.array_equal(saved['fit'],fit) and np.array_equal(saved['pred'],test)
        reloaded=m.reload_predict(ctx,test,run.OUT/'neural_model')
        np.testing.assert_allclose(reloaded,saved[f"epoch{challenger['epoch']}"],atol=1e-7,rtol=1e-7)
        priority=m.priority(reloaded,challenger['threshold'])
        challenger_csv=pd.read_csv(run.OUT/'submission_neural_challenger.csv',dtype={'id':str})
        challenger_csv=challenger_csv.set_index('id').loc[data.id.iloc[test]]
        np.testing.assert_allclose(challenger_csv.priority,priority,atol=1e-10)
        assert sorted(range(len(test)),key=lambda i:(-priority[i],data.id.iloc[test[i]]))==sorted(range(len(test)),key=lambda i:(-challenger_csv.priority.iloc[i],data.id.iloc[test[i]]))
        candidate=pd.read_csv(run.OUT/'submission_candidate.csv',dtype={'id':str})
        for f in [candidate,challenger_csv.reset_index()]:
            assert list(f.columns)==['id','priority'] and len(f)==596 and f.id.is_unique
            assert set(f.id)==set(ctx['template'].id) and np.isfinite(f.priority).all() and (f.priority>=0).all()
        expected=run.PRIOR/'submission_candidate.csv' if choices['selection_procedure']['family']=='incumbent' else run.OUT/'submission_neural_challenger.csv'
        assert m.digest(expected)==m.digest(run.OUT/'submission_candidate.csv')
        result=dict(status='PASSED',original_labels_preserved=True,all_title_embeddings_reencoded=True,
                    frozen_cache_max_abs_difference=largest,inner_fits_checked=inner_fits,outer_fits_checked=outer_fits,
                    metric_rows_recomputed=metrics_checked,inner_choices_checked=choices_checked,
                    adapter_gradients_and_updates_verified=True,final_model_reload=True,final_ranking_identical=True,
                    test_rows=596,candidate_selected=int(candidate.priority.gt(0).sum()),
                    challenger_selected=int((priority>0).sum()),candidate_sha256=m.digest(run.OUT/'submission_candidate.csv'),
                    uploaded=False)
        (run.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        run.report()
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
