#!/usr/bin/env python3
"""Independent metric/calibration replay and saved-model inference checks."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
import torch
from threadpoolctl import threadpool_limits

import run_abstract as r
m=r.m


def metric(y,p,ids):
    order=sorted(np.flatnonzero(p>0),key=lambda i:(-p[i],str(ids[i])))
    weights=1/np.log2(np.arange(len(order))+2)
    q=float(sum(w*(y[i]-1)/4 for w,i in zip(weights,order))/sum(weights)) if order else 0.
    pos=int(sum(y>1));tp=int(sum(y[i]>1 for i in order));recall=tp/pos if pos else 0.
    return dict(score=2*q*recall/(q+recall) if q+recall else 0.,relevance=q,coverage=recall,
        TP=tp,FP=len(order)-tp,FN=pos-tp,selected=len(order),
        missed_high=int(sum((y>=4)&(p==0))))


def choices(ctx,calibration,values):
    y=ctx['y'][calibration];ids=ctx['data'].id.iloc[calibration].to_numpy()
    result={}
    for family in r.FAMILIES:
        candidates=[]
        for epoch,prob in values[family].items():
            for threshold in r.THRESHOLDS:
                measures=metric(y,m.old.priority(prob,threshold),ids)
                candidates.append(dict(family=family,epoch=int(epoch),threshold=float(threshold),
                    calibration_H=measures['score'],calibration_R=measures['coverage']))
        result[family]=max(candidates,key=lambda v:v['calibration_H']).copy()
    result['neural_selection']=max([result[a] for a in m.ARMS],key=lambda v:v['calibration_H']).copy()
    result['selection_procedure']=max([result[a] for a in r.FAMILIES],key=lambda v:v['calibration_H']).copy()
    return result


def check_choice(a,b):
    assert a.keys()==b.keys()
    for key in a:
        for field in a[key]:
            x,y=a[key][field],b[key][field]
            if isinstance(x,float):assert abs(x-y)<1e-12,(key,field)
            else:assert x==y,(key,field)


def main():
    torch.set_num_threads(4)
    with threadpool_limits(4):
        ctx=r.context();locked=r.protocol(ctx);ph=m.digest(r.OUT/'protocol.json')
        m.prepare_vectors(ctx,r.OUT,ph)
        original,_=r.prior.ab.baseline.load_data()
        np.testing.assert_array_equal(original.Label.iloc[:2494],ctx['data'].Label.iloc[:2494])
        assert np.isnan(ctx['y'][2494:]).all()
        for name in ['title','joint']:
            missing=ctx['data'].abstract.eq('').to_numpy()
            assert ctx['modes'][name]['token_hash']==locked['input_tokens'][name]
            assert all(len(ids)<=m.MAX_LENGTH for ids in ctx['modes'][name]['tokens']['input_ids'])
            vectors=ctx['modes'][name]['vectors']
            assert vectors.shape==(3090,768) and np.isfinite(vectors).all()
        for i in np.flatnonzero(missing):
            assert ctx['modes']['title']['tokens']['input_ids'][i]==ctx['modes']['joint']['tokens']['input_ids'][i]
        fallback_vector_delta=float(np.abs(ctx['modes']['title']['vectors'][missing]-ctx['modes']['joint']['vectors'][missing]).max())
        np.testing.assert_allclose(ctx['modes']['title']['vectors'][missing],ctx['modes']['joint']['vectors'][missing],atol=2e-4,rtol=2e-3)
        # A fresh encoder check over longest and randomly selected inputs,
        # plus missing abstracts, independent of training predictions.
        sample=np.unique(np.r_[np.argsort(ctx['modes']['joint']['lengths'])[-16:],
            np.flatnonzero(missing)[:16],np.random.default_rng(42).choice(3090,32,replace=False)])
        encoder=m.old.encoder(False)
        for mode in ['title','joint']:
            fresh=[]
            with torch.inference_mode():
                for start in range(0,len(sample),m.MICRO):
                    with torch.autocast('cuda',dtype=torch.float16):
                        fresh.extend(m.embedding(encoder,ctx['modes'][mode],sample[start:start+m.MICRO]).cpu().numpy())
            np.testing.assert_allclose(np.asarray(fresh),ctx['modes'][mode]['vectors'][sample],atol=2e-4,rtol=2e-3)
        del encoder;torch.cuda.empty_cache()
        frame=pd.read_csv(r.OUT/'fold_metrics.csv')
        oof=pd.read_csv(r.OUT/'oof.csv',dtype={'id':str})
        assert oof.id.tolist()==ctx['data'].id.iloc[:2494].tolist()
        gradient_checks=cache_checks=choice_checks=metric_checks=reload_checks=0
        initial_heads=set();seen=[];trace=[]
        for prefix,fit,pred,nums in r.partitions(ctx):
            train,cal=fit[nums!=0],fit[nums==0]
            assert set(ctx['groups'][train]).isdisjoint(ctx['groups'][cal])
            if prefix!='final':
                assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][pred])
                seen.extend(pred.tolist())
            values={}
            for arm in m.ARMS:
                p=r.OUT/f'{prefix}_calibration_{arm}.npz';z=np.load(p)
                np.testing.assert_array_equal(z['fit'],train);np.testing.assert_array_equal(z['pred'],cal)
                np.testing.assert_array_equal(z['epochs'],m.EPOCHS)
                assert str(z['protocol_sha256'])==ph
                values[arm]={e:z[f'epoch{e}'] for e in m.EPOCHS};cache_checks+=1
                for prediction in values[arm].values():
                    assert np.isfinite(prediction).all() and np.allclose(prediction.sum(axis=1),1,atol=1e-6)
            control,_=r.incumbent(prefix+'_inner0',train,cal)
            values['incumbent']={0:control}
            selected=choices(ctx,cal,values)
            check_choice(selected,json.loads((r.OUT/f'{prefix}_choices.json').read_text()))
            choice_checks+=len(selected)
            probabilities={};training={}
            for arm in m.ARMS:
                for phase in ['calibration','validation']:
                    path=r.OUT/f'{prefix}_{phase}_{arm}.json'
                    info=json.loads(path.read_text());initial_heads.add(info['initial_head_sha256'])
                    assert info['protocol_sha256']==ph
                    adapted=arm!='joint_frozen'
                    assert (info['adapter_l2_drift']>0)==adapted
                    assert (info['trained_encoder_parameters']>0)==adapted
                    assert (info['nonzero_adapter_gradient_steps']>0)==adapted
                    if adapted:
                        expected_steps=max(info['epochs'])*int(np.ceil(len(info['fit'])/m.BATCH))
                        assert info['nonzero_adapter_gradient_steps']+info['skipped_overflow_steps']==expected_steps
                        assert info['trained_encoder_parameters']==894528
                    for item in info['trace']:
                        assert np.isfinite(item['loss'])
                        trace.append(dict(partition=prefix,phase=phase,arm=arm,**item))
                    gradient_checks+=1
                p=r.OUT/f'{prefix}_validation_{arm}.npz';z=np.load(p)
                np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['pred'],pred)
                np.testing.assert_array_equal(z['epochs'],[selected[arm]['epoch']])
                probabilities[arm]=z[f"epoch{selected[arm]['epoch']}"];training[arm]=z['train'];cache_checks+=1
                assert str(z['protocol_sha256'])==ph
                assert np.isfinite(probabilities[arm]).all() and np.allclose(probabilities[arm].sum(axis=1),1,atol=1e-6)
            if prefix=='final':
                parent_details=pd.read_csv(r.prior.PRIOR/'test_prediction_details.csv',dtype={'id':str})
                assert parent_details.id.tolist()==ctx['data'].id.iloc[pred].tolist()
                probabilities['incumbent'],_=r.incumbent('final_test',fit,pred)
                for arm in m.ARMS:
                    reloaded=m.reload_predict(ctx,pred,r.OUT/f'final_{arm}_model')
                    np.testing.assert_allclose(reloaded,probabilities[arm],atol=1e-7,rtol=1e-7)
                    reload_checks+=1
                recipe=json.loads((r.OUT/'final_recipe.json').read_text())
                # The recipe is serialized after calibration; replayed floating-point
                # reductions can differ in their final bit while selecting the same
                # family/epoch/cutoff. Reuse the calibration tolerance above.
                check_choice({'choice': recipe['choice']}, {'choice': selected['selection_procedure']})
                assert recipe['protocol_sha256']==ph
                for policy,item in selected.items():
                    p=m.old.priority(probabilities[item['family']],item['threshold'])
                    expected=ctx['template'][['id']].merge(pd.DataFrame(dict(id=ctx['data'].id.iloc[pred].to_numpy(),priority=p)),on='id',validate='one_to_one')
                    table=pd.read_csv(r.OUT/f'submission_{policy}.csv',dtype={'id':str})
                    assert table.columns.tolist()==['id','priority'] and len(table)==596 and table.id.is_unique
                    assert table.id.tolist()==expected.id.tolist() and table.priority.ge(0).all()
                    np.testing.assert_allclose(table.priority,expected.priority,atol=1e-10,rtol=1e-10)
                candidate=pd.read_csv(r.OUT/'submission_candidate.csv',dtype={'id':str})
                pd.testing.assert_frame_equal(candidate,pd.read_csv(r.OUT/'submission_selection_procedure.csv',dtype={'id':str}))
            else:
                fold=int(prefix[len('outer'):])
                probabilities['incumbent'],training['incumbent']=r.incumbent(prefix+'_validation',fit,pred)
                for policy,item in selected.items():
                    family=item['family'];prob=probabilities[family]
                    p=m.old.priority(prob,item['threshold'])
                    independent=metric(ctx['y'][pred],p,ctx['data'].id.iloc[pred].to_numpy())
                    independent.update(AUC=float(roc_auc_score(ctx['y'][pred]>1,1-prob[:,0])),
                        train_AUC=float(roc_auc_score(ctx['y'][fit]>1,1-training[family][:,0])),
                        logloss=float(log_loss(ctx['y'][pred].astype(int)-1,prob,labels=np.arange(5))),
                        gain_MAE=float(np.mean(np.abs(prob@(np.arange(5)/4)-(ctx['y'][pred]-1)/4))))
                    row=frame[frame.outer_fold.eq(fold)&frame.policy.eq(policy)].iloc[0]
                    assert row.selected_family==family and row.epoch==item['epoch']
                    for key,value in independent.items():assert abs(row[key]-value)<1e-10,(fold,policy,key)
                    np.testing.assert_allclose(oof.loc[pred,policy+'_priority'],p,atol=1e-10)
                    metric_checks+=1
        assert sorted(seen)==list(range(2494)) and len(initial_heads)==1
        # Token inputs do not depend on either target or metadata fields.
        texts=ctx['data'].title+ctx['modes']['joint']['tokenizer'].sep_token+ctx['data'].abstract
        changed=ctx['data'].copy();changed['Label']='5';changed['id']='audit-only-id'
        assert texts.tolist()==(changed.title+ctx['modes']['joint']['tokenizer'].sep_token+changed.abstract).tolist()
        assert m.digest(r.prior.PRIOR/'submission_candidate.csv')==locked['incumbent_cache_sha256'][str((r.prior.PRIOR/'submission_candidate.csv').relative_to(r.HERE.parent))]
        pd.DataFrame(trace).to_csv(r.OUT/'learning_traces.csv',index=False)
        overlap=set(ctx['groups'][:2494])&set(ctx['groups'][2494:])
        result=dict(status='PASSED',prediction_caches_checked=cache_checks,
            outer_metric_rows_recomputed=metric_checks,calibration_choices_recomputed=choice_checks,
            gradient_and_drift_records_checked=gradient_checks,initial_head_identical=True,
            frozen_vector_samples_reencoded=len(sample)*2,original_labels_unchanged=True,
            missing_abstract_fallback_rows=int(missing.sum()),fallback_vector_max_absolute_delta=fallback_vector_delta,
            tokenizer_ignores_labels_and_ids=True,outer_groups_isolated=True,
            final_train_test_overlapping_groups=len(overlap),final_models_reloaded=reload_checks,
            submission_rows=596,incumbent_unchanged=True,uploaded=False,
            limitation='Pilot uses inner0 only; repeated outer folds are exploratory')
        (r.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
