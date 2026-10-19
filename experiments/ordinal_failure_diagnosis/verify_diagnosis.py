#!/usr/bin/env python3
"""Independent selection/metric verification and reloads for all diagnostic controls."""
import json
import importlib.util

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import run_controls as diag
import defaults_control as dc
base=diag.base
spec=importlib.util.spec_from_file_location('ordinal_independent_verifier',base.HERE/'verify.py')
independent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)
metric,view,priorities=independent.metric,independent.view,independent.priorities


def cached_new(ctx,fit,pred,name,family,training=False,defaults=False):
    root=dc.OUT if defaults else diag.OUT
    path=root/(name+'.npz' if defaults else f'{name}_{family}.npz')
    z=np.load(path)
    assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
    raw=z['raw'];v=view(raw,True)
    assert raw.shape==(len(pred),4) and np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all()
    if defaults:
        params=json.loads((root/(name+'_params.json')).read_text())
        assert len(params)==4
        for p in params:
            assert p['bootstrap_type']=='Bayesian' and p['bagging_temperature']==1 and p['leaf_estimation_iterations']==1
            assert p['depth']==4 and p['l2_leaf_reg']==8 and p['iterations']==300 and p['random_seed']==base.SEED
    else:
        info=json.loads((root/f'{name}_{family}.json').read_text())
        assert info['fit']==fit.tolist() and info['pred']==pred.tolist()
        assert info['columns']==diag.columns(ctx,family) and len(info['models'])==4
        for j,k in enumerate([2,3,4,5]):assert info['models'][j]['target']==k and info['models'][j]['positive']==int((ctx['y'][fit]>=k).sum())
    return v,view(z['train_raw'],True) if training else None


def collect_inner(ctx,fit,nums,prefix,defaults=False):
    families=['multiclass','ordinal'] if defaults else diag.FAMILIES
    arrays={f:np.full((len(fit),4 if f.startswith('ORD') or f=='ordinal' else 5),np.nan) for f in families}
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask];name=f'{prefix}_inner{k}'
        assert set(ctx['groups'][a]).isdisjoint(ctx['groups'][b])
        if defaults:
            v,_=cached_new(ctx,a,b,name,'ordinal',defaults=True)
            arrays['ordinal'][mask]=v['raw']
            inc,_=base.incumbent(a,b,name);arrays['multiclass'][mask]=inc['probability']
        else:
            old,_=diag.old_views(a,b,name)
            for f in diag.NEW:old[f],_=cached_new(ctx,a,b,name,f)
            for f,v in old.items():arrays[f][mask]=v['raw'] if f.startswith('ORD') else v['probability']
    return {f:view(a,f.startswith('ORD') or f=='ordinal') for f,a in arrays.items()}


def choose(ctx,fit,nums,views):
    choices={}
    for f,v in views.items():
        rows=[]
        for t in base.CUTOFFS:
            p=priorities(v,t)
            h=np.mean([metric(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]].to_numpy())['score'] for k in range(4)])
            rows.append(dict(family=f,threshold=float(t),inner_score=float(h)))
        choices[f]=max(rows,key=lambda r:r['inner_score'])
    choices['selection_procedure']=max(choices.values(),key=lambda r:r['inner_score']).copy()
    return choices


def check_choices(a,b):
    assert a.keys()==b.keys()
    for f in a:
        assert a[f]['family']==b[f]['family'] and abs(a[f]['threshold']-b[f]['threshold'])<1e-12
        assert abs(a[f]['inner_score']-b[f]['inner_score'])<1e-12


def main():
    with threadpool_limits(4):
        ctx=base.context();base.protocol(ctx);diag.protocol(ctx)
        protocol=json.loads((dc.OUT/'protocol.json').read_text())
        assert protocol['source_sha256']==base.digest(dc.__file__)
        assert protocol['plan_sha256']==base.digest(diag.HERE/'DEFAULTS_PLAN.md')
        original,_=base.ab.baseline.load_data()
        np.testing.assert_array_equal(ctx['y'][:2494],original.Label.iloc[:2494].astype(int))
        counts=0;choices_count=0;seen=[]
        for defaults in [False,True]:
            root=dc.OUT if defaults else diag.OUT
            frame=pd.read_csv(root/('fold_metrics.csv' if defaults else 'control_fold_metrics.csv'))
            oof=pd.read_csv(root/('oof.csv' if defaults else 'control_oof.csv'),dtype={'id':str})
            assert oof.id.tolist()==ctx['data'].id.iloc[:2494].tolist()
            for fold,fit,val,nums in base.previous.outer_members(ctx['data']):
                if not defaults:seen.extend(val.tolist())
                assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][val])
                choice=choose(ctx,fit,nums,collect_inner(ctx,fit,nums,f'outer{fold}',defaults))
                check_choices(choice,json.loads((root/f'outer{fold}_choices.json').read_text()));choices_count+=len(choice)
                if defaults:
                    v,t=cached_new(ctx,fit,val,f'outer{fold}_validation','ordinal',True,True)
                    inc,ti=base.incumbent(fit,val,f'outer{fold}_validation',True)
                    views=dict(multiclass=inc,ordinal=v);trains=dict(multiclass=ti,ordinal=t)
                else:
                    views,trains=diag.old_views(fit,val,f'outer{fold}_validation',True)
                    for f in diag.NEW:views[f],trains[f]=cached_new(ctx,fit,val,f'outer{fold}_validation',f,True)
                for family,c in choice.items():
                    v=views[c['family']];p=priorities(v,c['threshold'])
                    expected=metric(ctx['y'][val],p,ctx['data'].id.iloc[val].to_numpy())
                    expected.update(AUC=roc_auc_score(ctx['y'][val]>1,v['read']),gain_MAE=np.abs(v['gain']-(ctx['y'][val]-1)/4).mean())
                    if c['family'] in trains:
                        expected['train_AUC']=roc_auc_score(ctx['y'][fit]>1,trains[c['family']]['read'])
                        expected['AUC_gap']=expected['train_AUC']-expected['AUC']
                    row=frame[frame.outer_fold.eq(fold)&frame.family.eq(family)].iloc[0]
                    for k,value in expected.items():assert abs(row[k]-value)<1e-10,(defaults,fold,family,k)
                    np.testing.assert_allclose(oof.loc[val,family+'_priority'],p,atol=1e-12)
                    counts+=1
        assert sorted(seen)==list(range(2494))
        # Rebuild all test features from the original fold-local recipe.
        oldctx=base.previous.context();data,_,y,g,title,retrieval,content,_=oldctx
        fit=np.arange(2494);test=np.arange(2494,len(data))
        pack=base.feature_pack(ctx,'final_test',fit,test)
        rebuilt=base.previous.feat.rebuild_prediction(data,y,g,title,retrieval,content,fit,test,pack['recipe'])
        pd.testing.assert_frame_equal(rebuilt,pack['b'],check_dtype=False,atol=1e-11,rtol=1e-11)
        nums=pd.read_csv(base.PRIOR/'final_inner_splits.csv').inner_fold.to_numpy()
        final_views={};reloaded=0
        for family in diag.NEW:
            raw=[]
            for k in [2,3,4,5]:
                model=CatBoostClassifier();model.load_model(str(diag.OUT/f'{family}_ge{k}.cbm'))
                params=model.get_all_params();strong=family.endswith('regularized')
                assert params['depth']==(3 if strong else 4) and params['l2_leaf_reg']==(32 if strong else 8)
                assert params['bootstrap_type']=='MVS' and params['leaf_estimation_iterations']==10
                assert model.feature_names_==diag.columns(ctx,family)
                raw.append(model.predict_proba(rebuilt[diag.columns(ctx,family)])[:,1]);reloaded+=1
            raw=np.column_stack(raw);cached,_=cached_new(ctx,fit,test,'final_test',family)
            np.testing.assert_allclose(raw,cached['raw'],atol=1e-12);final_views[family]=view(raw,True)
        for family in ['MC_full','MC_regularized','MC_without_neighbors']:
            model=CatBoostClassifier();model.load_model(str(base.PRIOR/'final_classifier.cbm' if family=='MC_full' else diag.OUT/(family+'.cbm')))
            p=model.predict_proba(rebuilt[diag.columns(ctx,family)]);final_views[family]=view(p,False);reloaded+=1
            z=np.load(diag.OUT/'final_predictions.npz');np.testing.assert_allclose(p,z[family],atol=1e-12)
        z=np.load(base.OUT/'final_test.npz');final_views['ORD_full']=view(z['raw'],True)
        choice=choose(ctx,fit,nums,collect_inner(ctx,fit,nums,'final'))
        check_choices(choice,json.loads((diag.OUT/'final_choices.json').read_text()));choices_count+=len(choice)
        def verify_csv(path,v,c):
            f=pd.read_csv(path,dtype={'id':str});p=priorities(v,c['threshold'])
            byid=dict(zip(data.id.iloc[test],p));expected=np.array([byid[i] for i in f.id])
            assert len(f)==596 and f.id.is_unique and f.id.tolist()==ctx['template'].id.tolist()
            assert np.isfinite(f.priority).all() and (f.priority>=0).all()
            np.testing.assert_allclose(f.priority,expected,atol=1e-11)
            sorter=lambda a:sorted(np.flatnonzero(a>0),key=lambda i:(-a[i],f.id.iloc[i]))
            assert sorter(f.priority.to_numpy())==sorter(expected)
        for f,v in final_views.items():verify_csv(diag.OUT/f'submission_{f}.csv',v,choice[f])
        source=choice['selection_procedure']['family']
        assert base.digest(diag.OUT/'submission_candidate.csv')==base.digest(diag.OUT/f'submission_{source}.csv')
        # Explicit-default branch final model parameters and probabilities.
        raw=[]
        for k in [2,3,4,5]:
            model=CatBoostClassifier();model.load_model(str(dc.OUT/f'ordinal_ge{k}.cbm'));params=model.get_all_params()
            assert params['bootstrap_type']=='Bayesian' and params['bagging_temperature']==1 and params['leaf_estimation_iterations']==1
            raw.append(model.predict_proba(rebuilt[ctx['columns']])[:,1]);reloaded+=1
        raw=np.column_stack(raw);v,_=cached_new(ctx,fit,test,'final_test','ordinal',defaults=True)
        np.testing.assert_allclose(raw,v['raw'],atol=1e-12)
        choice=choose(ctx,fit,nums,collect_inner(ctx,fit,nums,'final',True))
        check_choices(choice,json.loads((dc.OUT/'final_choices.json').read_text()));choices_count+=len(choice)
        verify_csv(dc.OUT/'submission_ordinal.csv',view(raw,True),choice['ordinal'])
        expected_path=base.PRIOR/'submission_candidate.csv' if choice['selection_procedure']['family']=='multiclass' else dc.OUT/'submission_ordinal.csv'
        assert base.digest(dc.OUT/'submission_candidate.csv')==base.digest(expected_path)
        assert base.digest(base.PRIOR/'submission_candidate.csv')=='8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf'
        mechanism=json.loads((diag.OUT/'mechanisms/summary.json').read_text())
        assert mechanism['pav_changed_selection_count']==0
        h=pd.read_csv(diag.OUT/'mechanisms/H_decomposition.csv')
        np.testing.assert_allclose(h.delta_H,h.gate_effect+h.rank_effect,atol=1e-12)
        result=dict(status='PASSED',original_labels_preserved=True,outer_metrics_recomputed=counts,
            inner_choices_recomputed=choices_count,new_binary_fits_checked=360,final_models_reloaded=reloaded,
            final_features_rebuilt=True,predictions_and_rankings_identical=True,incumbent_csv_unchanged=True,
            causal_caveat='Interventions identify package-level effects in this configuration; reused outer data is exploratory',uploaded=False)
        (diag.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
