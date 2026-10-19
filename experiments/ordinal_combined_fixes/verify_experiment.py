#!/usr/bin/env python3
"""Independent scoring, threshold selection, model reload and feature rebuild."""
import importlib.util
import json

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import run_combined as r
base=r.base
spec=importlib.util.spec_from_file_location('independent_ordinal_checks',base.HERE/'verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def cached(ctx,fit,pred,name,training=False):
    views,trains=r.collect(ctx,fit,pred,name,training)
    # The verifier requires every fit to exist and never calls this unless all caches exist.
    for f in r.NEW:
        z=np.load(r.OUT/f'{name}_{f}.npz');raw=z['raw']
        assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
        assert np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all()
        independent=v.view(raw,f.startswith('ORD'))
        np.testing.assert_allclose(independent['probability'],views[f]['probability'],atol=1e-14)
        np.testing.assert_allclose(independent['gain'],views[f]['gain'],atol=1e-14)
        meta=json.loads((r.OUT/f'{name}_{f}.json').read_text())
        assert meta['columns']==ctx['newprotocol']['columns'][f]
        assert meta['feature_sha256']==base.digest(base.PRIOR/f'{name}_features.joblib')
        assert [item['target'] for item in meta['models']]==([2,3,4,5] if f.startswith('ORD') else [None])
        for item in meta['models']:
            p=item['params'];strong=f.endswith('regularized')
            assert p['bootstrap_type']=='Bayesian' and p['bagging_temperature']==1 and p['leaf_estimation_iterations']==1
            assert p['depth']==(3 if strong else 4) and p['l2_leaf_reg']==(32 if strong else 8)
            assert p['iterations']==300 and p['random_seed']==base.SEED
            assert p['loss_function']==('Logloss' if f.startswith('ORD') else 'MultiClass')
        views[f]=independent
        if training:trains[f]=v.view(z['train_raw'],f.startswith('ORD'))
    return views,trains


def inner(ctx,fit,nums,prefix):
    buffers={f:np.full((len(fit),4 if f.startswith('ORD') else 5),np.nan) for f in r.FAMILIES}
    for k in range(4):
        mask=nums==k;vs,_=cached(ctx,fit[~mask],fit[mask],f'{prefix}_inner{k}')
        for f,a in vs.items():buffers[f][mask]=a['raw'] if f.startswith('ORD') else a['probability']
    return {f:v.view(a,f.startswith('ORD')) for f,a in buffers.items()}


def choose(ctx,fit,nums,vs):
    choices={}
    for f in r.FAMILIES:
        options=[]
        for t in base.CUTOFFS:
            p=v.priorities(vs[f],t)
            h=np.mean([v.metric(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]].to_numpy())['score'] for k in range(4)])
            options.append(dict(family=f,threshold=float(t),inner_score=float(h)))
        choices[f]=max(options,key=lambda a:a['inner_score']).copy()
    choices['selection_procedure']=max(choices.values(),key=lambda a:a['inner_score']).copy()
    return choices


def main():
    with threadpool_limits(4):
        ctx=base.context();base.protocol(ctx);ctx['newprotocol']=r.protocol(ctx)
        expected_names=[f'outer{i}_inner{k}' for i in range(5) for k in range(4)]+[f'outer{i}_validation' for i in range(5)]+[f'final_inner{k}' for k in range(4)]+['final_test']
        for n in expected_names:
            for f in r.NEW:assert (r.OUT/f'{n}_{f}.npz').exists()
        original,_=base.ab.baseline.load_data()
        np.testing.assert_array_equal(ctx['y'][:2494],original.Label.iloc[:2494].astype(int))
        frame=pd.read_csv(r.OUT/'fold_metrics.csv');oof=pd.read_csv(r.OUT/'oof.csv',dtype={'id':str})
        assert oof.id.tolist()==ctx['data'].id.iloc[:2494].tolist()
        count=0;seen=[]
        for fold,fit,val,nums in base.previous.outer_members(ctx['data']):
            seen.extend(val.tolist());choices=choose(ctx,fit,nums,inner(ctx,fit,nums,f'outer{fold}'))
            v.check_choices(choices,json.loads((r.OUT/f'outer{fold}_choices.json').read_text()))
            vs,ts=cached(ctx,fit,val,f'outer{fold}_validation',True)
            for f,c in choices.items():
                source=c['family'];p=v.priorities(vs[source],c['threshold'])
                expected=v.metric(ctx['y'][val],p,ctx['data'].id.iloc[val].to_numpy())
                expected['AUC']=roc_auc_score(ctx['y'][val]>1,vs[source]['read'])
                if source in ts:expected['train_AUC']=roc_auc_score(ctx['y'][fit]>1,ts[source]['read'])
                row=frame[frame.outer_fold.eq(fold)&frame.family.eq(f)].iloc[0]
                for key,value in expected.items():assert abs(row[key]-value)<1e-10,(fold,f,key)
                np.testing.assert_allclose(oof.loc[val,f+'_priority'],p,atol=1e-12)
                count+=1
        assert sorted(seen)==list(range(2494))
        fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
        nums=pd.read_csv(base.PRIOR/'final_inner_splits.csv').inner_fold.to_numpy()
        choices=choose(ctx,fit,nums,inner(ctx,fit,nums,'final'))
        v.check_choices(choices,json.loads((r.OUT/'final_choices.json').read_text()))
        oldctx=base.previous.context();data,_,y,g,title,retrieval,content,_=oldctx
        pack=base.feature_pack(ctx,'final_test',fit,test)
        rebuilt=base.previous.feat.rebuild_prediction(data,y,g,title,retrieval,content,fit,test,pack['recipe'])
        pd.testing.assert_frame_equal(rebuilt,pack['b'],check_dtype=False,atol=1e-11,rtol=1e-11)
        # Compute count transformation independently and compare with production transform.
        refs=[c for c in rebuilt if c.startswith('neighbor_') and c.endswith('_references')]
        normalized=rebuilt.drop(columns=refs).copy()
        for ref in refs:
            support=ref.removesuffix('_references')+'_support'
            normalized[support]=[a/b if b else 0. for a,b in zip(rebuilt[support],rebuilt[ref])]
        pd.testing.assert_frame_equal(normalized,r.transform(rebuilt,'ORD_count_fraction'))
        vs,_=cached(ctx,fit,test,'final_test');reloads=0
        for f in r.NEW:
            x=r.transform(rebuilt,f);raw=[]
            for k in ([2,3,4,5] if f.startswith('ORD') else [None]):
                path=r.OUT/(f'{f}_ge{k}.cbm' if k is not None else f'{f}.cbm')
                m=CatBoostClassifier();m.load_model(str(path));assert m.feature_names_==list(x)
                p=m.predict_proba(x);raw.append(p[:,1] if k is not None else p);reloads+=1
            a=np.column_stack(raw) if f.startswith('ORD') else raw[0]
            z=np.load(r.OUT/f'final_test_{f}.npz');np.testing.assert_allclose(a,z['raw'],atol=1e-12)
            vs[f]=v.view(a,f.startswith('ORD'))
        for f,a in vs.items():
            table=pd.read_csv(r.OUT/f'submission_{f}.csv',dtype={'id':str})
            assert len(table)==596 and table.id.is_unique and table.id.tolist()==ctx['template'].id.tolist()
            mapping=dict(zip(data.id.iloc[test],v.priorities(a,choices[f]['threshold'])))
            expected=np.array([mapping[i] for i in table.id]);assert np.isfinite(table.priority).all() and (table.priority>=0).all()
            np.testing.assert_allclose(table.priority,expected,atol=1e-11)
            order=lambda a:sorted(np.flatnonzero(a>0),key=lambda i:(-a[i],table.id.iloc[i]))
            assert order(table.priority.to_numpy())==order(expected)
        source=choices['selection_procedure']['family']
        assert base.digest(r.OUT/'submission_candidate.csv')==base.digest(r.OUT/f'submission_{source}.csv')
        assert base.digest(base.PRIOR/'submission_candidate.csv')==ctx['newprotocol']['incumbent_sha256']
        result=dict(status='PASSED',outer_metrics_recomputed=count,inner_choices_recomputed=54,
            new_fits_checked=390,final_models_reloaded=reloads,final_features_rebuilt=True,
            independent_count_transform=True,original_labels_preserved=True,incumbent_unchanged=True,
            final_probabilities_priorities_and_rankings_verified=True,uploaded=False)
        (r.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
