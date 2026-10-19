#!/usr/bin/env python3
"""Finite combined-fix and sample-count representation experiment."""
import json
from pathlib import Path
import shutil
import sys
import time

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'ordinal_failure_diagnosis'))
import run_controls as diag
import defaults_control as dc
base=diag.base
om=base.om
CONTROLS=['MC_full','ORD_defaults_full','MC_without_neighbors','ORD_without_neighbors']
NEW=['ORD_combined','ORD_combined_regularized','ORD_count_fraction','MC_count_fraction']
FAMILIES=CONTROLS+NEW


def transform(x,family):
    if family in ['MC_without_neighbors','ORD_without_neighbors','ORD_combined','ORD_combined_regularized']:
        return x[[c for c in x if not c.startswith('neighbor_')]].copy()
    if family.endswith('count_fraction'):
        z=x.copy()
        refs=[c for c in z if c.startswith('neighbor_') and c.endswith('_references')]
        assert len(refs)==6
        for r in refs:
            s=r.removesuffix('_references')+'_support'
            den=z[r].to_numpy(float);num=z[s].to_numpy(float)
            assert (den>=0).all() and (num>=0).all() and (num<=den).all()
            z[s]=np.divide(num,den,out=np.zeros(len(z)),where=den>0)
        return z.drop(columns=refs)
    return x.copy()


def estimator(family):
    binary=family.startswith('ORD');strong=family.endswith('regularized')
    return CatBoostClassifier(iterations=300,depth=3 if strong else 4,l2_leaf_reg=32 if strong else 8,
        learning_rate=.05,loss_function='Logloss' if binary else 'MultiClass',
        **({} if binary else {'classes_count':5}),random_seed=base.SEED,thread_count=4,
        verbose=False,allow_writing_files=False,bootstrap_type='Bayesian',bagging_temperature=1,
        leaf_estimation_iterations=1)


def protocol(ctx):
    r=dict(plan_sha256=base.digest(HERE/'PLAN.md'),source_sha256=base.digest(__file__),
        parent_protocol_sha256=base.digest(base.OUT/'protocol.json'),
        defaults_protocol_sha256=base.digest(dc.OUT/'protocol.json'),
        diagnosis_protocol_sha256=base.digest(diag.OUT/'protocol.json'),
        families=FAMILIES,new=NEW,seed=base.SEED,cutoffs=base.CUTOFFS.tolist(),
        incumbent_sha256=base.digest(base.PRIOR/'submission_candidate.csv'),
        columns={f:list(transform(pd.DataFrame(columns=ctx['columns']),f).columns) for f in FAMILIES
                 if not f.endswith('count_fraction')})
    # Empty frames have no useful numeric dtypes; obtain transformed names without division.
    for f in ['ORD_count_fraction','MC_count_fraction']:
        r['columns'][f]=[c for c in ctx['columns'] if not(c.startswith('neighbor_') and c.endswith('_references'))]
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==r
    else:p.write_text(json.dumps(r,indent=2)+'\n')
    return r


def fit_new(ctx,pack,name,family,training=False,save=False):
    fit,pred=pack['fit'],pack['pred'];path=OUT/f'{name}_{family}.npz'
    if path.exists():
        z=np.load(path);assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
        return z['raw'],z['train_raw'] if training else None
    x=transform(pack['a'][ctx['columns']],family);xp=transform(pack['b'][ctx['columns']],family)
    assert list(x)==list(xp)
    binary=family.startswith('ORD');width=4 if binary else 5
    raw=np.zeros((len(pred),width));tr=np.zeros((len(fit),width)) if training else None
    logs=[]
    for j,k in enumerate([2,3,4,5] if binary else [None]):
        start=time.monotonic();m=estimator(family)
        y=(ctx['y'][fit]>=k).astype(int) if binary else ctx['y'][fit].astype(int)-1
        m.fit(x,y,cat_features=['venue']);p=m.predict_proba(xp)
        if binary:raw[:,j]=p[:,1]
        else:raw[:]=p
        if training:
            p=m.predict_proba(x)
            if binary:tr[:,j]=p[:,1]
            else:tr[:]=p
        if save:m.save_model(str(OUT/(f'{family}_ge{k}.cbm' if binary else f'{family}.cbm')))
        logs.append(dict(target=k,seconds=time.monotonic()-start,params=m.get_all_params()))
    values=dict(fit=fit,pred=pred,raw=raw)
    if training:values['train_raw']=tr
    np.savez_compressed(path,**values)
    (OUT/f'{name}_{family}.json').write_text(json.dumps(dict(columns=list(x),models=logs,
        feature_sha256=base.digest(base.PRIOR/f'{name}_features.joblib')),indent=2)+'\n')
    print(name,family,round(sum(a['seconds'] for a in logs),1),'seconds',flush=True)
    return raw,tr


def view(raw,family):return om.ordinal_view(raw) if family.startswith('ORD') else om.multiclass_view(raw)


def collect(ctx,fit,pred,name,training=False,save=False):
    if name=='final_test':
        d=np.load(diag.OUT/'final_predictions.npz');old=np.load(diag.OUT/f'{name}_ORD_without_neighbors.npz')
        arrays={'MC_full':d['MC_full'],'MC_without_neighbors':d['MC_without_neighbors'],
                'ORD_without_neighbors':old['raw']};trains={}
    else:
        vs,ts=diag.old_views(fit,pred,name,training)
        arrays={f:vs[f]['raw'] if f.startswith('ORD') else vs[f]['probability']
                for f in ['MC_full','MC_without_neighbors']}
        o=np.load(diag.OUT/f'{name}_ORD_without_neighbors.npz')
        assert np.array_equal(o['fit'],fit) and np.array_equal(o['pred'],pred)
        arrays['ORD_without_neighbors']=o['raw']
        trains={f:ts[f]['probability'] for f in ['MC_full'] if training}
        if training:trains['ORD_without_neighbors']=o['train_raw']
    d=np.load(dc.OUT/f'{name}.npz');assert np.array_equal(d['fit'],fit) and np.array_equal(d['pred'],pred)
    arrays['ORD_defaults_full']=d['raw']
    if training:trains['ORD_defaults_full']=d['train_raw']
    pack=base.feature_pack(ctx,name,fit,pred)
    for f in NEW:
        arrays[f],tr=fit_new(ctx,pack,name,f,training,save)
        if training:trains[f]=tr
    return {f:view(arrays[f],f) for f in FAMILIES},{f:view(v,f) for f,v in trains.items()}


def inner(ctx,fit,nums,prefix):
    buffers={f:np.full((len(fit),4 if f.startswith('ORD') else 5),np.nan) for f in FAMILIES}
    for k in range(4):
        mask=nums==k;vs,_=collect(ctx,fit[~mask],fit[mask],f'{prefix}_inner{k}')
        for f,v in vs.items():buffers[f][mask]=v['raw'] if f.startswith('ORD') else v['probability']
    return {f:view(v,f) for f,v in buffers.items()}


def choose(ctx,fit,nums,views):
    rows=[];choices={}
    for f in FAMILIES:
        candidates=[]
        for t in base.CUTOFFS:
            p=om.priorities(views[f],t)
            h=np.mean([base.ab.evaluate(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]])['score'] for k in range(4)])
            r=dict(family=f,threshold=float(t),inner_score=float(h));rows.append(r);candidates.append(r)
        choices[f]=max(candidates,key=lambda r:r['inner_score']).copy()
    choices['selection_procedure']=max(choices.values(),key=lambda r:r['inner_score']).copy()
    return choices,rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(4):
        ctx=base.context();base.protocol(ctx);protocol(ctx);records=[];search=[]
        oof=ctx['data'][['id','title','venue','Label']].iloc[:2494].copy()
        for fold,fit,val,nums in base.previous.outer_members(ctx['data']):
            choices,rows=choose(ctx,fit,nums,inner(ctx,fit,nums,f'outer{fold}'))
            (OUT/f'outer{fold}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
            search.extend(dict(outer_fold=fold,**r) for r in rows)
            vs,ts=collect(ctx,fit,val,f'outer{fold}_validation',training=True)
            for f,c in choices.items():
                source=c['family'];v=vs[source];p=om.priorities(v,c['threshold'])
                m=base.ab.evaluate(ctx['y'][val],p,ctx['data'].id.iloc[val])
                m.update(AUC=float(roc_auc_score(ctx['y'][val]>1,v['read'])),
                    train_AUC=float(roc_auc_score(ctx['y'][fit]>1,ts[source]['read'])) if source in ts else np.nan,
                    gain_MAE=float(np.abs(v['gain']-(ctx['y'][val]-1)/4).mean()),
                    missed_high=int(((ctx['y'][val]>=4)&(p==0)).sum()))
                m['AUC_gap']=m['train_AUC']-m['AUC']
                records.append(dict(outer_fold=fold,family=f,source=source,threshold=c['threshold'],**m))
                oof.loc[val,f+'_priority']=p
            oof.loc[val,'outer_fold']=fold
            pd.DataFrame(records).to_csv(OUT/'fold_metrics.csv',index=False)
            pd.DataFrame(search).to_csv(OUT/'inner_search.csv',index=False)
            oof.to_csv(OUT/'oof.csv',index=False)
            print('OUTER',fold,pd.DataFrame(records).query('outer_fold==@fold')[['family','score','AUC','FP','FN']].to_string(index=False),flush=True)
        fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
        nums=pd.read_csv(base.PRIOR/'final_inner_splits.csv').inner_fold.to_numpy()
        choices,rows=choose(ctx,fit,nums,inner(ctx,fit,nums,'final'))
        (OUT/'final_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
        pd.DataFrame(rows).to_csv(OUT/'final_search.csv',index=False)
        vs,_=collect(ctx,fit,test,'final_test',save=True)
        for f,v in vs.items():
            p=om.priorities(v,choices[f]['threshold'])
            result=ctx['template'][['id']].merge(pd.DataFrame(dict(id=ctx['data'].id.iloc[test].to_numpy(),priority=p)),on='id',validate='one_to_one')
            result.to_csv(OUT/f'submission_{f}.csv',index=False)
        source=choices['selection_procedure']['family']
        shutil.copyfile(OUT/f'submission_{source}.csv',OUT/'submission_candidate.csv')
        pack=base.feature_pack(ctx,'final_test',fit,test)
        joblib.dump(dict(feature_recipe=pack['recipe'],columns=ctx['columns'],fit=fit,pred=test,
            choices=choices,transform_source_sha256=base.digest(__file__)),OUT/'final_recipe.joblib',compress=3)
        summary=pd.DataFrame(records).groupby('family',sort=False).mean(numeric_only=True)
        summary.to_csv(OUT/'summary.csv')
        print(summary[['score','AUC','train_AUC','FP','FN']].to_string(),flush=True)
        print('FINAL',json.dumps(choices),flush=True)


if __name__=='__main__':main()
