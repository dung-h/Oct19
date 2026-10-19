#!/usr/bin/env python3
"""Fixed-model learning curves with nested subset/reference isolation."""
import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'preference_augmentation'))
import run_augmentation as previous
import features as feat
import audit_content

SUBSET_SEEDS=[20261019,20261020]


def protocol(ctx):
    OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads((previous.OUT/'protocol.json').read_text())
    for name,digest in prior['source_sha256'].items():
        assert feat.digest_file(previous.HERE/name)==digest
    value=dict(plan_sha256=feat.digest_file(HERE/'PLAN.md'),
        source_sha256={p:feat.digest_file(HERE/p) for p in ['run_diagnosis.py','audit_content.py']},
        previous_protocol_sha256=feat.digest_file(previous.OUT/'protocol.json'),
        data_sha256=previous.ab.baseline.input_digest(ctx[0]),subset_seeds=SUBSET_SEEDS,
        catboost=dict(depth=4,iterations=300,learning_rate=.05,l2_leaf_reg=8,seed=feat.SEED),
        ridge=dict(alpha=.01,solver='lsqr',tol=1e-6),cutoffs=dict(catboost_read=.5,ridge_gain=.25),
        fractions=[.25,.5,.75,1.],cues=audit_content.CUES,
        scope='exploratory diagnosis only; no new tuning, submission or preference labels')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==value,'Protocol changed; use a new experiment directory'
    else:path.write_text(json.dumps(value,indent=2)+'\n')


def subsets(ctx):
    data,_,y,groups,*_=ctx
    for fold,fit,val,_ in previous.outer_members(data):
        for seed in SUBSET_SEEDS:
            chunks=[]
            splitter=StratifiedGroupKFold(4,shuffle=True,random_state=seed)
            for _,q in splitter.split(fit,y[fit],groups[fit]):chunks.append(fit[q])
            for quarter in [1,2,3]:
                selected=np.sort(np.concatenate(chunks[:quarter]))
                yield dict(name=f'outer{fold}_seed{seed}_q{quarter}',outer_fold=fold,subset_seed=seed,
                           fraction=quarter/4,fit=selected,val=val)
        yield dict(name=f'outer{fold}_full',outer_fold=fold,subset_seed=0,fraction=1.,fit=fit,val=val)


def priorities(model,raw):
    if model=='ridge':return previous.ab.priority(raw,.25)
    gain=raw@(np.arange(5)/4)
    return np.where((1-raw[:,0]>.5)&(gain>0),10*gain,0.)


def metrics(y,raw,ids,model):
    rating=raw if model=='ridge' else raw@np.arange(1,6)
    read=rating if model=='ridge' else 1-raw[:,0]
    result=previous.ab.evaluate(y,priorities(model,raw),ids)
    result.update(rating_mae=mean_absolute_error(y,rating),rating_rmse=np.sqrt(mean_squared_error(y,rating)),
                  read_auc=roc_auc_score(y>1,read),missed_high=int(((y>=4)&(priorities(model,raw)==0)).sum()))
    if model=='catboost':
        result.update(multiclass_logloss=log_loss(y-1,raw,labels=np.arange(5)),
                      read_logloss=log_loss((y>1).astype(int),np.clip(read,1e-12,1-1e-12),labels=[0,1]))
    return result


def fit_partition(ctx,spec):
    data,_,y,groups,title,retrieval,content,_=ctx
    name=spec['name'];fit=spec['fit'];val=spec['val'];target=OUT/(name+'.npz')
    assert set(groups[fit]).isdisjoint(groups[val])
    if target.exists():
        with np.load(target) as z:
            assert np.array_equal(z['fit'],fit) and np.array_equal(z['val'],val)
            return {k:z[k] for k in ['ridge_train','ridge_validation','catboost_train','catboost_validation']}
    featurepath=OUT/(name+'_features.joblib')
    if spec['fraction']==1.:
        pack=joblib.load(previous.OUT/f"outer{spec['outer_fold']}_validation_features.joblib")
        assert np.array_equal(pack['fit'],fit) and np.array_equal(pack['pred'],val)
        a,b,lineage=[pack[k] for k in ['a','b','lineage']]
        origin='previous verified full-size feature cache'
    elif featurepath.exists():
        pack=joblib.load(featurepath)
        assert np.array_equal(pack['fit'],fit) and np.array_equal(pack['val'],val)
        a,b,lineage=[pack[k] for k in ['a','b','lineage']]
        origin='current subset feature cache'
    else:
        a,b,_,lineage,_=feat.build_features(data,y,groups,title,retrieval,content,fit,val)
        joblib.dump(dict(a=a,b=b,lineage=lineage,fit=fit,val=val),featurepath,compress=3)
        origin='rebuilt from selected subset only'
    (OUT/(name+'_lineage.json')).write_text(json.dumps(dict(fit=fit.tolist(),val=val.tolist(),
        crossfits=lineage,origin=origin)))
    xf,xv,_=feat.ridge_space(data,title,fit,val)
    ridge=Ridge(alpha=.01,solver='lsqr',tol=1e-6).fit(xf,y[fit])
    values=dict(ridge_train=ridge.predict(xf),ridge_validation=ridge.predict(xv))
    cols=feat.columns_for(a,'both')
    model=previous.estimator(dict(depth=4))
    model.fit(a[cols],y[fit].astype(int)-1,cat_features=['venue'])
    values.update(catboost_train=model.predict_proba(a[cols]),catboost_validation=model.predict_proba(b[cols]))
    if spec['fraction']==1.:
        with np.load(previous.OUT/f"outer{spec['outer_fold']}_validation.npz") as z:
            np.testing.assert_allclose(values['ridge_validation'],z['ridge'],atol=1e-10)
            np.testing.assert_allclose(values['catboost_validation'],z['both_d4'],atol=1e-10)
    np.savez_compressed(target,fit=fit,val=val,**values)
    return values


def run(ctx):
    data,_,y,*_=ctx
    rows=[];membership=[];started=time.monotonic()
    for spec in subsets(ctx):
        values=fit_partition(ctx,spec)
        for index in spec['fit']:
            membership.append(dict(name=spec['name'],outer_fold=spec['outer_fold'],subset_seed=spec['subset_seed'],
                                   fraction=spec['fraction'],id=data.id.iloc[index],index=int(index)))
        for model in ['ridge','catboost']:
            for part,ix in [('train',spec['fit']),('validation',spec['val'])]:
                raw=values[model+'_'+part]
                row={k:v for k,v in spec.items() if k not in ['fit','val']}
                row.update(model=model,partition=part,n_fit=len(spec['fit']),n_eval=len(ix),
                           **metrics(y[ix],raw,data.id.iloc[ix],model))
                rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/'learning_curve_metrics.csv',index=False)
        pd.DataFrame(membership).to_csv(OUT/'subset_membership.csv',index=False)
        print(spec['name'],'n_fit',len(spec['fit']),'seconds',round(time.monotonic()-started,1),flush=True)
        current=pd.DataFrame(rows[-4:])
        print(current[['model','partition','score','read_auc','rating_mae']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','curves','all'],nargs='?',default='all')
    args=parser.parse_args()
    with threadpool_limits(4):
        ctx=previous.context();protocol(ctx);audit_content.export_audit(ctx,OUT)
        if args.stage in ['curves','all']:run(ctx)
    print('DIAGNOSIS FITS COMPLETE',flush=True)


if __name__=='__main__':main()
