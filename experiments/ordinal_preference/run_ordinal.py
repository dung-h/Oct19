#!/usr/bin/env python3
"""Fixed-feature cumulative ordinal experiment, grouped nested CV, cached incumbent."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits

import ordinal_math as om

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'preference_augmentation'))
import run_augmentation as previous
ab=previous.ab
PRIOR=previous.OUT
CLAIM=HERE.parent/'claim_stability'/'outputs'
SEED=20261018
CUTOFFS=np.arange(0,.901,.05)
FAMILIES=['multiclass','ordinal']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context():
    data,template=ab.load_clean_data()
    y=np.full(len(data),np.nan);y[:2494]=data.Label.iloc[:2494].astype(int)
    recipe=joblib.load(PRIOR/'final_recipe.joblib')
    assert recipe['key']=='both_d4' and len(recipe['columns'])==277
    return dict(data=data,template=template,y=y,groups=ab.baseline.grouping(data.reset_index(drop=True)),columns=recipe['columns'])


def protocol(ctx):
    value=dict(plan_sha256=digest(HERE/'PLAN.md'),source_sha256={f:digest(HERE/f) for f in ['ordinal_math.py','run_ordinal.py']},
        labels_sha256=hashlib.sha256(ctx['y'][:2494].astype(np.int64).tobytes()).hexdigest(),
        source_zip_sha256=digest(ab.baseline.ROOT/'assignment-data-mining-261.zip'),
        input_sha256=ab.baseline.input_digest(ctx['data']),split_sha256=digest(ab.V2/'nested_splits.csv'),
        final_split_sha256=digest(PRIOR/'final_inner_splits.csv'),prior_protocol_sha256=digest(PRIOR/'protocol.json'),
        feature_sha256={p.name:digest(p) for p in sorted(PRIOR.glob('*_features.joblib'))},
        prior_feature_source_sha256=digest(previous.HERE/'features.py'),
        columns=ctx['columns'],targets=om.THRESHOLDS.tolist(),seed=SEED,
        estimator=dict(iterations=300,depth=4,learning_rate=.05,l2_leaf_reg=8,loss_function='Logloss',thread_count=4),
        projection='equal-weight per-row least-squares PAV decreasing; label-independent; sum-preserving',
        policy='select P(read)>cutoff, rank10*expected gain; cutoff/family by mean4 inner H',
        cutoffs=CUTOFFS.tolist(),score='detailed Evaluation H; original Overview60/40 conflict not externally resolved',
        limitation='Exploratory reuse of outer dataset; four binary ensembles have different total capacity from one multiclass model')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==value,'Protocol/source changed'
    else:path.write_text(json.dumps(value,indent=2)+'\n')
    ctx['protocol']=value


def feature_pack(ctx,name,fit,pred):
    path=PRIOR/(name+'_features.joblib')
    assert digest(path)==ctx['protocol']['feature_sha256'][path.name]
    pack=joblib.load(path)
    assert np.array_equal(pack['fit'],fit) and np.array_equal(pack['pred'],pred)
    assert set(fit)<=set(range(2494)) and not(set(fit)&set(pred))
    if (pred<2494).all():assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][pred])
    assert list(pack['a'][ctx['columns']].columns)==list(pack['b'][ctx['columns']].columns)==ctx['columns']
    return pack


def estimator():
    return CatBoostClassifier(iterations=300,depth=4,learning_rate=.05,l2_leaf_reg=8,loss_function='Logloss',
        random_seed=SEED,thread_count=4,verbose=False,allow_writing_files=False)


def partition(ctx,fit,pred,name,save=False,include_train=False):
    path=OUT/(name+'.npz')
    if path.exists() and (not save or all((OUT/f'ordinal_ge{k}.cbm').exists() for k in om.THRESHOLDS)):
        z=np.load(path)
        assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
        return om.ordinal_view(z['raw']),z['train_raw'] if include_train else None
    pack=feature_pack(ctx,name,fit,pred);x=pack['a'][ctx['columns']];xp=pack['b'][ctx['columns']]
    target=om.targets(ctx['y'][fit]);raw=np.zeros((len(pred),4));train=np.zeros((len(fit),4)) if include_train else None
    logs=[]
    for j,k in enumerate(om.THRESHOLDS):
        start=time.monotonic();model=estimator()
        model.fit(x,target[:,j],cat_features=['venue'])
        assert np.array_equal(model.classes_,[0,1])
        raw[:,j]=model.predict_proba(xp)[:,1]
        if include_train:train[:,j]=model.predict_proba(x)[:,1]
        logs.append(dict(target=int(k),positive=int(target[:,j].sum()),negative=int((target[:,j]==0).sum()),
                         seconds=time.monotonic()-start))
        if save:model.save_model(str(OUT/f'ordinal_ge{k}.cbm'))
        print(name,'target >=',int(k),'seconds',round(logs[-1]['seconds'],2),flush=True)
    view=om.ordinal_view(raw)
    meta=dict(fit=fit.tolist(),pred=pred.tolist(),models=logs,feature_sha256=digest(PRIOR/(name+'_features.joblib')),
        crossing_rows=int((np.diff(raw,axis=1)>0).any(1).sum()),n_pred=len(pred),
        mean_abs_projection=float(np.abs(view['tails']-raw).mean()),
        mean_abs_read_change=float(np.abs(view['read']-raw[:,0]).mean()),
        max_abs_gain_change=float(np.abs(view['gain']-raw.mean(1)).max()))
    values=dict(fit=fit,pred=pred,raw=raw,tails=view['tails'],probability=view['probability'])
    if include_train:values['train_raw']=train
    np.savez_compressed(path,**values)
    (OUT/(name+'_training.json')).write_text(json.dumps(meta,indent=2)+'\n')
    if save:
        joblib.dump(dict(columns=ctx['columns'],feature_recipe=pack['recipe'],fit=fit,pred=pred,
                         targets=om.THRESHOLDS,projection='PAV decreasing equal weight'),OUT/'final_recipe.joblib',compress=3)
    return view,train


def incumbent(fit,pred,name,training=False):
    z=np.load(PRIOR/(name+'.npz'))
    assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
    view=om.multiclass_view(z['both_d4']);train=None
    if training:
        old=np.load(CLAIM/(name+'_predictions.npz'))
        assert np.array_equal(old['fit'],fit) and np.array_equal(old['pred'],pred)
        np.testing.assert_allclose(old['reference'],view['probability'],atol=1e-14)
        train=om.multiclass_view(old['train_reference'])
    return view,train


def inner(ctx,fit,nums,prefix):
    values={f:np.full((len(fit),5 if f=='multiclass' else 4),np.nan) for f in FAMILIES}
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask]
        view,_=partition(ctx,a,b,f'{prefix}_inner{k}')
        values['ordinal'][mask]=view['raw']
        view,_=incumbent(a,b,f'{prefix}_inner{k}')
        values['multiclass'][mask]=view['probability']
    return {f:om.multiclass_view(v) if f=='multiclass' else om.ordinal_view(v) for f,v in values.items()}


def choose(ctx,fit,nums,views):
    rows=[];choices={}
    for family in FAMILIES:
        candidates=[]
        for cutoff in CUTOFFS:
            p=om.priorities(views[family],cutoff)
            h=np.mean([ab.evaluate(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]])['score'] for k in range(4)])
            row=dict(family=family,threshold=float(cutoff),inner_score=float(h))
            candidates.append(row);rows.append(row)
        choices[family]=max(candidates,key=lambda r:r['inner_score']).copy()
    choices['selection_procedure']=max(choices.values(),key=lambda r:r['inner_score']).copy()
    return choices,rows


def metrics(ctx,fit,pred,view,train,cutoff):
    y=ctx['y'][pred];p=om.priorities(view,cutoff)
    result=ab.evaluate(y,p,ctx['data'].id.iloc[pred])
    result.update(AUC=float(roc_auc_score(y>1,view['read'])),
        logloss=float(log_loss(y-1,view['probability'],labels=np.arange(5))),
        gain_MAE=float(np.abs(view['gain']-(y-1)/4).mean()),
        missed_high=int(((y>=4)&(p==0)).sum()),
        train_AUC=float(roc_auc_score(ctx['y'][fit]>1,train['read'])))
    result['AUC_gap']=result['train_AUC']-result['AUC']
    return result,p


def run_outer(ctx):
    data=ctx['data'];oof=data[['id','title','venue','Label']].iloc[:2494].copy()
    records=[];search=[];tailmetrics=[]
    for fold,fit,val,nums in previous.outer_members(data):
        choices,rows=choose(ctx,fit,nums,inner(ctx,fit,nums,f'outer{fold}'))
        search.extend(dict(outer_fold=fold,**r) for r in rows)
        (OUT/f'outer{fold}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
        v,t=partition(ctx,fit,val,f'outer{fold}_validation',include_train=True)
        views={'ordinal':v};trains={'ordinal':om.ordinal_view(t)}
        views['multiclass'],trains['multiclass']=incumbent(fit,val,f'outer{fold}_validation',True)
        for family,choice in choices.items():
            source=choice['family'];view=views[source]
            result,p=metrics(ctx,fit,val,view,trains[source],choice['threshold'])
            records.append(dict(outer_fold=fold,family=family,selected_family=source,threshold=choice['threshold'],
                                inner_score=choice['inner_score'],**result))
            oof.loc[val,family+'_priority']=p
            for k in range(5):oof.loc[val,family+f'_p{k+1}']=view['probability'][:,k]
        for family,view in views.items():
            target=om.targets(ctx['y'][val])
            for j,k in enumerate(om.THRESHOLDS):
                for stage in ['raw','tails']:
                    prob=view[stage][:,j]
                    tailmetrics.append(dict(outer_fold=fold,family=family,target=int(k),stage=stage,
                        AUC=float(roc_auc_score(target[:,j],prob)),Brier=float(np.mean((prob-target[:,j])**2))))
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(records).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(search).to_csv(OUT/'inner_search.csv',index=False)
        pd.DataFrame(tailmetrics).to_csv(OUT/'tail_metrics.csv',index=False)
        oof.to_csv(OUT/'oof_predictions.csv',index=False)
        print('OUTER',fold,pd.DataFrame(records).query('outer_fold==@fold')[['family','score','FP','FN','AUC']].to_string(index=False),flush=True)


def run_final(ctx):
    fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
    membership=pd.read_csv(PRIOR/'final_inner_splits.csv',dtype={'id':str})
    assert membership.id.tolist()==ctx['data'].id.iloc[:2494].tolist()
    nums=membership.inner_fold.to_numpy();membership.to_csv(OUT/'final_inner_splits.csv',index=False)
    choices,rows=choose(ctx,fit,nums,inner(ctx,fit,nums,'final'))
    (OUT/'final_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'final_search.csv',index=False)
    view,_=partition(ctx,fit,test,'final_test',save=True)
    priority=om.priorities(view,choices['ordinal']['threshold'])
    details=ctx['data'][['id','title','venue']].iloc[test].copy();details['priority']=priority
    for j in range(4):details[f'raw_ge{j+2}']=view['raw'][:,j];details[f'q_ge{j+2}']=view['tails'][:,j]
    for j in range(5):details[f'p{j+1}']=view['probability'][:,j]
    details.to_csv(OUT/'test_predictions.csv',index=False)
    submission=ctx['template'][['id']].merge(details[['id','priority']],on='id',validate='one_to_one')
    submission.to_csv(OUT/'submission_ordinal.csv',index=False)
    if choices['selection_procedure']['family']=='multiclass':
        old=json.loads((PRIOR/'final_choice.json').read_text())
        assert old['key']=='both_d4' and abs(old['threshold']-choices['multiclass']['threshold'])<1e-12
        shutil.copyfile(PRIOR/'submission_candidate.csv',OUT/'submission_candidate.csv')
    else:submission.to_csv(OUT/'submission_candidate.csv',index=False)
    print('FINAL',json.dumps(choices),'ordinal selected',int((priority>0).sum()),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','outer','final','all'],default='all',nargs='?')
    args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True);started=time.monotonic()
    with threadpool_limits(4):
        ctx=context();protocol(ctx)
        if args.stage in ['outer','all']:run_outer(ctx)
        if args.stage in ['final','all']:run_final(ctx)
    print('DONE',args.stage,'seconds',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':main()
