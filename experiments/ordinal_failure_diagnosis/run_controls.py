#!/usr/bin/env python3
"""Two fixed interventions to distinguish ordinal flexibility vs neighbor feature effects."""
import json
from pathlib import Path
import shutil
import sys
import time

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'ordinal_preference'))
import run_ordinal as base
om=base.om
FAMILIES=['MC_full','ORD_full','MC_regularized','ORD_regularized','MC_without_neighbors','ORD_without_neighbors']
NEW=['ORD_regularized','ORD_without_neighbors']


def columns(ctx,family):
    return [c for c in ctx['columns'] if not(family.endswith('without_neighbors') and c.startswith('neighbor_'))]


def estimator(family,binary=True):
    strong=family.endswith('regularized')
    return CatBoostClassifier(iterations=300,depth=3 if strong else 4,l2_leaf_reg=32 if strong else 8,
        learning_rate=.05,loss_function='Logloss' if binary else 'MultiClass',
        **({} if binary else {'classes_count':5}),random_seed=base.SEED,thread_count=4,verbose=False,allow_writing_files=False)


def protocol(ctx):
    record=dict(plan_sha256=base.digest(HERE/'PLAN.md'),source_sha256=base.digest(__file__),
        ordinal_protocol_sha256=base.digest(base.OUT/'protocol.json'),claim_protocol_sha256=base.digest(base.CLAIM/'protocol.json'),
        families=FAMILIES,new=NEW,columns={f:columns(ctx,f) for f in FAMILIES},
        seed=base.SEED,cutoffs=base.CUTOFFS.tolist(),reused_labels_sha256=ctx['protocol']['labels_sha256'])
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==record
    else:p.write_text(json.dumps(record,indent=2)+'\n')


def fit_new(ctx,fit,pred,name,family,train=False,save=False):
    p=OUT/f'{name}_{family}.npz'
    if p.exists() and (not save or all((OUT/f'{family}_ge{k}.cbm').exists() for k in [2,3,4,5])):
        z=np.load(p);assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
        return om.ordinal_view(z['raw']),om.ordinal_view(z['train_raw']) if train else None
    pack=base.feature_pack(ctx,name,fit,pred);cols=columns(ctx,family)
    raw=np.zeros((len(pred),4));training=np.zeros((len(fit),4)) if train else None;logs=[]
    for j,k in enumerate([2,3,4,5]):
        start=time.monotonic();model=estimator(family)
        target=(ctx['y'][fit]>=k).astype(int)
        model.fit(pack['a'][cols],target,cat_features=['venue'])
        raw[:,j]=model.predict_proba(pack['b'][cols])[:,1]
        if train:training[:,j]=model.predict_proba(pack['a'][cols])[:,1]
        if save:model.save_model(str(OUT/f'{family}_ge{k}.cbm'))
        logs.append(dict(target=k,positive=int(target.sum()),seconds=time.monotonic()-start))
    arrays=dict(fit=fit,pred=pred,raw=raw)
    if train:arrays['train_raw']=training
    np.savez_compressed(p,**arrays)
    (OUT/f'{name}_{family}.json').write_text(json.dumps(dict(family=family,fit=fit.tolist(),pred=pred.tolist(),columns=cols,models=logs),indent=2)+'\n')
    print(name,family,'seconds',round(sum(r['seconds'] for r in logs),1),flush=True)
    return om.ordinal_view(raw),om.ordinal_view(training) if train else None


def old_views(fit,pred,name,train=False):
    z=np.load(base.PRIOR/(name+'.npz'));o=np.load(base.OUT/(name+'.npz'));c=np.load(base.CLAIM/(name+'_predictions.npz'))
    for a in [z,o,c]:assert np.array_equal(a['fit'],fit) and np.array_equal(a['pred'],pred)
    views=dict(MC_full=om.multiclass_view(z['both_d4']),ORD_full=om.ordinal_view(o['raw']),
        MC_regularized=om.multiclass_view(c['regularized']),MC_without_neighbors=om.multiclass_view(z['content_d4']))
    training={}
    if train:
        training=dict(MC_full=om.multiclass_view(c['train_reference']),ORD_full=om.ordinal_view(o['train_raw']),
                      MC_regularized=om.multiclass_view(c['train_regularized']))
        # Only the full-model train gap is needed for the no-neighbor contrast;
        # cached augmentation did not save training probabilities for content_d4.
    return views,training


def inner(ctx,fit,nums,prefix):
    buffers={f:np.full((len(fit),4 if f.startswith('ORD') else 5),np.nan) for f in FAMILIES}
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask];name=f'{prefix}_inner{k}'
        views,_=old_views(a,b,name)
        for f in NEW:views[f],_=fit_new(ctx,a,b,name,f)
        for f,v in views.items():buffers[f][mask]=v['raw'] if f.startswith('ORD') else v['probability']
    return {f:om.ordinal_view(a) if f.startswith('ORD') else om.multiclass_view(a) for f,a in buffers.items()}


def choose(ctx,fit,nums,views):
    choices={};rows=[]
    for f in FAMILIES:
        candidates=[]
        for cutoff in base.CUTOFFS:
            p=om.priorities(views[f],cutoff)
            h=np.mean([base.ab.evaluate(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]])['score'] for k in range(4)])
            row=dict(family=f,threshold=float(cutoff),inner_score=float(h));candidates.append(row);rows.append(row)
        choices[f]=max(candidates,key=lambda r:r['inner_score']).copy()
    choices['selection_procedure']=max(choices.values(),key=lambda r:r['inner_score']).copy()
    return choices,rows


def outer(ctx):
    records=[];search=[];d=ctx['data'];oof=d[['id','title','venue','Label']].iloc[:2494].copy()
    for fold,fit,val,nums in base.previous.outer_members(d):
        choices,rows=choose(ctx,fit,nums,inner(ctx,fit,nums,f'outer{fold}'))
        (OUT/f'outer{fold}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
        search.extend(dict(outer_fold=fold,**r) for r in rows)
        views,trains=old_views(fit,val,f'outer{fold}_validation',True)
        for f in NEW:views[f],trains[f]=fit_new(ctx,fit,val,f'outer{fold}_validation',f,train=True)
        for family,choice in choices.items():
            f=choice['family'];v=views[f];p=om.priorities(v,choice['threshold'])
            result=base.ab.evaluate(ctx['y'][val],p,d.id.iloc[val])
            from sklearn.metrics import roc_auc_score
            result.update(AUC=float(roc_auc_score(ctx['y'][val]>1,v['read'])),
                gain_MAE=float(np.mean(np.abs(v['gain']-(ctx['y'][val]-1)/4))),
                missed_high=int(((ctx['y'][val]>=4)&(p==0)).sum()),
                train_AUC=float(roc_auc_score(ctx['y'][fit]>1,trains[f]['read'])) if f in trains else np.nan)
            result['AUC_gap']=result['train_AUC']-result['AUC']
            records.append(dict(outer_fold=fold,family=family,source=f,threshold=choice['threshold'],**result))
            oof.loc[val,family+'_priority']=p
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(records).to_csv(OUT/'control_fold_metrics.csv',index=False)
        pd.DataFrame(search).to_csv(OUT/'control_inner_search.csv',index=False)
        oof.to_csv(OUT/'control_oof.csv',index=False)
        print('OUTER',fold,pd.DataFrame(records).query('outer_fold==@fold')[['family','score','AUC','FP','FN']].to_string(index=False),flush=True)


def final(ctx):
    fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
    nums=pd.read_csv(base.PRIOR/'final_inner_splits.csv').inner_fold.to_numpy()
    choices,rows=choose(ctx,fit,nums,inner(ctx,fit,nums,'final'))
    (OUT/'final_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'final_search.csv',index=False)
    pack=base.feature_pack(ctx,'final_test',fit,test)
    old_ordinal=np.load(base.OUT/'final_test.npz')
    raw={};prob={};views={'ORD_full':om.ordinal_view(old_ordinal['raw'])}
    for f in NEW:
        views[f],_=fit_new(ctx,fit,test,'final_test',f,save=True);raw[f]=views[f]['raw']
    for f in ['MC_full','MC_regularized','MC_without_neighbors']:
        model=CatBoostClassifier()
        if f=='MC_full':model.load_model(str(base.PRIOR/'final_classifier.cbm'))
        else:
            model=estimator(f,False);model.fit(pack['a'][columns(ctx,f)],ctx['y'][fit].astype(int)-1,cat_features=['venue'])
            model.save_model(str(OUT/(f+'.cbm')))
        prob[f]=model.predict_proba(pack['b'][columns(ctx,f)]);views[f]=om.multiclass_view(prob[f])
    for f,v in views.items():
        p=om.priorities(v,choices[f]['threshold'])
        result=ctx['template'][['id']].merge(pd.DataFrame(dict(id=ctx['data'].id.iloc[test].to_numpy(),priority=p)),on='id',validate='one_to_one')
        result.to_csv(OUT/f'submission_{f}.csv',index=False)
    source=choices['selection_procedure']['family']
    shutil.copyfile(OUT/f'submission_{source}.csv',OUT/'submission_candidate.csv')
    np.savez_compressed(OUT/'final_predictions.npz',fit=fit,pred=test,**{f:raw[f] if f.startswith('ORD') else prob[f] for f in [*NEW,*prob]})
    print('FINAL',json.dumps(choices),flush=True)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(4):
        ctx=base.context();base.protocol(ctx);protocol(ctx);outer(ctx);final(ctx)
    print('CONTROLS COMPLETE',flush=True)


if __name__=='__main__':main()
