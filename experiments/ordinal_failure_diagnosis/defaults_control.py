#!/usr/bin/env python3
"""Single explicit default-matching intervention, separate from the primary controls."""
import json
from pathlib import Path
import shutil

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_controls as diag
base=diag.base
om=base.om
OUT=diag.OUT/'defaults'


def fit(ctx,train,pred,name,save=False,include_train=False):
    path=OUT/(name+'.npz')
    if path.exists():
        z=np.load(path);assert np.array_equal(z['fit'],train) and np.array_equal(z['pred'],pred)
        return om.ordinal_view(z['raw']),om.ordinal_view(z['train_raw']) if include_train else None
    pack=base.feature_pack(ctx,name,train,pred);cols=ctx['columns']
    raw=np.zeros((len(pred),4));tr=np.zeros((len(train),4)) if include_train else None
    params=[]
    for j,k in enumerate([2,3,4,5]):
        model=CatBoostClassifier(iterations=300,depth=4,l2_leaf_reg=8,learning_rate=.05,
            loss_function='Logloss',random_seed=base.SEED,thread_count=4,verbose=False,
            allow_writing_files=False,bootstrap_type='Bayesian',bagging_temperature=1,leaf_estimation_iterations=1)
        model.fit(pack['a'][cols],(ctx['y'][train]>=k).astype(int),cat_features=['venue'])
        raw[:,j]=model.predict_proba(pack['b'][cols])[:,1]
        if include_train:tr[:,j]=model.predict_proba(pack['a'][cols])[:,1]
        if save:model.save_model(str(OUT/f'ordinal_ge{k}.cbm'))
        params.append(model.get_all_params())
    arrays=dict(fit=train,pred=pred,raw=raw)
    if include_train:arrays['train_raw']=tr
    np.savez_compressed(path,**arrays)
    (OUT/(name+'_params.json')).write_text(json.dumps(params,indent=2)+'\n')
    print(name,'default-matched complete',flush=True)
    return om.ordinal_view(raw),om.ordinal_view(tr) if include_train else None


def inner(ctx,train,nums,prefix):
    a=np.full((len(train),4),np.nan);b=np.full((len(train),5),np.nan)
    for k in range(4):
        mask=nums==k;x=train[~mask];q=train[mask]
        v,_=fit(ctx,x,q,f'{prefix}_inner{k}');a[mask]=v['raw']
        v,_=base.incumbent(x,q,f'{prefix}_inner{k}');b[mask]=v['probability']
    return dict(multiclass=om.multiclass_view(b),ordinal=om.ordinal_view(a))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(4):
        ctx=base.context();base.protocol(ctx)
        protocol=dict(plan_sha256=base.digest(diag.HERE/'DEFAULTS_PLAN.md'),source_sha256=base.digest(__file__),
            parent_ordinal_protocol_sha256=base.digest(base.OUT/'protocol.json'),
            changed_parameters=dict(bootstrap_type='Bayesian',bagging_temperature=1,leaf_estimation_iterations=1))
        path=OUT/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol
        else:path.write_text(json.dumps(protocol,indent=2)+'\n')
        records=[];data=ctx['data'];oof=data[['id','title','venue','Label']].iloc[:2494].copy()
        for fold,train,val,nums in base.previous.outer_members(data):
            choices,rows=base.choose(ctx,train,nums,inner(ctx,train,nums,f'outer{fold}'))
            (OUT/f'outer{fold}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
            pd.DataFrame(rows).to_csv(OUT/f'outer{fold}_search.csv',index=False)
            v,t=fit(ctx,train,val,f'outer{fold}_validation',include_train=True)
            inc,ti=base.incumbent(train,val,f'outer{fold}_validation',True)
            views=dict(ordinal=v,multiclass=inc);trains=dict(ordinal=t,multiclass=ti)
            for f,c in choices.items():
                actual,p=base.metrics(ctx,train,val,views[c['family']],trains[c['family']],c['threshold'])
                records.append(dict(outer_fold=fold,family=f,source=c['family'],threshold=c['threshold'],**actual))
                oof.loc[val,f+'_priority']=p
            oof.loc[val,'outer_fold']=fold
            pd.DataFrame(records).to_csv(OUT/'fold_metrics.csv',index=False);oof.to_csv(OUT/'oof.csv',index=False)
            print('OUTER',fold,pd.DataFrame(records).query('outer_fold==@fold')[['family','score','AUC','train_AUC']].to_string(index=False),flush=True)
        train=np.arange(2494);test=np.arange(2494,len(data))
        nums=pd.read_csv(base.PRIOR/'final_inner_splits.csv').inner_fold.to_numpy()
        choices,rows=base.choose(ctx,train,nums,inner(ctx,train,nums,'final'))
        (OUT/'final_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
        pd.DataFrame(rows).to_csv(OUT/'final_search.csv',index=False)
        v,_=fit(ctx,train,test,'final_test',save=True)
        p=om.priorities(v,choices['ordinal']['threshold'])
        result=ctx['template'][['id']].merge(pd.DataFrame(dict(id=data.id.iloc[test].to_numpy(),priority=p)),on='id',validate='one_to_one')
        result.to_csv(OUT/'submission_ordinal.csv',index=False)
        shutil.copyfile(base.PRIOR/'submission_candidate.csv' if choices['selection_procedure']['family']=='multiclass' else OUT/'submission_ordinal.csv',OUT/'submission_candidate.csv')
    print('DEFAULT CONTROL COMPLETE',flush=True)


if __name__=='__main__':main()
