#!/usr/bin/env python3
"""Locked feature ablations with inherited grouped folds and control replay."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'preference_augmentation'))
import run_augmentation as prior
import repair_features as feat

ARMS=feat.ARMS
THRESHOLDS=np.arange(0,.901,.05)


def context():
    data,template,y,groups,title,retrieval,content,_=prior.context()
    fixed,evidence,changes,rules=feat.fixed_content(data,content,prior.feat.RULES)
    return dict(data=data,template=template,y=y,groups=groups,title=title,retrieval=retrieval,
                old_content=content,fixed_content=fixed,evidence=evidence,changes=changes,rules=rules)


def jobs(ctx):
    result=[]
    for fold,fit,val,nums in prior.outer_members(ctx['data']):
        for k in range(4):
            mask=nums==k
            result.append((f'outer{fold}_inner{k}',fit[~mask],fit[mask]))
        result.append((f'outer{fold}_validation',fit,val))
    fit=np.arange(2494)
    membership=pd.read_csv(prior.OUT/'final_inner_splits.csv',dtype={'id':str})
    assert membership.id.tolist()==ctx['data'].id.iloc[fit].tolist()
    nums=membership.inner_fold.to_numpy()
    for k in range(4):
        result.append((f'final_inner{k}',fit[nums!=k],fit[nums==k]))
    result.append(('final_test',fit,np.arange(2494,len(ctx['data']))))
    return result


def protocol(ctx,create=False):
    sources=[HERE/'PLAN.md',HERE/'repair_features.py',HERE/'run_repairs.py',
        Path(prior.__file__),Path(prior.feat.__file__),Path(prior.ab.__file__),
        Path(prior.ab.diag.__file__),Path(prior.ab.baseline.__file__),
        HERE.parent/'specter2_v2/prepare_data.py']
    parents={}
    for name,_,_ in jobs(ctx):
        for suffix in ['_features.joblib']:
            path=prior.OUT/(name+suffix)
            parents[path.name]=feat.digest(path)
        path=prior.OUT/('test_prediction_details.csv' if name=='final_test' else name+'.npz')
        parents[path.name]=feat.digest(path)
    rec=dict(sources={str(p.relative_to(HERE.parents[1])):feat.digest(p) for p in sources},
        full_input_sha256=feat.input_digest(ctx['data']),
        source_zip_sha256=feat.digest(prior.ab.baseline.ROOT/'assignment-data-mining-261.zip'),
        clean_abstract_sha256=feat.digest(HERE.parent/'specter2_v2/outputs/abstracts_cleaned.csv'),
        embedding_sha256={str(p.relative_to(HERE.parent)):feat.digest(p) for p in [
            prior.ab.OUT/'field_embeddings.npz',HERE.parent/'diagnose_bottleneck/outputs/embeddings_retrieval.npz']},
        parent_artifacts_sha256=parents,
        outer_split_sha256=feat.digest(prior.ab.V2/'nested_splits.csv'),
        final_split_sha256=feat.digest(prior.OUT/'final_inner_splits.csv'),
        incumbent_sha256=feat.digest(prior.OUT/'submission_candidate.csv'),
        arms=ARMS,thresholds=THRESHOLDS.tolist(),seed=feat.SEED,
        model=dict(iterations=300,depth=4,learning_rate=.05,l2_leaf_reg=8,loss='MultiClass'),
        rules=ctx['rules'],evaluation='detailed H; exploratory on previously inspected grouped folds')
    path=OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==rec,'Source/input/dependency changed after protocol lock'
    else:
        assert create,'Run prepare to lock the protocol first'
        path.write_text(json.dumps(rec,indent=2)+'\n')
    return rec


def parent_pack(name,fit,pred):
    pack=joblib.load(prior.OUT/(name+'_features.joblib'))
    assert np.array_equal(pack['fit'],fit) and np.array_equal(pack['pred'],pred)
    return pack


def parent_prediction(ctx,name,fit,pred):
    if name=='final_test':
        table=pd.read_csv(prior.OUT/'test_prediction_details.csv',dtype={'id':str})
        assert table.id.tolist()==ctx['data'].id.iloc[pred].tolist()
        return table[[f'p{k}' for k in range(1,6)]].to_numpy()
    z=np.load(prior.OUT/(name+'.npz'))
    assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
    return z['both_d4']


def preflight(ctx):
    name,fit,pred=jobs(ctx)[0]
    old=parent_pack(name,fit,pred)
    started=time.monotonic()
    rich=feat.build_rich(ctx,old,fit,pred,prior.feat)
    shape={arm:list(feat.frame(ctx,old,rich,arm,'a').shape) for arm in ARMS}
    for arm in ARMS:
        a=feat.frame(ctx,old,rich,arm,'a')
        for view in range(3 if arm in ['matched','combined'] else 1):
            b=feat.frame(ctx,old,rich,arm,'b',view)
            assert a.columns.tolist()==b.columns.tolist()
    model=prior.estimator(dict(depth=4))
    model.fit(old['a'],ctx['y'][fit].astype(int)-1,cat_features=['venue'])
    values=model.predict_proba(old['b'])
    archived=parent_prediction(ctx,name,fit,pred)
    np.testing.assert_allclose(values,archived,atol=1e-10,rtol=1e-10)
    result=dict(status='PASSED',partition=name,shape=shape,
        baseline_max_probability_error=float(np.max(np.abs(values-archived))),
        seconds=time.monotonic()-started,feature_audit=rich['audit'],
        note='Control replay and feature integrity only; no repaired outer score inspected')
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def prepare(ctx):
    OUT.mkdir(exist_ok=True,parents=True)
    protocol(ctx,create=True)
    ctx['changes'].to_csv(OUT/'rule_changes.csv',index=False)
    ctx['evidence'].to_csv(OUT/'rule_evidence.csv',index=False)
    keys=ctx['data'][['split','id']].reset_index(drop=True)
    pd.concat([keys,ctx['fixed_content']],axis=1).to_csv(OUT/'repaired_content.csv',index=False)
    print('PROTOCOL LOCKED',len(jobs(ctx)),'partitions; rule changes',len(ctx['changes']),flush=True)


def fit_partition(ctx,name,fit,pred):
    target=OUT/(name+'_predictions.npz')
    protocol_hash=feat.digest(OUT/'protocol.json')
    if target.exists():
        z=np.load(target)
        assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
        assert str(z['protocol_sha256'])==protocol_hash
        print('CACHED',name,flush=True)
        return
    old=parent_pack(name,fit,pred)
    rich_path=OUT/(name+'_rich.joblib')
    if rich_path.exists():
        rich=joblib.load(rich_path)
        assert rich['protocol_sha256']==protocol_hash
    else:
        started=time.monotonic()
        rich=feat.build_rich(ctx,old,fit,pred,prior.feat)
        rich['protocol_sha256']=protocol_hash
        joblib.dump(rich,rich_path,compress=3)
        print(name,'FEATURES',round(time.monotonic()-started,1),'seconds',flush=True)
    predictions={};training={};logs=[]
    reference_model=None
    for arm in ARMS:
        started=time.monotonic()
        a=feat.frame(ctx,old,rich,arm,'a')
        if arm=='matched':
            model=reference_model
        else:
            model=prior.estimator(dict(depth=4))
            model.fit(a,ctx['y'][fit].astype(int)-1,cat_features=['venue'])
            assert np.array_equal(model.classes_,np.arange(5))
        if arm=='reference':
            reference_model=model
        views=[feat.frame(ctx,old,rich,arm,'b',k) for k in range(3 if arm in ['matched','combined'] else 1)]
        assert all(a.columns.tolist()==b.columns.tolist() for b in views)
        values=np.mean([model.predict_proba(b) for b in views],axis=0)
        assert np.isfinite(values).all() and np.allclose(values.sum(axis=1),1,atol=1e-7)
        if arm=='reference':
            control=parent_prediction(ctx,name,fit,pred)
            np.testing.assert_allclose(values,control,atol=1e-10,rtol=1e-10)
        predictions[arm]=values
        training[arm]=model.predict_proba(a)
        if name=='final_test' and arm!='matched':
            model.save_model(str(OUT/f'final_{arm}.cbm'))
            pd.DataFrame(dict(feature=a.columns,importance=model.feature_importances_)).to_csv(OUT/f'final_{arm}_importance.csv',index=False)
        logs.append(dict(arm=arm,n_features=len(a.columns),seconds=time.monotonic()-started))
        print(name,arm,'fit/predict',round(logs[-1]['seconds'],1),'seconds',flush=True)
    np.savez_compressed(target,fit=fit,pred=pred,protocol_sha256=protocol_hash,
        **predictions,**{'train_'+k:v for k,v in training.items()})
    (OUT/(name+'_audit.json')).write_text(json.dumps(rich['audit'],indent=2)+'\n')
    (OUT/(name+'_training.json')).write_text(json.dumps(logs,indent=2)+'\n')
    print('COMPLETE',name,flush=True)


def inner(ctx,fit,nums,prefix):
    values={arm:np.full((len(fit),5),np.nan) for arm in ARMS}
    for k in range(4):
        z=np.load(OUT/(f'{prefix}_inner{k}_predictions.npz'))
        mask=nums==k
        assert np.array_equal(z['fit'],fit[~mask]) and np.array_equal(z['pred'],fit[mask])
        for arm in ARMS:
            values[arm][mask]=z[arm]
    assert all(np.isfinite(v).all() for v in values.values())
    return values


def priority(values,cutoff):
    gain=values@(np.arange(5)/4)
    return np.where((1-values[:,0]>cutoff)&(gain>0),10*gain,0.)


def choose(ctx,fit,nums,values):
    records=[];choices={}
    for arm in ARMS:
        candidates=[]
        for cutoff in THRESHOLDS:
            p=priority(values[arm],cutoff)
            scores=[prior.ab.evaluate(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]])['score'] for k in range(4)]
            row=dict(arm=arm,threshold=float(cutoff),inner_score=float(np.mean(scores)))
            candidates.append(row);records.append(row)
        choices[arm]=max(candidates,key=lambda v:v['inner_score']).copy()
    choices['selection_procedure']=max(choices.values(),key=lambda v:v['inner_score']).copy()
    return choices,records


def score(ctx):
    rows=[];search=[]
    oof=ctx['data'][['id','title','venue','year','Label']].iloc[:2494].copy()
    for fold,fit,pred,nums in prior.outer_members(ctx['data']):
        choices,records=choose(ctx,fit,nums,inner(ctx,fit,nums,f'outer{fold}'))
        (OUT/f'outer{fold}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
        search.extend(dict(outer_fold=fold,**r) for r in records)
        z=np.load(OUT/f'outer{fold}_validation_predictions.npz')
        for policy,choice in choices.items():
            arm=choice['arm'];values=z[arm];p=priority(values,choice['threshold'])
            result=prior.ab.evaluate(ctx['y'][pred],p,ctx['data'].id.iloc[pred])
            result.update(AUC=float(roc_auc_score(ctx['y'][pred]>1,1-values[:,0])),
                train_AUC=float(roc_auc_score(ctx['y'][fit]>1,1-z['train_'+arm][:,0])),
                gain_MAE=float(np.mean(np.abs(values@(np.arange(5)/4)-(ctx['y'][pred]-1)/4))),
                missed_high=int(((ctx['y'][pred]>=4)&(p==0)).sum()))
            rows.append(dict(outer_fold=fold,policy=policy,selected_arm=arm,threshold=choice['threshold'],**result))
            oof.loc[pred,policy+'_priority']=p
        oof.loc[pred,'outer_fold']=fold
        print('OUTER',fold,pd.DataFrame(rows).query('outer_fold==@fold')[['policy','score','FP','FN','missed_high']].to_string(index=False),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.DataFrame(search).to_csv(OUT/'inner_search.csv',index=False)
    oof.to_csv(OUT/'oof.csv',index=False)
    fit=np.arange(2494);pred=np.arange(2494,len(ctx['data']))
    membership=pd.read_csv(prior.OUT/'final_inner_splits.csv',dtype={'id':str})
    choices,records=choose(ctx,fit,membership.inner_fold.to_numpy(),inner(ctx,fit,membership.inner_fold.to_numpy(),'final'))
    (OUT/'final_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    pd.DataFrame(records).to_csv(OUT/'final_search.csv',index=False)
    choice=choices['selection_procedure'];arm=choice['arm']
    z=np.load(OUT/'final_test_predictions.npz')
    for name,item in choices.items():
        p=priority(z[item['arm']],item['threshold'])
        table=ctx['template'][['id']].merge(pd.DataFrame(dict(id=ctx['data'].id.iloc[pred].to_numpy(),priority=p)),on='id',validate='one_to_one')
        table.to_csv(OUT/f'submission_{name}.csv',index=False)
    shutil.copyfile(OUT/'submission_selection_procedure.csv',OUT/'submission_candidate.csv')
    if arm=='reference':
        old_table=pd.read_csv(prior.OUT/'submission_candidate.csv',dtype={'id':str})
        table=pd.read_csv(OUT/'submission_candidate.csv',dtype={'id':str})
        if old_table.id.tolist()==table.id.tolist() and np.allclose(old_table.priority,table.priority,atol=1e-12,rtol=1e-12):
            shutil.copyfile(prior.OUT/'submission_candidate.csv',OUT/'submission_candidate.csv')
    old=parent_pack('final_test',fit,pred);rich=joblib.load(OUT/'final_test_rich.joblib')
    joblib.dump(dict(choice=choice,fit=fit,pred=pred,lineage=old['lineage'],base_recipe=old['recipe'],
        projection=rich['projection'],columns=feat.frame(ctx,old,rich,arm,'a').columns.tolist(),
        model_file=f"final_{'reference' if arm=='matched' else arm}.cbm",
        protocol_sha256=feat.digest(OUT/'protocol.json')),OUT/'final_recipe.joblib',compress=3)
    print('FINAL',json.dumps(choice),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preflight','prepare','fit','score'])
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    with threadpool_limits(4):
        ctx=context()
        if args.stage=='preflight':
            preflight(ctx)
        elif args.stage=='prepare':
            prepare(ctx)
        else:
            protocol(ctx)
            if args.stage=='fit':
                assert 0<=args.shard<args.shards
                for i,(name,fit,pred) in enumerate(jobs(ctx)):
                    if i%args.shards==args.shard:
                        fit_partition(ctx,name,fit,pred)
            else:
                score(ctx)
    print('DONE',args.stage,args.shard,flush=True)


if __name__=='__main__':
    main()
