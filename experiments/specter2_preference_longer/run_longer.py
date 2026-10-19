#!/usr/bin/env python3
"""Nested comparison of adapted SPECTER2 checkpoints 8 and 16 epochs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
OLD=HERE.parent/'specter2_preference'
sys.path.insert(0,str(OLD))
import modeling as m
import run_preference as prior
from run_preference import ab

EPOCHS=[8,16]
FAMILIES=['adapted_8','adapted_16','incumbent']


def context():return prior.context()


def protocol(ctx):
    rec=dict(plan_sha256=m.digest(HERE/'PLAN.md'),source_sha256=m.digest(__file__),
        parent_protocol_sha256=m.digest(prior.OUT/'protocol.json'),
        model_sha256={name:m.digest(OLD/name) for name in ['modeling.py','run_preference.py']},
        data_sha256=ab.baseline.input_digest(ctx['data']),
        source_zip_sha256=m.digest(ab.baseline.ROOT/'assignment-data-mining-261.zip'),
        split_sha256=m.digest(ab.V2/'nested_splits.csv'),final_split_sha256=m.digest(prior.PRIOR/'final_inner_splits.csv'),
        labels_sha256=hashlib.sha256(ctx['y'][:2494].astype(np.int64).tobytes()).hexdigest(),
        parent_submission_sha256=m.digest(prior.PRIOR/'submission_candidate.csv'),
        preflight_correction_sha256=m.digest(HERE/'PROTOCOL_AMENDMENT.md'),
        epochs=EPOCHS,seed=m.SEED,thresholds=m.THRESHOLDS.tolist(),families=FAMILIES,
        input='identical prior title-only SPECTER2 preference recipe; no abstract/metadata changes',
        policy='each checkpoint cutoff selected by mean four-inner H; global inner selection among epoch8, epoch16 and incumbent',
        score='detailed assignment H; exploratory repeated outer folds')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==rec,'Locked protocol/source changed'
    else:p.write_text(json.dumps(rec,indent=2)+'\n')
    return rec


def partition(ctx,fit,pred,name,include_train=False):
    path=OUT/(name+'_adapted.npz');meta=OUT/(name+'_adapted.json')
    if path.exists():
        z=np.load(path);assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
        assert np.array_equal(z['epochs'],EPOCHS)
        return {e:z[f'epoch{e}'] for e in EPOCHS},z['train'] if include_train else None
    # The assignment's test split contains duplicate groups also represented in
    # train. Match the existing final-run path that explicitly allows this case.
    save_dir=OUT/(name+'_training_model') if np.any(pred>=2494) else None
    started=time.monotonic();preds,info,train=m.fit(ctx,fit,pred,'adapted',EPOCHS,
        OUT/(name+'_adapted'),save_dir=save_dir,include_train=include_train)
    archived=np.load(prior.OUT/(name+'_adapted.npz'))
    assert 8 in archived['epochs'].tolist(),f'Prior run did not archive epoch8 for {name}'
    np.testing.assert_allclose(preds[8],archived['epoch8'],atol=1e-6,rtol=1e-6)
    # The archived control is a fixed-seed replay of the same first eight updates.
    copied=dict(info);copied['replay_seconds']=time.monotonic()-started
    meta.write_text(json.dumps(copied,indent=2)+'\n')
    return preds,train


def inner(ctx,fit,nums,prefix):
    values={'adapted_8':{8:np.full((len(fit),5),np.nan)},
            'adapted_16':{16:np.full((len(fit),5),np.nan)},
            'incumbent':{0:np.full((len(fit),5),np.nan)}}
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask];name=f'{prefix}_inner{k}'
        prediction,_=partition(ctx,a,b,name)
        values['adapted_8'][8][mask]=prediction[8]
        values['adapted_16'][16][mask]=prediction[16]
        values['incumbent'][0][mask],_=prior.load_incumbent(name,a,b)
    assert all(np.isfinite(p).all() for fam in values.values() for p in fam.values())
    return values


def choose(ctx,fit,nums,values):
    choices={};recall_choices={};rows=[]
    y=ctx['y'][fit];ids=ctx['data'].id.iloc[fit].to_numpy()
    for family,checkpoints in values.items():
        candidates=[]
        for epoch,probability in checkpoints.items():
            for cutoff in m.THRESHOLDS:
                priority=m.priority(probability,cutoff)
                measures=[prior.official(y[nums==k],priority[nums==k],ids[nums==k]) for k in range(4)]
                row=dict(family=family,epoch=int(epoch),threshold=float(cutoff),
                    inner_score=float(np.mean([v['score'] for v in measures])),
                    inner_recall=float(np.mean([v['coverage'] for v in measures])))
                candidates.append(row);rows.append(row)
        choices[family]=max(candidates,key=lambda v:v['inner_score']).copy()
        recall_choices[family]=min([v for v in candidates if v['epoch']==choices[family]['epoch']],
            key=lambda v:abs(v['inner_recall']-.8)).copy()
    choices['neural_selection']=max([choices['adapted_8'],choices['adapted_16']],key=lambda v:v['inner_score']).copy()
    choices['selection_procedure']=max([choices['incumbent'],choices['adapted_8'],choices['adapted_16']],
        key=lambda v:v['inner_score']).copy()
    return choices,recall_choices,rows


def view_for(preds,family):return preds[{'adapted_8':8,'adapted_16':16,'incumbent':0}[family]]


def evaluate(y,pred,ids,probability):
    priority=m.priority(probability,pred['threshold'])
    result=prior.official(y,priority,ids)
    result['AUC']=float(roc_auc_score(y>1,1-probability[:,0]))
    result['missed_high']=int(((y>=4)&(priority==0)).sum())
    result['gain_MAE']=float(np.abs(probability@(np.arange(5)/4)-(y-1)/4).mean())
    return result,priority


def run_outer(ctx):
    rows=[];search=[];oof=ctx['data'][['id','title','venue','Label']].iloc[:2494].copy()
    for fold,fit,val,nums in prior.outer_members(ctx):
        choices,_,candidate_rows=choose(ctx,fit,nums,inner(ctx,fit,nums,f'outer{fold}'))
        (OUT/f'outer{fold}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
        search.extend(dict(outer_fold=fold,**r) for r in candidate_rows)
        pred,_=partition(ctx,fit,val,f'outer{fold}_validation')
        inc,_=prior.load_incumbent(f'outer{fold}_validation',fit,val)
        preds={8:pred[8],16:pred[16],0:inc}
        for policy,choice in choices.items():
            fam=choice['family'];p=preds[{'adapted_8':8,'adapted_16':16,'incumbent':0}[fam]]
            result,priority=evaluate(ctx['y'][val],choice,ctx['data'].id.iloc[val].to_numpy(),p)
            rows.append(dict(outer_fold=fold,policy=policy,selected_family=fam,
                epoch=choice['epoch'],threshold=choice['threshold'],inner_H=choice['inner_score'],**result))
            oof.loc[val,policy+'_priority']=priority
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(search).to_csv(OUT/'inner_search.csv',index=False)
        oof.to_csv(OUT/'oof.csv',index=False)
        print('OUTER',fold,pd.DataFrame(rows).query('outer_fold==@fold')[['policy','selected_family','score','FP','FN','missed_high']].to_string(index=False),flush=True)


def run_final(ctx):
    fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
    membership=pd.read_csv(prior.PRIOR/'final_inner_splits.csv',dtype={'id':str})
    assert membership.id.tolist()==ctx['data'].id.iloc[:2494].tolist()
    nums=membership.inner_fold.to_numpy();values=inner(ctx,fit,nums,'final')
    choices,_,search=choose(ctx,fit,nums,values)
    (OUT/'final_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    pd.DataFrame(search).to_csv(OUT/'final_search.csv',index=False)
    pred,_=partition(ctx,fit,test,'final_test')
    for e in EPOCHS:
        priority=m.priority(pred[e],choices[f'adapted_{e}']['threshold'])
        table=ctx['template'][['id']].merge(pd.DataFrame({'id':ctx['data'].id.iloc[test].to_numpy(),'priority':priority}),on='id',validate='one_to_one')
        table.to_csv(OUT/f'submission_epoch{e}.csv',index=False)
    family=choices['selection_procedure']['family'];choice=choices['selection_procedure']
    if family=='incumbent':shutil.copyfile(prior.PRIOR/'submission_candidate.csv',OUT/'submission_candidate.csv')
    else:shutil.copyfile(OUT/f"submission_epoch{choice['epoch']}.csv",OUT/'submission_candidate.csv')
    selected=choice['epoch'] if family.startswith('adapted') else None
    if selected is not None:
        m.fit(ctx,fit,test,'adapted',[selected],OUT/'final_model',save_dir=OUT/'final_model')
    (OUT/'final_recipe.json').write_text(json.dumps(dict(family=family,epoch=selected,choice=choice,
        source_sha256=m.digest(__file__),model_path='final_model' if selected is not None else 'parent incumbent'),indent=2)+'\n')
    print('FINAL',json.dumps(choices),flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['outer','final','all'],default='all',nargs='?')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    with threadpool_limits(4):
        ctx=context();prior.protocol(ctx)
        # Reuse the exact pinned-tokenizer/cache embeddings, preserving the previous run.
        m.prepare(ctx,prior.OUT)
        m.CHECKPOINTS=EPOCHS.copy()
        oldcache=prior.OUT/'frozen_title_embeddings.npz';newcache=OUT/'frozen_title_embeddings.npz'
        if not newcache.exists():shutil.copyfile(oldcache,newcache)
        m.prepare(ctx,OUT);protocol(ctx)
        if args.stage in ['outer','all']:run_outer(ctx)
        if args.stage in ['final','all']:run_final(ctx)
    print('DONE',args.stage,flush=True)


if __name__=='__main__':main()
