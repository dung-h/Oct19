#!/usr/bin/env python3
"""Grouped five-fold pilot for label-trained scientific abstract representations."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
import torch
from threadpoolctl import threadpool_limits

import model_abstract as m
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'specter2_preference'))
import run_preference as prior

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
FAMILIES=['incumbent',*m.ARMS]
THRESHOLDS=m.old.THRESHOLDS.copy()


def context():
    ctx=prior.context()
    return m.prepare(ctx,OUT)


def full_digest(data):
    columns=['split','id','title','abstract','venue','year','authors','doi','Label']
    return hashlib.sha256(json.dumps(data[columns].fillna('').to_dict('records'),sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def partitions(ctx):
    for fold,fit,val,nums in prior.outer_members(ctx):
        yield f'outer{fold}',fit,val,nums
    fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
    frame=pd.read_csv(prior.PRIOR/'final_inner_splits.csv',dtype={'id':str})
    assert frame.id.tolist()==ctx['data'].id.iloc[fit].tolist()
    yield 'final',fit,test,frame.inner_fold.to_numpy()


def protocol(ctx,create=False):
    files=[HERE/'PLAN.md',HERE/'model_abstract.py',HERE/'run_abstract.py',
        Path(m.old.__file__),Path(prior.__file__),Path(prior.ab.__file__),
        Path(prior.ab.diag.__file__),Path(prior.ab.baseline.__file__),
        HERE.parent/'specter2_v2/prepare_data.py']
    caches=[]
    for name,_,_,_ in partitions(ctx):
        caches.append(prior.CLAIM/(name+'_inner0_predictions.npz'))
        if name!='final':caches.append(prior.CLAIM/(name+'_validation_predictions.npz'))
    caches.extend([prior.PRIOR/'submission_candidate.csv',prior.PRIOR/'test_prediction_details.csv'])
    rec=dict(source_sha256={str(p.relative_to(HERE.parents[1])):m.digest(p) for p in files},
        full_input_sha256=full_digest(ctx['data']),raw_zip_sha256=m.digest(prior.ab.baseline.ROOT/'assignment-data-mining-261.zip'),
        grouping_sha256=hashlib.sha256(ctx['groups'].astype(np.int64).tobytes()).hexdigest(),
        labels_sha256=hashlib.sha256(ctx['y'][:2494].astype(np.int64).tobytes()).hexdigest(),
        split_sha256=m.digest(prior.ab.V2/'nested_splits.csv'),final_split_sha256=m.digest(prior.PRIOR/'final_inner_splits.csv'),
        incumbent_cache_sha256={str(p.relative_to(HERE.parent)):m.digest(p) for p in caches},
        input_tokens={k:v['token_hash'] for k,v in ctx['modes'].items()},
        base=m.old.BASE,base_revision=m.old.BASE_REV,adapter=m.old.ADAPTER,adapter_revision=m.old.ADAPTER_REV,
        seed=m.SEED,arms=m.ARMS,epochs=m.EPOCHS,thresholds=THRESHOLDS.tolist(),
        max_length=m.MAX_LENGTH,microbatch=m.MICRO,effective_batch=m.BATCH,
        optimizer='AdamW head1e-3 adapter5e-5 wd.01 clip1; CE5+.5BCE(read)',
        validation='5 inherited outer groups, inner0 calibration only; exploratory pilot, not 4-inner mean',
        versions={name:importlib.metadata.version(name) for name in ['torch','transformers','adapters','numpy','pandas','scikit-learn']})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==rec,'Locked protocol/source/input changed'
    else:
        assert create,'Run prepare first'
        path.write_text(json.dumps(rec,indent=2)+'\n')
    return rec


def prepare(ctx):
    protocol(ctx,create=True)
    m.prepare_vectors(ctx,OUT,m.digest(OUT/'protocol.json'))
    print('PREPARED',flush=True)


def incumbent(name,fit,pred):
    if name=='final_test':
        table=pd.read_csv(prior.PRIOR/'test_prediction_details.csv',dtype={'id':str})
        return table[[f'p{k}' for k in range(1,6)]].to_numpy(),None
    return prior.load_incumbent(name,fit,pred,train=('validation' in name))


def calibrate(ctx,fit,nums,prefix):
    train=fit[nums!=0];val=fit[nums==0]
    values={};info={}
    for arm in m.ARMS:
        values[arm],info[arm],_=m.fit(ctx,train,val,arm,m.EPOCHS,OUT/f'{prefix}_calibration_{arm}',m.digest(OUT/'protocol.json'))
    p,_=incumbent(prefix+'_inner0',train,val)
    values['incumbent']={0:p}
    return values,info


def choose(ctx,calibration,values):
    rows=[];choices={}
    y=ctx['y'][calibration];ids=ctx['data'].id.iloc[calibration].to_numpy()
    for family in FAMILIES:
        candidates=[]
        for epoch,probability in values[family].items():
            for threshold in THRESHOLDS:
                p=m.old.priority(probability,threshold)
                result=prior.official(y,p,ids)
                item=dict(family=family,epoch=int(epoch),threshold=float(threshold),
                    calibration_H=result['score'],calibration_R=result['coverage'])
                candidates.append(item);rows.append(item)
        choices[family]=max(candidates,key=lambda v:v['calibration_H']).copy()
    choices['neural_selection']=max([choices[a] for a in m.ARMS],key=lambda v:v['calibration_H']).copy()
    choices['selection_procedure']=max([choices[a] for a in FAMILIES],key=lambda v:v['calibration_H']).copy()
    return choices,rows


def measure(ctx,fit,val,prediction,training,threshold):
    p=m.old.priority(prediction,threshold)
    result=prior.official(ctx['y'][val],p,ctx['data'].id.iloc[val].to_numpy())
    result.update(AUC=float(roc_auc_score(ctx['y'][val]>1,1-prediction[:,0])),
        logloss=float(log_loss(ctx['y'][val].astype(int)-1,prediction,labels=np.arange(5))),
        gain_MAE=float(np.mean(np.abs(prediction@(np.arange(5)/4)-(ctx['y'][val]-1)/4))),
        train_AUC=float(roc_auc_score(ctx['y'][fit]>1,1-training[:,0])))
    result['AUC_gap']=result['train_AUC']-result['AUC']
    return result,p


def run(ctx,stage):
    protocol(ctx)
    m.prepare_vectors(ctx,OUT,m.digest(OUT/'protocol.json'))
    metrics=[];search=[]
    oof=ctx['data'][['id','title','venue','year','Label']].iloc[:2494].copy()
    for prefix,fit,pred,nums in partitions(ctx):
        final=prefix=='final'
        if stage=='outer' and final:break
        if stage=='final' and not final:continue
        values,_=calibrate(ctx,fit,nums,prefix)
        choices,rows=choose(ctx,fit[nums==0],values)
        (OUT/f'{prefix}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
        pd.DataFrame(rows).to_csv(OUT/f'{prefix}_calibration_search.csv',index=False)
        predictions={};training={}
        for arm in m.ARMS:
            epoch=choices[arm]['epoch']
            save_dir=OUT/f'final_{arm}_model' if final else None
            result,_,train=m.fit(ctx,fit,pred,arm,[epoch],OUT/f'{prefix}_validation_{arm}',
                m.digest(OUT/'protocol.json'),include_train=True,save_dir=save_dir)
            predictions[arm]=result[epoch];training[arm]=train
        if final:
            predictions['incumbent'],_=incumbent('final_test',fit,pred)
            for policy,choice in choices.items():
                probability=predictions[choice['family']]
                priority=m.old.priority(probability,choice['threshold'])
                details=ctx['data'][['id','title','venue']].iloc[pred].copy()
                details['priority']=priority
                for k in range(5):details[f'p{k+1}']=probability[:,k]
                details.to_csv(OUT/f'final_{policy}_predictions.csv',index=False)
                table=ctx['template'][['id']].merge(details[['id','priority']],on='id',validate='one_to_one')
                table.to_csv(OUT/f'submission_{policy}.csv',index=False)
            shutil.copyfile(OUT/'submission_selection_procedure.csv',OUT/'submission_candidate.csv')
            choice=choices['selection_procedure']
            recipe=dict(choice=choice,model_dir=f"final_{choice['family']}_model" if choice['family']!='incumbent' else 'parent incumbent',
                protocol_sha256=m.digest(OUT/'protocol.json'),evaluation='pilot inner0 calibration; exploratory')
            (OUT/'final_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
            print('FINAL',json.dumps(choices),flush=True)
        else:
            predictions['incumbent'],training['incumbent']=incumbent(prefix+'_validation',fit,pred)
            fold=int(prefix[len('outer'):])
            for policy,choice in choices.items():
                family=choice['family']
                result,p=measure(ctx,fit,pred,predictions[family],training[family],choice['threshold'])
                metrics.append(dict(outer_fold=fold,policy=policy,selected_family=family,
                    epoch=choice['epoch'],threshold=choice['threshold'],calibration_H=choice['calibration_H'],**result))
                oof.loc[pred,policy+'_priority']=p
                for k in range(5):oof.loc[pred,policy+f'_p{k+1}']=predictions[family][:,k]
            oof.loc[pred,'outer_fold']=fold
            search.extend(dict(outer_fold=fold,**v) for v in rows)
            pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
            pd.DataFrame(search).to_csv(OUT/'calibration_search.csv',index=False)
            oof.to_csv(OUT/'oof.csv',index=False)
            print('OUTER',fold,pd.DataFrame(metrics).query('outer_fold==@fold')[['policy','score','FP','FN','missed_high','AUC']].to_string(index=False),flush=True)
    print('DONE',stage,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','prepare','all','outer','final'])
    args=parser.parse_args();torch.set_num_threads(4)
    with threadpool_limits(4):
        ctx=context()
        if args.stage=='preflight':m.preflight(ctx,OUT)
        elif args.stage=='prepare':prepare(ctx)
        else:run(ctx,args.stage)


if __name__=='__main__':main()
