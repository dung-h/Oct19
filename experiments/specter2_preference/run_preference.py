#!/usr/bin/env python3
"""Resumable nested SPECTER2 preference adaptation with a matched frozen control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
import torch
from threadpoolctl import threadpool_limits

import modeling as m

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'
sys.path.insert(0, str(HERE.parent/'input_ablation'))
import run_ablation as ab

PRIOR = HERE.parent/'preference_augmentation'/'outputs'
CLAIM = HERE.parent/'claim_stability'/'outputs'
ARMS = ['frozen', 'adapted']


def context():
    data, template = ab.load_clean_data()
    y = np.full(len(data), np.nan)
    y[:2494] = data.Label.iloc[:2494].astype(int)
    return dict(data=data, template=template, y=y, groups=ab.baseline.grouping(data.reset_index(drop=True)))


def protocol(ctx):
    value = dict(plan_sha256=m.digest(HERE/'PLAN.md'),
                 sources={f:m.digest(HERE/f) for f in ['modeling.py', 'run_preference.py']},
                 data_sha256=ab.baseline.input_digest(ctx['data']),
                 split_sha256=m.digest(ab.V2/'nested_splits.csv'),
                 final_split_sha256=m.digest(PRIOR/'final_inner_splits.csv'),
                 base=m.BASE, base_revision=m.BASE_REV, adapter=m.ADAPTER, adapter_revision=m.ADAPTER_REV,
                 seed=m.SEED, epochs=m.CHECKPOINTS, thresholds=m.THRESHOLDS.tolist(),
                 input='title + SEP; CLS L2; max128; same pretrained regression adapter',
                 loss='CE5 + 0.5*BCE(logsumexp(logits2..5)-logit1, label>1)',
                 optimizer='AdamW head0.001 adapter0.00005 wd0.01 clip1 batch32; no resampling',
                 model='base frozen; encoder dropout disabled; same MLP64/dropout0.1 head initialization',
                 policy='each arm epoch and cutoff maximize mean four inner H; compare incumbent both_d4',
                 limitation='Repeatedly inspected outer data: exploratory, not an untouched benchmark')
    path=OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Protocol/source changed'
    else:
        path.write_text(json.dumps(value, indent=2)+'\n')
    return value


def outer_members(ctx):
    data=ctx['data'];splits=pd.read_csv(ab.V2/'nested_splits.csv', dtype={'id':str})
    lookup=dict(zip(data.id.iloc[:2494], range(2494)))
    for fold in range(5):
        rows=splits[splits.outer_fold.eq(fold)]
        f=rows[rows.partition.eq('outer_train')];v=rows[rows.partition.eq('outer_validation')]
        yield fold, np.array([lookup[i] for i in f.id]), np.array([lookup[i] for i in v.id]), f.inner_fold.to_numpy()


def official(y, p, ids):
    order=np.array(sorted(np.flatnonzero(p>0), key=lambda i:(-p[i],str(ids[i]))), dtype=int)
    weights=1/np.log2(np.arange(len(order))+2)
    q=float(np.average((y[order]-1)/4, weights=weights)) if len(order) else 0.
    tp=int((y[order]>1).sum());pos=int((y>1).sum());r=tp/pos if pos else 0.
    return dict(score=2*q*r/(q+r) if q+r else 0., relevance=q, coverage=r,
                TP=tp, FP=len(order)-tp, FN=pos-tp, selected=len(order),
                missed_high=int(((y>=4)&(p==0)).sum()))


def load_incumbent(prefix, fit, pred, train=False):
    path=CLAIM/(prefix+'_predictions.npz')
    z=np.load(path)
    assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
    return z['reference'], z['train_reference'] if train else None


def inner(ctx, fit, nums, prefix):
    values={arm:{e:np.full((len(fit),5),np.nan) for e in m.CHECKPOINTS} for arm in ARMS}
    values['incumbent']={0:np.full((len(fit),5),np.nan)}
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask]
        assert set(ctx['groups'][a]).isdisjoint(ctx['groups'][b])
        for arm in ARMS:
            result,_,_=m.fit(ctx,a,b,arm,m.CHECKPOINTS,OUT/f'{prefix}_inner{k}_{arm}')
            for e in m.CHECKPOINTS:
                values[arm][e][mask]=result[e]
        p,_=load_incumbent(f'{prefix}_inner{k}',a,b)
        values['incumbent'][0][mask]=p
        print(prefix,'inner',k,'complete',flush=True)
    assert all(np.isfinite(p).all() for epochs in values.values() for p in epochs.values())
    return values


def choose(ctx, fit, nums, values):
    y=ctx['y'][fit];ids=ctx['data'].id.iloc[fit].to_numpy()
    rows=[];choices={};recall_choices={}
    for arm, epochs in values.items():
        candidates=[]
        for epoch, p in epochs.items():
            for cutoff in m.THRESHOLDS:
                priority=m.priority(p,cutoff)
                scores=[official(y[nums==k],priority[nums==k],ids[nums==k]) for k in range(4)]
                row=dict(family=arm,epoch=epoch,threshold=float(cutoff),
                         inner_score=float(np.mean([r['score'] for r in scores])),
                         inner_recall=float(np.mean([r['coverage'] for r in scores])))
                candidates.append(row);rows.append(row)
        choices[arm]=max(candidates,key=lambda r:r['inner_score']).copy()
        same_epoch=[r for r in candidates if r['epoch']==choices[arm]['epoch']]
        recall_choices[arm]=min(same_epoch,key=lambda r:abs(r['inner_recall']-.8)).copy()
    choices['neural_selection']=max([choices[a] for a in ARMS],key=lambda r:r['inner_score']).copy()
    # Incumbent wins exact ties. No outer score is consulted.
    choices['selection_procedure']=max([choices[a] for a in ['incumbent',*ARMS]],key=lambda r:r['inner_score']).copy()
    return choices,recall_choices,rows


def measure(ctx, fit, val, probability, train_probability, cutoff):
    y=ctx['y'];priority=m.priority(probability,cutoff)
    result=official(y[val],priority,ctx['data'].id.iloc[val].to_numpy())
    result.update(AUC=float(roc_auc_score(y[val]>1,1-probability[:,0])),
                  logloss=float(log_loss(y[val].astype(int)-1,probability,labels=np.arange(5))),
                  train_AUC=float(roc_auc_score(y[fit]>1,1-train_probability[:,0])))
    result['AUC_gap']=result['train_AUC']-result['AUC']
    return result,priority


def run_outer(ctx):
    data=ctx['data'];oof=data[['id','title','venue','Label']].iloc[:2494].copy()
    metrics=[];matched=[];search=[]
    for fold,fit,val,nums in outer_members(ctx):
        assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][val])
        values=inner(ctx,fit,nums,f'outer{fold}')
        choices,recall_choices,rows=choose(ctx,fit,nums,values)
        search.extend(dict(outer_fold=fold,**r) for r in rows)
        (OUT/f'outer{fold}_choices.json').write_text(json.dumps(dict(primary=choices,recall80=recall_choices),indent=2)+'\n')
        predictions={};training={}
        for arm in ARMS:
            e=choices[arm]['epoch']
            result,_,train=m.fit(ctx,fit,val,arm,[e],OUT/f'outer{fold}_validation_{arm}',include_train=True)
            predictions[arm]=result[e];training[arm]=train
        predictions['incumbent'],training['incumbent']=load_incumbent(f'outer{fold}_validation',fit,val,train=True)
        for family,choice in choices.items():
            a=choice['family']
            result,p=measure(ctx,fit,val,predictions[a],training[a],choice['threshold'])
            metrics.append(dict(outer_fold=fold,family=family,selected_family=a,epoch=choice['epoch'],
                                threshold=choice['threshold'],inner_score=choice['inner_score'],**result))
            oof.loc[val,family+'_priority']=p
            for k in range(5):oof.loc[val,family+f'_p{k+1}']=predictions[a][:,k]
        for family,choice in recall_choices.items():
            result,_=measure(ctx,fit,val,predictions[family],training[family],choice['threshold'])
            matched.append(dict(outer_fold=fold,family=family,epoch=choice['epoch'],threshold=choice['threshold'],
                                inner_recall=choice['inner_recall'],**result))
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(matched).to_csv(OUT/'recall80_fold_metrics.csv',index=False)
        pd.DataFrame(search).to_csv(OUT/'inner_search.csv',index=False)
        oof.to_csv(OUT/'oof_predictions.csv',index=False)
        print('OUTER',fold,pd.DataFrame(metrics).query('outer_fold==@fold')[['family','score','FP','FN','AUC']].to_string(index=False),flush=True)


def run_final(ctx):
    fit=np.arange(2494);pred=np.arange(2494,len(ctx['data']))
    membership=pd.read_csv(PRIOR/'final_inner_splits.csv',dtype={'id':str})
    membership=ctx['data'][['id']].iloc[:2494].merge(membership,on='id',validate='one_to_one')
    nums=membership.inner_fold.to_numpy()
    membership.to_csv(OUT/'final_inner_splits.csv',index=False)
    values=inner(ctx,fit,nums,'final')
    choices,recall_choices,search=choose(ctx,fit,nums,values)
    (OUT/'final_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    pd.DataFrame(search).to_csv(OUT/'final_search.csv',index=False)
    challenger=choices['neural_selection'];arm=challenger['family'];epoch=challenger['epoch']
    directory=OUT/'neural_model'
    result,_,_=m.fit(ctx,fit,pred,arm,[epoch],OUT/'final_neural',save_dir=directory)
    p=m.priority(result[epoch],challenger['threshold'])
    details=ctx['data'][['id','title','venue']].iloc[pred].copy()
    details['priority']=p
    for k in range(5):details[f'p{k+1}']=result[epoch][:,k]
    details.to_csv(OUT/'neural_test_predictions.csv',index=False)
    submission=ctx['template'][['id']].merge(details[['id','priority']],on='id',validate='one_to_one')
    submission.to_csv(OUT/'submission_neural_challenger.csv',index=False)
    choice=choices['selection_procedure']
    if choice['family']=='incumbent':
        previous=json.loads((PRIOR/'final_choice.json').read_text())
        assert previous['key']=='both_d4' and abs(previous['threshold']-choice['threshold'])<1e-12
        shutil.copyfile(PRIOR/'submission_candidate.csv',OUT/'submission_candidate.csv')
    else:
        assert choice==challenger
        submission.to_csv(OUT/'submission_candidate.csv',index=False)
    print('FINAL',json.dumps(choices),flush=True)


def report():
    metrics=pd.read_csv(OUT/'fold_metrics.csv')
    def summarize(frame):
        return frame.groupby('family',sort=False).agg(H=('score','mean'),H_std=('score','std'),Q=('relevance','mean'),
            R=('coverage','mean'),FP=('FP','sum'),FN=('FN','sum'),missed_high=('missed_high','sum'),
            AUC=('AUC','mean'),train_AUC=('train_AUC','mean'),AUC_gap=('AUC_gap','mean'),logloss=('logloss','mean'))
    summary=summarize(metrics);summary.to_csv(OUT/'summary.csv')
    matched=summarize(pd.read_csv(OUT/'recall80_fold_metrics.csv'));matched.to_csv(OUT/'recall80_summary.csv')
    effects=[]
    for arm,ref in [('adapted','frozen'),('adapted','incumbent'),('selection_procedure','incumbent')]:
        a=metrics[metrics.family.eq(arm)].set_index('outer_fold');b=metrics[metrics.family.eq(ref)].set_index('outer_fold')
        for f in a.index:
            effects.append(dict(outer_fold=int(f),arm=arm,reference=ref,
                **{'delta_'+k:float(a.loc[f,k]-b.loc[f,k]) for k in ['score','FP','FN','AUC','coverage','relevance','missed_high']}))
    pd.DataFrame(effects).to_csv(OUT/'paired_effects.csv',index=False)
    choice=json.loads((OUT/'final_choices.json').read_text())
    primary=summary.loc['adapted','H']-summary.loc['frozen','H']
    wins=int((pd.DataFrame(effects).query("arm=='adapted' and reference=='frozen'").delta_score>0).sum())
    verified=json.loads((OUT/'verification.json').read_text()) if (OUT/'verification.json').exists() else {'status':'pending'}
    text=f'''# SPECTER2 học theo sở thích — kết quả

Đối chứng matched: cùng title, pretrained regression adapter, CLS L2, head và loss.
`adapted` cập nhật adapter SPECTER2 theo nhãn; `frozen` chỉ học head. Base transformer
giữ cố định. Cả hai chọn epoch/ngưỡng bằng bốn inner folds; outer refit đúng epoch đó.

{summary.reset_index().to_markdown(index=False,floatfmt='.5f')}

Adapted − frozen: **{primary:+.5f} H**, cải thiện **{wins}/5 folds**.
Điểm policy được báo riêng, không thay bằng family thắng outer.

## Chẩn đoán tại inner recall gần 0.80

Giữ epoch đã chọn theo inner H, chọn ngưỡng có mean inner recall gần0.80 nhất.
Outer recall không bị ép bằng0.80; cần đọc FP cùng recall thực tế.

{matched.reset_index().to_markdown(index=False,floatfmt='.5f')}

## Chọn cuối

```json
{json.dumps(choice,indent=2)}
```

`submission_neural_challenger.csv` là neural arm được inner chọn trên toàn train.
`submission_candidate.csv` là kết quả global policy, có thể giữ incumbent. Các
model cũ không bị thay thế; chưa gửi Kaggle. Reload neural model và kiểm tra CSV:
`{verified.get('status','pending')}`; chi tiết `outputs/verification.json`.

## Giới hạn và tái lập

Đây là một cấu hình adapter (LR5e-5, checkpoints1/2/4/8), không kiểm tra full
fine-tuning, mọi learning rate hoặc mọi encoder. Không tự động kết luận 0.7 bất khả
thi khi cấu hình này thất bại. Outer đã được xem trong các nghiên cứu trước; nested
selection không biến nó thành benchmark độc lập. Không đổi nhãn, chọn seed theo
outer, sinh sở thích hay dùng nhãn test.

`PLAN.md`, `outputs/protocol.json` cố định thiết kế và hash. Raw probabilities,
training traces, membership và inner choices được lưu để kiểm tra lại.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/run_preference.py all
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/verify.py
```

Tài liệu: [SPECTER2](https://huggingface.co/allenai/specter2),
[AdapterHub](https://docs.adapterhub.ml/training.html).
'''
    (HERE/'RESULTS.md').write_text(text)
    print(summary.to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','outer','final','report','all'],default='all',nargs='?')
    args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic()
    with threadpool_limits(4):
        if args.stage=='report':
            report();return
        ctx=context();protocol(ctx);m.prepare(ctx,OUT)
        if args.stage in ['outer','all']:run_outer(ctx)
        if args.stage in ['final','all']:run_final(ctx)
        if args.stage=='all':report()
    print('DONE',args.stage,'seconds',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':main()
