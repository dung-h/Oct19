#!/usr/bin/env python3
"""Resumable grouped nested CV, guarded against the v1 collapse failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss

HERE=Path(__file__).resolve().parent; OUT=HERE/"outputs"
sys.path.insert(0,str(HERE.parent/"abstract_preference_margin"))
import apm_context as ac
import model_v2 as m

ARMS=("query_only","contextual")
THRESHOLDS=np.arange(0,.901,.05)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y,p,ids,threshold):
    selected=1-p[:,0]>threshold; priority=np.where(selected,1+9*(p@(np.arange(5)/4)),0.)
    order=np.asarray(sorted(np.flatnonzero(selected),key=lambda i:(-priority[i],str(ids[i]))),int)
    q=float(np.average((y[order]-1)/4,weights=1/np.log2(np.arange(len(order))+2))) if len(order) else 0.
    r=float(((y>1)&selected).sum()/(y>1).sum()); h=float(2*q*r/(q+r)) if q+r else 0.
    return dict(H=h,Q=q,R=r,accepted=int(selected.sum()),TP=int(((y>1)&selected).sum()),FP=int(((y==1)&selected).sum()),
                FN=int(((y>1)&~selected).sum()),missed_label4=int(((y==4)&~selected).sum()),missed_label5=int(((y==5)&~selected).sum()),priority=priority)


def loss_forward(ctx,encoder,head,arm,q,refs=None,monitor=False):
    result=m.forward(ctx,encoder,head,arm,q,refs,monitor=monitor); labels=torch.tensor(ctx["labels"][q],device="cuda",dtype=torch.long)
    if arm=="query_only":
        logits,hidden=result if monitor else (result,None)
        return m.query_loss(logits.float(),labels),logits,hidden
    logits,compare,hidden=result if monitor else (*result,None)
    ref_labels=torch.tensor(ctx["labels"][refs],device="cuda",dtype=torch.long)
    return m.contextual_loss(logits.float(),compare.float(),labels,ref_labels),logits,hidden


def predict(ctx,encoder,head,arm,ref_pool,pred):
    refs,audit=m.references(ctx,ref_pool,pred) if arm=="contextual" else (None,[])
    probability=[];active=[];encoder.eval();head.eval()
    with torch.inference_mode():
        for start in range(0,len(pred),m.QUERY_BATCH):
            q=pred[start:start+m.QUERY_BATCH];r=None if refs is None else refs[start:start+len(q)]
            with torch.autocast("cuda",dtype=torch.float16): _,logits,hidden=loss_forward(ctx,encoder,head,arm,q,r,monitor=True)
            probability.append(torch.softmax(logits.float(),1).cpu().numpy());active.append(float((hidden.abs().sum(1)>0).float().mean()))
    probability=np.vstack(probability); unique,counts=np.unique(probability,axis=0,return_counts=True);read=1-probability[:,0]
    audit=dict(reference_rows=len(audit),unique_probability_vectors=int(len(unique)),largest_duplicate=int(counts.max()),
               read_std=float(read.std()),active_relu_fraction=float(np.mean(active)))
    return probability,audit


def fit_predict(ctx,fit,pred,arm,prefix):
    path=Path(str(prefix)+".npz");meta=Path(str(prefix)+".json")
    if path.exists() and meta.exists():
        z=np.load(path);info=json.loads(meta.read_text());np.testing.assert_array_equal(z["fit"],fit);np.testing.assert_array_equal(z["pred"],pred)
        if info["arm"]==arm and info["epochs"]==m.EPOCHS and info["status"]=="PASSED": return z["probability"],info
    assert set(fit).isdisjoint(pred)
    if np.all(pred<len(ctx["labels"])): assert set(ctx["groups"][fit]).isdisjoint(ctx["groups"][pred])
    torch.manual_seed(m.SEED);rng=np.random.default_rng(m.SEED);encoder=m.adapter();head=m.head(arm)
    params=[p for p in encoder.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params]
    optimizer=torch.optim.AdamW([{"params":head.parameters(),"lr":.001},{"params":params,"lr":.00005}],weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.); refs,_=m.references(ctx,fit,fit) if arm=="contextual" else (None,None)
    trace=[];nonzero=total=0;active=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,m.EPOCHS+1):
        encoder.eval();head.train();order=rng.permutation(len(fit));losses=[]
        for start in range(0,len(order),m.QUERY_BATCH):
            loc=order[start:start+m.QUERY_BATCH];q=fit[loc];r=None if refs is None else refs[loc];optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16): value,_,hidden=loss_forward(ctx,encoder,head,arm,q,r,monitor=True)
            if not torch.isfinite(value): raise RuntimeError(f"{prefix}: non-finite loss")
            scaler.scale(value).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(list(head.parameters())+params,1.)
            if not bool(torch.isfinite(norm)): raise RuntimeError(f"{prefix}: non-finite gradient")
            nonzero+=int(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in params));total+=1
            active.append(float((hidden.abs().sum(1)>0).float().mean()));scaler.step(optimizer);scaler.update();losses.append(float(value.detach()))
        trace.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.monotonic()-started,scale=float(scaler.get_scale())))
        print(Path(prefix).name,arm,"epoch",epoch,"loss",round(trace[-1]["loss"],5),"seconds",round(trace[-1]["seconds"],1),flush=True)
    probability,pred_audit=predict(ctx,encoder,head,arm,fit,pred)
    checks=dict(adapter_gradient=nonzero==total,train_relu=float(np.mean(active))>.99,pred_relu=pred_audit["active_relu_fraction"]>.99,
                variation=pred_audit["unique_probability_vectors"]>=max(8,len(pred)//10),no_dominant_vector=pred_audit["largest_duplicate"]<len(pred)//2,
                read_variance=pred_audit["read_std"]>1e-4,finite=bool(np.isfinite(probability).all()))
    checks={key:bool(value) for key,value in checks.items()}
    info=dict(status="PASSED" if all(checks.values()) else "FAILED",arm=arm,epochs=m.EPOCHS,seed=m.SEED,fit=fit.tolist(),pred=pred.tolist(),
        trace=trace,seconds=time.monotonic()-started,peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        nonzero_adapter_steps=nonzero,total_steps=total,train_active_relu_fraction=float(np.mean(active)),
        adapter_l2_drift=float(sum((p.detach()-q).square().sum().item() for p,q in zip(params,initial))**.5),prediction_audit=pred_audit,checks=checks)
    if info["status"]!="PASSED": raise RuntimeError(f"{prefix}: collapse guard failed: {checks}")
    np.savez_compressed(path,fit=fit,pred=pred,probability=probability);meta.write_text(json.dumps(info,indent=2)+"\n")
    del encoder,head,optimizer,params,initial;torch.cuda.empty_cache();return probability,info


def inner(ctx,fit,nums,prefix):
    oof={arm:np.full((len(fit),5),np.nan,np.float32) for arm in ARMS}
    for k in range(4):
        mask=nums==k;a,b=fit[~mask],fit[mask]
        for arm in ARMS:oof[arm][mask],_=fit_predict(ctx,a,b,arm,OUT/f"{prefix}_inner{k}_{arm}")
        print(prefix,"inner",k,"complete",flush=True)
    choices={}
    for arm,p in oof.items():
        candidates=[]
        for threshold in THRESHOLDS:
            folds=[metrics(ctx["labels"][fit[nums==k]],p[nums==k],ctx["ids"][fit[nums==k]],threshold) for k in range(4)]
            candidates.append(dict(arm=arm,threshold=float(threshold),inner_H=float(np.mean([x["H"] for x in folds])),inner_Q=float(np.mean([x["Q"] for x in folds])),inner_R=float(np.mean([x["R"] for x in folds]))))
        choices[arm]=max(candidates,key=lambda x:x["inner_H"])
    return choices


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fold",type=int,choices=range(5),action="append");args=parser.parse_args()
    ctx=ac.context();stats=m.prepare(ctx);protocol=json.loads((OUT/"protocol.json").read_text());assert protocol["input_stats"]==stats
    manifest=dict(status="started",protocol_sha256=digest(OUT/"protocol.json"),source_sha256=digest(__file__));(OUT/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    requested=set(args.fold) if args.fold else set(range(5));rows=[]
    for fold,fit,val,nums in ac.outer_jobs(ctx):
        if fold not in requested:continue
        choices=inner(ctx,fit,nums,f"outer{fold}");(OUT/f"outer{fold}_choices.json").write_text(json.dumps(choices,indent=2)+"\n")
        for arm,choice in choices.items():
            p,info=fit_predict(ctx,fit,val,arm,OUT/f"outer{fold}_validation_{arm}");result=metrics(ctx["labels"][val],p,ctx["ids"][val],choice["threshold"])
            rows.append(dict(outer_fold=fold,arm=arm,threshold=choice["threshold"],inner_H=choice["inner_H"],AUC=float(roc_auc_score(ctx["labels"][val]>1,1-p[:,0])),logloss=float(log_loss(ctx["labels"][val]-1,p,labels=np.arange(5))),**{key:value for key,value in result.items() if key!="priority"},seconds=info["seconds"]))
        pd.DataFrame(rows).to_csv(OUT/"fold_metrics_partial.csv",index=False);print(pd.DataFrame(rows).query("outer_fold==@fold").to_string(index=False),flush=True)
    manifest.update(status="complete",completed_outer_folds=sorted(requested));(OUT/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":main()
