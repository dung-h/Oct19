#!/usr/bin/env python3
"""Fail-fast training smoke test for v2; it deliberately does not compute H."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
import apm_context as ac
import model_v2 as m

ARMS = ("query_only", "contextual")
TRAIN_N = 512
PRED_N = 128


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(ctx, stats):
    return dict(plan_sha256=digest(HERE / "PLAN.md"), source_sha256={x: digest(HERE / x) for x in ["model_v2.py", "run_smoke.py"]},
                data_sha256=hashlib.sha256(ctx["data"].to_csv(index=False).encode()).hexdigest(), split_sha256=digest(ac.split_path()),
                seed=m.SEED, encoder=m.v1.old.BASE, encoder_revision=m.v1.old.BASE_REV,
                adapter=m.v1.old.ADAPTER, adapter_revision=m.v1.old.ADAPTER_REV,
                fix="L2-normalize CLS before a head that multiplies by 10; GradScaler then unscale before clip",
                smoke=dict(train_n=TRAIN_N, pred_n=PRED_N, epochs=m.EPOCHS, arms=ARMS, batch=m.QUERY_BATCH), input_stats=stats)


def forward_loss(ctx, encoder, head, arm, query, refs=None, monitor=False):
    value = m.forward(ctx, encoder, head, arm, query, refs, monitor=monitor)
    labels = torch.tensor(ctx["labels"][query], device="cuda", dtype=torch.long)
    if arm == "query_only":
        logits, hidden = value if monitor else (value, None)
        return m.query_loss(logits.float(), labels), logits, hidden
    logits, compare, hidden = value if monitor else (*value, None)
    ref_labels = torch.tensor(ctx["labels"][refs], device="cuda", dtype=torch.long)
    return m.contextual_loss(logits.float(), compare.float(), labels, ref_labels), logits, hidden


def prediction(ctx, encoder, head, arm, ref_pool, query):
    refs, audit = m.references(ctx, ref_pool, query) if arm == "contextual" else (None, [])
    values=[]; active=[]
    encoder.eval(); head.eval()
    with torch.inference_mode():
        for start in range(0, len(query), m.QUERY_BATCH):
            q=query[start:start+m.QUERY_BATCH]; r=None if refs is None else refs[start:start+len(q)]
            with torch.autocast("cuda", dtype=torch.float16):
                _, logits, hidden = forward_loss(ctx, encoder, head, arm, q, r, monitor=True)
            values.append(torch.softmax(logits.float(), 1).cpu().numpy())
            active.append(float((hidden.abs().sum(1) > 0).float().mean()))
    return np.vstack(values), dict(reference_rows=len(audit), active_relu_fraction=float(np.mean(active)))


def run_arm(ctx, train, pred, arm):
    torch.manual_seed(m.SEED); rng=np.random.default_rng(m.SEED)
    encoder=m.adapter(); head=m.head(arm); params=[p for p in encoder.parameters() if p.requires_grad]
    initial=[p.detach().clone() for p in params]
    optimizer=torch.optim.AdamW([{"params":head.parameters(),"lr":.001},{"params":params,"lr":.00005}],weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    refs,_=m.references(ctx, train, train) if arm=="contextual" else (None,None)
    # Same weights and same train batch in FP32/AMP before update: numerical guard, not a score.
    probe=train[:m.QUERY_BATCH]; probe_ref=None if refs is None else refs[:m.QUERY_BATCH]
    encoder.eval();head.eval()
    with torch.inference_mode():
        fp32=forward_loss(ctx,encoder,head,arm,probe,probe_ref)[1].float()
        with torch.autocast("cuda",dtype=torch.float16): amp=forward_loss(ctx,encoder,head,arm,probe,probe_ref)[1].float()
    fp32_amp_max_abs=float((fp32-amp).abs().max())
    rows=[]; nonzero=0; total=0; active=[]; started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,m.EPOCHS+1):
        order=train[rng.permutation(len(train))]; losses=[]
        encoder.eval();head.train()
        for start in range(0,len(order),m.QUERY_BATCH):
            q=order[start:start+m.QUERY_BATCH]; r=None if refs is None else refs[np.searchsorted(train,q)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16): loss,_,hidden=forward_loss(ctx,encoder,head,arm,q,r,monitor=True)
            if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(list(head.parameters())+params,1.)
            if not bool(torch.isfinite(norm)): raise RuntimeError("non-finite gradient norm")
            nonzero+=int(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in params));total+=1
            active.append(float((hidden.abs().sum(1)>0).float().mean()))
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        rows.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.monotonic()-started,scale=float(scaler.get_scale())))
    probs,pred_monitor=prediction(ctx,encoder,head,arm,train,pred)
    unique,counts=np.unique(probs,axis=0,return_counts=True);read=1-probs[:,0]
    report=dict(arm=arm,trace=rows,amp_fp32_max_abs=fp32_amp_max_abs,train_active_relu_fraction=float(np.mean(active)),
        nonzero_adapter_steps=nonzero,total_steps=total,adapter_gradient_fraction=nonzero/total,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),adapter_l2_drift=float(sum((p.detach()-q).square().sum().item() for p,q in zip(params,initial))**.5),
        probabilities=dict(n=len(pred),unique=int(len(unique)),largest_duplicate=int(counts.max()),read_std=float(read.std()),active_relu_fraction=pred_monitor["active_relu_fraction"]),
        heldout_auc=float(roc_auc_score(ctx["labels"][pred]>1,read)),heldout_score_is_smoke_only=True)
    # Varying output, active head, and adapter gradient are hard requirements. AUC is descriptive only.
    checks=dict(adapter_gradient=nonzero==total,train_relu=report["train_active_relu_fraction"]>.99,
        pred_relu=pred_monitor["active_relu_fraction"]>.99,variation=len(unique)>=max(8,len(pred)//10),
        no_dominant_vector=int(counts.max())<len(pred)//2,read_variance=read.std()>1e-4,finite=np.isfinite(probs).all())
    checks={key: bool(value) for key, value in checks.items()}
    report["checks"]=checks;report["status"]="PASSED" if all(checks.values()) else "FAILED"
    np.savez_compressed(OUT/f"smoke_{arm}_prediction.npz",train=train,pred=pred,probability=probs)
    del encoder,head,optimizer,params,initial;torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--arm",choices=[*ARMS,"both"],default="both");args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);ctx=ac.context();stats=m.prepare(ctx)
    protocol=hashes(ctx,stats);(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    fold,pool,_,nums=ac.outer_jobs(ctx)[0]; train_pool=pool[nums!=0]; heldout=pool[nums==0]
    rng=np.random.default_rng(m.SEED); train=np.sort(rng.choice(train_pool,size=TRAIN_N,replace=False));pred=np.sort(rng.choice(heldout,size=PRED_N,replace=False))
    assert set(ctx["groups"][train]).isdisjoint(ctx["groups"][pred])
    selected=ARMS if args.arm=="both" else (args.arm,);reports={arm:run_arm(ctx,train,pred,arm) for arm in selected}
    outcome=dict(status="PASSED" if all(x["status"]=="PASSED" for x in reports.values()) else "FAILED",outer_fold=int(fold),inner_fold=0,protocol_sha256=digest(OUT/"protocol.json"),reports=reports)
    (OUT/"smoke.json").write_text(json.dumps(outcome,indent=2)+"\n");print(json.dumps(outcome,indent=2))


if __name__=="__main__":main()
