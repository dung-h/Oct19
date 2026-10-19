#!/usr/bin/env python3
"""Locked grouped-CV comparison of TabPFN V2 and incumbent CatBoost."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tabpfn_lib as x

SOURCE=["PLAN.md","tabpfn_lib.py","metric_reference.py","test_tabpfn.py","run.py","verify.py","report.py"]

def _names(ctx):
    ans=[]
    for f,fit,pred,nums in x.jobs(ctx):
        ans += [(f"outer{f}_inner{k}",fit[nums!=k],fit[nums==k]) for k in range(4)]
        ans += [(f"outer{f}_validation",fit,pred)]
    return ans

def _inputs(ctx):
    p=[x.PROJECT/"assignment-data-mining-261.zip",x.EXPERIMENTS/"specter2_v2/outputs/nested_splits.csv"]
    for name,fit,pred in _names(ctx):
        p += [x.ac.source.pack_path(name),x.ac.source.CLAIM/f"{name}_predictions.npz"]
    assert all(q.is_file() for q in p),[str(q) for q in p if not q.is_file()]
    return p

def preflight():
    x.OUT.mkdir(parents=True,exist_ok=True)
    text=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (x.OUT/"requirements.lock.txt").write_text(text)
    info=dict(status="PASSED",device=torch.cuda.get_device_name(0),vram_mib=round(torch.cuda.get_device_properties(0).total_memory/2**20),
              torch=torch.__version__,cuda=torch.version.cuda,tabpfn=importlib.metadata.version("tabpfn"),catboost=importlib.metadata.version("catboost"),
              n_estimators=4,fit_mode="fit_with_cache",precision="auto",seed=x.SEED)
    (x.OUT/"preflight.json").write_text(json.dumps(info,indent=2)+"\n")
    return info

def protocol(ctx):
    inp=_inputs(ctx); info=preflight()
    record=dict(plan_sha256=x.digest(x.HERE/"PLAN.md"),source_sha256={q:x.digest(x.HERE/q) for q in SOURCE},
                input_sha256={str(q.relative_to(x.PROJECT)):x.digest(q) for q in inp},requirements_sha256=x.digest(x.OUT/"requirements.lock.txt"),
                label_sha256=hashlib.sha256(ctx["labels"].astype(np.int8).tobytes()).hexdigest(),
                group_sha256=hashlib.sha256(ctx["groups"][:x.N_FIT].astype(np.int64).tobytes()).hexdigest(),
                tabpfn=info,model=dict(target="Label1..5 mapped0..4",classes=[0,1,2,3,4],categorical_index=0,venue_encoding="fit categories sorted; unseen missing"),
                threshold_grid=x.THRESHOLDS.tolist(),tie_rule="lower threshold",priority="0 or 1+9*expected_gain",outer_folds=5,inner_folds=4,uploaded=False)
    p=x.OUT/"protocol.json"
    if p.exists(): assert json.loads(p.read_text())==record,"Locked protocol/input changed"
    else: p.write_text(json.dumps(record,indent=2)+"\n")
    return record

def _cat_parity(ctx,name,fit,pred):
    _,a,b=x.pack(name,fit,pred); y=ctx["labels"][fit]-1
    m=x.catboost(); m.fit(a,y,cat_features=["venue"]); got=m.predict_proba(b)
    np.testing.assert_allclose(got,x.incumbent(name,fit,pred),atol=1e-7,rtol=1e-6)

def audit(ctx):
    subprocess.run([sys.executable,str(x.HERE/"test_tabpfn.py")],cwd=x.HERE,check=True)
    for name,fit,pred in _names(ctx):
        assert set(ctx["groups"][fit]).isdisjoint(ctx["groups"][pred])
        _,a,b=x.pack(name,fit,pred); xf,xp,cats=x.matrix(a,b)
        assert xf.shape==(len(fit),277) and xp.shape==(len(pred),277) and len(cats)>0
        _cat_parity(ctx,name,fit,pred)
    protocol(ctx)
    result=dict(status="PASSED",partitions=25,catboost_parity=True,group_isolation=True,feature_contract=True,
                protocol_sha256=x.digest(x.OUT/"protocol.json"),uploaded=False)
    (x.OUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)

def tab_prediction(ctx,name,fit,pred,sha):
    path=x.OUT/f"{name}_tabpfn.npz"
    if path.exists(): return x.load_prediction(path,fit,pred,sha)[0]
    _,a,b=x.pack(name,fit,pred); xf,xp,cats=x.matrix(a,b); y=ctx["labels"][fit]-1
    started=time.perf_counter(); p=x.probability(x.tabpfn(),xf,y,xp); elapsed=time.perf_counter()-started
    x.save_prediction(path,fit,pred,p,cats,elapsed,sha)
    print(name,"TabPFN",round(elapsed,1),"s",flush=True)
    return p

def _inner(ctx,fold,fit,nums,sha):
    tab=np.full((len(fit),5),np.nan); inc=np.full((len(fit),5),np.nan)
    for k in range(4):
        mask=nums==k; name=f"outer{fold}_inner{k}"
        tab[mask]=tab_prediction(ctx,name,fit[~mask],fit[mask],sha); inc[mask]=x.incumbent(name,fit[~mask],fit[mask])
    assert np.isfinite(tab).all() and np.isfinite(inc).all()
    return tab,inc

def _record(rows,oof,cal,ctx,fold,pred,arm,prob,threshold,rank_prob):
    prior=np.where(x.priority(prob,threshold)>0,1+9*(rank_prob@x.GAIN),0.)
    m=x.metric(ctx["labels"][pred],prior,ctx["ids"][pred]);m.update(x.diagnostics(ctx["labels"][pred],prob));m.update(outer_fold=fold,arm=arm,threshold=threshold)
    rows.append(m);cal.extend(x.calibration(ctx["labels"][pred],prob,fold,arm))
    for j,index in enumerate(pred): oof.append(dict(id=ctx["ids"][index],outer_fold=fold,arm=arm,Label=int(ctx["labels"][index]),selected=bool(prior[j]>0),priority=float(prior[j]),read_probability=float(1-prob[j,0]),expected_gain=float(rank_prob[j]@x.GAIN)))

def outer(ctx):
    protocol(ctx);sha=x.digest(x.OUT/"protocol.json");rows=[];choices=[];oof=[];cal=[]
    for fold,fit,pred,nums in x.jobs(ctx):
        tab_in,inc_in=_inner(ctx,fold,fit,nums,sha)
        tc,tr=x.choose(tab_in,ctx["labels"][fit],ctx["ids"][fit],nums);ic,ir=x.choose(inc_in,ctx["labels"][fit],ctx["ids"][fit],nums)
        choices += [dict(outer_fold=fold,arm="tabpfn",selected=r==tc,**r) for r in tr]+[dict(outer_fold=fold,arm="incumbent",selected=r==ic,**r) for r in ir]
        tab=tab_prediction(ctx,f"outer{fold}_validation",fit,pred,sha);inc=x.incumbent(f"outer{fold}_validation",fit,pred)
        _record(rows,oof,cal,ctx,fold,pred,"C0_deployed",inc,.55,inc)
        _record(rows,oof,cal,ctx,fold,pred,"C1_matched",inc,ic["threshold"],inc)
        _record(rows,oof,cal,ctx,fold,pred,"T1_primary",tab,tc["threshold"],tab)
        _record(rows,oof,cal,ctx,fold,pred,"T2_tab_gate_inc_rank",tab,tc["threshold"],inc)
        # T3 keeps the deployed incumbent selected set, but replaces ranking gain.
        prior=np.where(x.priority(inc,.55)>0,1+9*(tab@x.GAIN),0.);m=x.metric(ctx["labels"][pred],prior,ctx["ids"][pred]);m.update(x.diagnostics(ctx["labels"][pred],tab));m.update(outer_fold=fold,arm="T3_inc_gate_tab_rank",threshold=.55);rows.append(m);cal.extend(x.calibration(ctx["labels"][pred],tab,fold,"T3_inc_gate_tab_rank"))
        for j,index in enumerate(pred):oof.append(dict(id=ctx["ids"][index],outer_fold=fold,arm="T3_inc_gate_tab_rank",Label=int(ctx["labels"][index]),selected=bool(prior[j]>0),priority=float(prior[j]),read_probability=float(1-tab[j,0]),expected_gain=float(tab[j]@x.GAIN)))
        print("outer",fold,"tab_t",tc["threshold"],"H",round(rows[-3]["H"],6),flush=True)
    pd.DataFrame(rows).to_csv(x.OUT/"fold_metrics.csv",index=False);pd.DataFrame(choices).to_csv(x.OUT/"inner_thresholds.csv",index=False);pd.DataFrame(oof).to_csv(x.OUT/"outer_predictions.csv",index=False);pd.DataFrame(cal).to_csv(x.OUT/"calibration.csv",index=False)
    (x.OUT/"outer_run.json").write_text(json.dumps(dict(status="COMPLETE",protocol_sha256=sha,partitions=25,uploaded=False),indent=2)+"\n")

def main(stage):
    ctx=x.context()
    if stage=="audit":audit(ctx)
    elif stage=="outer":audit(ctx);outer(ctx)
    else:raise ValueError(stage)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["audit","outer"]);main(p.parse_args().stage)
