"""Independent lineage, feature, metric, model-reload and CSV verification."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from apm_context import (ARMS, N_FIT, N_TOTAL, NEW_ARMS, OUT, PRIMARY,
                         READ_THRESHOLD, context, digest, final_pack_path,
                         load_source_pack, outer_jobs, source,
                         source_probabilities)
from apm_features import build_partition_features, feature_names
from apm_metrics import official_metrics
from apm_runner import add_features, estimator, write_protocol


def _independent_features(ctx, query, refs, arm):
    """Reference implementation by direct per-query/per-class sorting."""
    names = feature_names()
    result = {name:0.0 for name in names}
    if not ctx["abstract_available"][query]:
        return np.zeros(43,dtype=np.float32)
    qvec = ctx["spaces"][arm][query]
    if np.linalg.norm(qvec) == 0:
        return np.zeros(43,dtype=np.float32)
    allowed = [int(ref) for ref in refs
               if ctx["abstract_available"][ref]
               and np.linalg.norm(ctx["spaces"][arm][ref]) > 0
               and ctx["groups"][ref] != ctx["groups"][query]]
    similarities = {ref:float(qvec @ ctx["spaces"][arm][ref]) for ref in allowed}
    labels = {"N":{1},"P":{2,3,4,5},"G":{4,5}}
    stats_by_scope = {}
    for scope in ["all","venue"]:
        in_scope = [ref for ref in allowed if scope=="all" or
                    ctx["venues"][ref]==ctx["venues"][query]]
        group_stats={}
        for group, accepted in labels.items():
            members=[ref for ref in in_scope if int(ctx["labels"][ref]) in accepted]
            members.sort(key=lambda ref:(-similarities[ref],str(ctx["ids"][ref])))
            values=[similarities[ref] for ref in members]
            if values:
                stat=dict(max=values[0],mean5=float(np.mean(values[:5])),
                          mean15=float(np.mean(values[:15])),support15=min(len(values),15)/15,
                          present=1.0)
            else:
                stat=dict(max=0.0,mean5=0.0,mean15=0.0,support15=0.0,present=0.0)
            group_stats[group]=stat
            for key,value in stat.items():
                result[f"{scope}_{group}_{key}"]=value
        for left,right in [("P","N"),("G","N")]:
            for key in ["max","mean5","mean15"]:
                result[f"{scope}_{left}minus{right}_{key}"]=(
                    group_stats[left][key]-group_stats[right][key]
                    if group_stats[left]["present"] and group_stats[right]["present"] else 0.0)
        stats_by_scope[scope]=in_scope
    result["query_abstract_available"]=1.0
    return np.asarray([result[name] for name in names],dtype=np.float32)


def _audit_independent_features(ctx, name, fit, pred, feature_pack):
    source_pack, _, _ = load_source_pack(name,fit,pred)
    rng=np.random.default_rng(20261018+sum(map(ord,name)))
    checked=0
    train_new=feature_pack["train_new"]
    pred_new=feature_pack["pred_new"]
    fit_position={int(index):pos for pos,index in enumerate(fit)}
    pred_position={int(index):pos for pos,index in enumerate(pred)}
    for view_index, entry in enumerate(source_pack["lineage"]):
        refs=np.asarray(entry["refs"],int)
        queries=np.asarray(entry["queries"],int)
        train_sample=list(queries[:2])
        valid_sample=list(pred[:min(3,len(pred))])
        if len(queries)>2:
            train_sample += list(rng.choice(queries[2:],size=min(2,len(queries)-2),replace=False))
        if len(pred)>3:
            valid_sample += list(rng.choice(pred[3:],size=min(3,len(pred)-3),replace=False))
        for arm_index,arm in enumerate(NEW_ARMS):
            for query in train_sample:
                expected=_independent_features(ctx,int(query),refs,arm)
                actual=train_new[arm_index,fit_position[int(query)]]
                np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
                checked+=1
            for query in valid_sample:
                expected=_independent_features(ctx,int(query),refs,arm)
                actual=pred_new[arm_index,view_index,pred_position[int(query)]]
                np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
                checked+=1
                # Changing a query's label cannot alter any feature for it.
                if query < N_FIT:
                    from apm_features import features_for_queries
                    changed=dict(ctx)
                    changed["labels"]=ctx["labels"].copy()
                    changed["labels"][int(query)]=1+(int(changed["labels"][int(query)])%5)
                    before,_,_=features_for_queries(ctx,[int(query)],refs,arm)
                    after,_,_=features_for_queries(changed,[int(query)],refs,arm)
                    np.testing.assert_array_equal(before,after)
    return checked


def _independent_metrics(labels, probability, incumbent, ids):
    labels=np.asarray(labels,int);probability=np.asarray(probability,float)
    read=1.0-probability[:,0]
    selected=read>READ_THRESHOLD
    gain=incumbent@ (np.arange(5,dtype=float)/4)
    priority=np.where(selected,1+9*gain,0.0)
    positive=labels>1
    tp=int((selected&positive).sum());fp=int((selected&(labels==1)).sum())
    fn=int((~selected&positive).sum());tn=int((~selected&(labels==1)).sum())
    order=sorted(np.flatnonzero(selected),key=lambda ix:(-priority[ix],str(ids[ix])))
    discounts=1/np.log2(np.arange(len(order),dtype=float)+2)
    q=float(np.sum(discounts*((labels[order]-1)/4))/discounts.sum()) if order else 0.0
    r=tp/int(positive.sum()) if positive.any() else 0.0
    return dict(accepted=int(selected.sum()),dropped=int((~selected).sum()),TP=tp,FP=fp,FN=fn,TN=tn,
        total_error=fp+fn,missed_high=int(((labels>=4)&~selected).sum()),
        missed_label4=int(((labels==4)&~selected).sum()),missed_label5=int(((labels==5)&~selected).sum()),
        Q=q,R=r,H=2*q*r/(q+r) if q+r else 0.0)


def _verify_outer(ctx, protocol_sha):
    metric_rows=[];models_reloaded=0;feature_samples=0;prediction_replays=0
    for fold,fit,pred,_ in outer_jobs(ctx):
        name=f"outer{fold}_validation"
        pack,xfit,xpred=load_source_pack(name,fit,pred)
        path=OUT/f"{name}_margin_features.joblib"
        margin=joblib.load(path)
        assert margin["protocol_sha256"]==protocol_sha
        np.testing.assert_array_equal(margin["fit"],fit);np.testing.assert_array_equal(margin["pred"],pred)
        feature_samples += _audit_independent_features(ctx,name,fit,pred,margin)
        with np.load(OUT/f"{name}_predictions.npz") as z:
            np.testing.assert_array_equal(z["fit"],fit);np.testing.assert_array_equal(z["pred"],pred)
            assert str(z["protocol_sha256"].item())==protocol_sha
            probabilities={arm:z[arm].copy() for arm in ARMS}
        base=source_probabilities(name,fit,pred,ctx)
        np.testing.assert_array_equal(probabilities["reference"],base)
        y=ctx["labels"][pred];ids=ctx["ids"][pred]
        for arm in ARMS:
            model=estimator()
            model.load_model(str(OUT/f"{name}_{arm}.cbm"))
            if arm=="reference":
                replay=model.predict_proba(xpred)
                np.testing.assert_allclose(replay,base,atol=1e-7,rtol=1e-6)
            else:
                arm_index=NEW_ARMS.index(arm)
                train_frame=add_features(xfit,margin["train_new"][arm_index])
                prediction_views=[]
                for view in range(margin["pred_new"].shape[1]):
                    frame=add_features(xpred,margin["pred_new"][arm_index,view])
                    prediction_views.append(model.predict_proba(frame))
                replay=np.mean(prediction_views,axis=0)
                missing=~ctx["abstract_available"][pred]
                replay[missing]=base[missing]
                np.testing.assert_allclose(replay,probabilities[arm],atol=1e-8,rtol=1e-7)
            models_reloaded+=1;prediction_replays+=1
            metric=_independent_metrics(y,probabilities[arm],base,ids)
            metric_rows.append(dict(outer_fold=int(fold),arm=arm,**metric))
            assert np.isfinite(probabilities[arm]).all()
            np.testing.assert_allclose(probabilities[arm].sum(axis=1),1,atol=1e-6)
    independent=pd.DataFrame(metric_rows)
    independent.to_csv(OUT/"independent_fold_metrics.csv",index=False)
    assert len(independent)==20
    reference=independent[independent.arm.eq("reference")]
    assert int(reference.accepted.sum())==1616
    assert int(reference.FP.sum())==338 and int(reference.FN.sum())==313
    assert int(reference.missed_high.sum())==38
    assert abs(float(reference.H.mean())-0.6402759969159176)<1e-10
    return dict(outer_metric_rows=len(independent),models_reloaded=models_reloaded,
                outer_prediction_replays=prediction_replays,independent_feature_values_checked=feature_samples,
                incumbent_metrics_reproduced=True)


def _verify_final(ctx,protocol_sha):
    name="final_test";fit=np.arange(N_FIT,dtype=int);pred=np.arange(N_FIT,N_TOTAL,dtype=int)
    pack,xfit,xpred=load_source_pack(name,fit,pred)
    margin=joblib.load(OUT/f"{name}_margin_features.joblib")
    assert margin["protocol_sha256"]==protocol_sha
    np.testing.assert_array_equal(margin["fit"],fit);np.testing.assert_array_equal(margin["pred"],pred)
    with np.load(OUT/f"{name}_predictions.npz") as z:
        np.testing.assert_array_equal(z["fit"],fit);np.testing.assert_array_equal(z["pred"],pred)
        probabilities={arm:z[arm].copy() for arm in ARMS}
    base=source_probabilities(name,fit,pred,ctx)
    np.testing.assert_array_equal(probabilities["reference"],base)
    reloaded=0
    for arm in ARMS:
        model=estimator();model.load_model(str(OUT/f"{name}_{arm}.cbm"))
        if arm=="reference":
            replay=model.predict_proba(xpred)
            np.testing.assert_allclose(replay,base,atol=1e-7,rtol=1e-6)
        else:
            arm_index=NEW_ARMS.index(arm)
            train_frame=add_features(xfit,margin["train_new"][arm_index])
            views=[model.predict_proba(add_features(xpred,margin["pred_new"][arm_index,v]))
                   for v in range(margin["pred_new"].shape[1])]
            replay=np.mean(views,axis=0)
            missing=~ctx["abstract_available"][pred]
            replay[missing]=base[missing]
            np.testing.assert_allclose(replay,probabilities[arm],atol=1e-8,rtol=1e-7)
        reloaded+=1
        sub=pd.read_csv(OUT/f"submission_{arm}.csv",dtype={"id":str})
        expected_ids=ctx["template"].id.astype(str).tolist()
        assert sub.columns.tolist()==["id","priority"]
        assert sub.id.tolist()==expected_ids and sub.id.is_unique and len(sub)==596
        assert np.isfinite(sub.priority).all() and (sub.priority>=0).all()
    candidate=(OUT/"submission_candidate.csv").read_bytes()
    assert candidate==(OUT/f"submission_{PRIMARY}.csv").read_bytes()
    return dict(final_models_reloaded=reloaded,final_submissions_checked=4,
                test_ids_checked=596,test_labels_used=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(4):
        ctx=context()
        assert json.loads((OUT/"outer_run.json").read_text())["status"]=="COMPLETE"
        protocol=write_protocol(ctx);protocol_sha=digest(OUT/"protocol.json")
        outer=_verify_outer(ctx,protocol_sha)
        result=dict(status="PASSED",protocol_sha256=protocol_sha,**outer,
                    incumbent_sha256=digest(source.INC/"submission_candidate.csv"),
                    incumbent_unchanged=True,uploaded=False)
        if (OUT/"final_test_predictions.npz").exists():
            result.update(_verify_final(ctx,protocol_sha))
        (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
