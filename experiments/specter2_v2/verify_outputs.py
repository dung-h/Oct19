#!/usr/bin/env python3
"""Check data repairs, nested-CV boundaries, scores, and saved-model predictions."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from prepare_data import OUT, ROOT, baseline, load_clean_data
from run_v2 import assemble, evaluate, make_priorities, predict_pool


def verify():
    data, template = load_clean_data()
    clean = pd.read_csv(OUT/"abstracts_cleaned.csv",dtype=str,keep_default_na=False)
    manifest=json.loads((OUT/"data_manifest.json").read_text())
    assert hashlib.sha256((ROOT/"abstracts_collected.csv").read_bytes()).hexdigest()==manifest["original_sha256"]
    assert hashlib.sha256((OUT/"abstracts_cleaned.csv").read_bytes()).hexdigest()==manifest["cleaned_sha256"]
    assert len(clean)==3090 and not clean.abstract.str.contains(r"\\documentclass|\\usepackage|\\setlength").any()
    rows=clean.set_index(["split","id"])
    assert rows.loc[("train","446"),"retrieved_doi"]=="10.4230/OASIcs.ICLP.2016.16"
    assert "Bug Report In this contribution" not in rows.loc[("train","175"),"abstract"]
    assert rows.loc[("train","392"),"abstract_usable_v2"]=="0"
    print("PASS: repaired sources, abstract boundaries, boilerplate and immutable baseline source")

    labeled=data[data.split=="train"].reset_index(drop=True)
    group_map=pd.Series(baseline.grouping(labeled),index=labeled.id)
    original_labels=labeled.set_index("id").Label.astype(int)
    oof=pd.read_csv(OUT/"nested_oof_predictions.csv",dtype={"id":str})
    assert len(oof)==2494 and oof.id.is_unique
    assert np.array_equal(oof.Label.astype(int),original_labels.loc[oof.id])
    splits=pd.read_csv(OUT/"nested_splits.csv",dtype={"id":str})
    for _,fold in splits.groupby("outer_fold"):
        assert len(fold)==2494 and fold.id.is_unique
        fit=fold[fold.partition=="outer_train"]
        validation=fold[fold.partition=="outer_validation"]
        assert set(group_map.loc[fit.id]).isdisjoint(group_map.loc[validation.id])
        for inner in range(4):
            assert set(group_map.loc[fit[fit.inner_fold!=inner].id]).isdisjoint(
                group_map.loc[fit[fit.inner_fold==inner].id])
    metrics=pd.read_csv(OUT/"nested_fold_metrics.csv")
    for row in metrics.itertuples():
        subset=oof[oof.outer_fold==row.outer_fold]
        priority=subset[row.family+"_priority"].to_numpy()
        assert np.isfinite(priority).all() and (priority>=0).all()
        score=evaluate(subset.Label,priority,subset.id)
        for key in ["score","relevance","coverage","FP","FN"]:
            assert abs(score[key]-getattr(row,key))<1e-9,(row.family,key)
    print("PASS: all nested outer/inner group boundaries, unchanged labels and independent score reproduction")

    raw=np.load(baseline.OUT/"embeddings.npz")
    repaired=np.load(OUT/"embeddings_clean.npz")
    embedding_manifest=json.loads((OUT/"embedding_manifest.json").read_text())
    assert embedding_manifest["input_sha256"]==baseline.input_digest(data)
    assert np.array_equal(raw["keys"],repaired["keys"])
    changed=np.isin(raw["keys"],embedding_manifest["reencoded_keys"])
    assert changed.sum()==16
    assert np.array_equal(raw["embeddings"][~changed],repaired["embeddings"][~changed])
    features={"raw":normalize(raw["embeddings"]),"clean":normalize(repaired["embeddings"])}
    indices=np.flatnonzero(data.split=="test")
    bundle=joblib.load(OUT/"model_v2.joblib")
    with threadpool_limits(limits=4):
        predictions=predict_pool(bundle,data,features,indices)
    gain,selection=assemble(bundle["selected"],predictions)
    priority=make_priorities(gain,selection,bundle["selected"]["threshold"])
    expected=pd.DataFrame({"id":data.id.iloc[indices],"priority":priority})
    actual=pd.read_csv(OUT/"submission_v2.csv",dtype={"id":str})
    assert actual.columns.tolist()==["id","priority"] and len(actual)==596 and actual.id.is_unique
    assert actual.id.tolist()==template.id.tolist()
    assert np.isfinite(actual.priority).all() and actual.priority.ge(0).all()
    joined=actual.merge(expected,on="id",validate="one_to_one")
    np.testing.assert_allclose(joined.priority_x,joined.priority_y,rtol=1e-6,atol=1e-5)
    assert np.array_equal(joined.priority_x>0,joined.priority_y>0)
    assert actual.sort_values(["priority","id"],ascending=[False,True]).id.tolist()==expected.sort_values(
        ["priority","id"],ascending=[False,True]).id.tolist()
    print("PASS: unchanged vectors reused, model reload, priorities/ranking and all 596 submission IDs")

    # Controlled data ablation: identical alpha=0.1, training sets, and selection method.
    # Thresholds come from each outer training partition's existing inner search.
    inner=pd.read_csv(OUT/"nested_inner_search.csv")
    train_indices=np.flatnonzero(data.split=="train")
    results=[]
    for fold in range(5):
        validation=oof.outer_fold.to_numpy()==fold
        fit_indices=train_indices[~validation]
        valid_indices=train_indices[validation]
        for space in ["raw","clean"]:
            chosen=inner[(inner.outer_fold==fold)&(inner.name==f"{space}_ridge_0.1")].iloc[0]
            with threadpool_limits(limits=4):
                model=Ridge(alpha=.1,solver="lsqr").fit(features[space][fit_indices],original_labels.loc[data.id.iloc[fit_indices]])
                gain=(model.predict(features[space][valid_indices])-1)/4
            priority=make_priorities(gain,gain,chosen.threshold)
            results.append({"outer_fold":fold,"space":space,"alpha":.1,"threshold":chosen.threshold,
                            **evaluate(original_labels.loc[data.id.iloc[valid_indices]],priority,data.id.iloc[valid_indices])})
    pd.DataFrame(results).to_csv(OUT/"controlled_data_ablation.csv",index=False)
    print("Controlled data-only ablation:",pd.DataFrame(results).groupby("space").score.mean().to_dict())
    print("All verification checks passed")


if __name__=="__main__":verify()
