#!/usr/bin/env python3
"""Evaluate text repairs and read-selection models using grouped nested CV."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from prepare_data import OUT, ROOT, baseline, load_clean_data

SEED = 20261015
RAW_ALPHAS = [0.1, 1.0, 10.0, 100.0]
CLEAN_ALPHAS = [0.03, 0.1, 0.3, 1.0]
TFIDF_ALPHAS = [0.3, 1.0, 3.0]


def evaluate(labels, priority, ids):
    labels, priority, ids = np.asarray(labels), np.asarray(priority), np.asarray(ids, dtype=str)
    selected = np.flatnonzero(priority > 0)
    order = selected[np.lexsort((ids[selected], -priority[selected]))]
    positives = labels > 1
    tp = int(positives[order].sum())
    fp, fn = len(order) - tp, int(positives.sum()) - tp
    if len(order):
        weight = 1 / np.log2(np.arange(len(order)) + 2)
        q = float(np.average((labels[order] - 1) / 4, weights=weight))
    else:
        q = 0.0
    r = float(tp / positives.sum()) if positives.sum() else 0.0
    return {"score": 2*q*r/(q+r) if q+r else 0.0, "relevance": q, "coverage": r,
            "selected": len(order), "TP": tp, "FP": fp, "FN": fn}


def configurations():
    configs = []
    for family, prefix, alphas, legacy in [
        ("raw_specter2_ridge", "raw", RAW_ALPHAS, True),
        ("clean_specter2_ridge", "clean", CLEAN_ALPHAS, False),
        ("clean_tfidf_ridge", "tfidf", TFIDF_ALPHAS, False),
    ]:
        for alpha in alphas:
            configs.append({"name": f"{prefix}_ridge_{alpha}", "family": family,
                            "kind": "ridge", "model": f"{prefix}_ridge_{alpha}", "legacy": legacy})
    for weight in [.25, .5, .75]:
        configs.append({"name": f"hybrid_ridge_{weight}", "family": "hybrid_ridge",
                        "kind": "hybrid_ridge", "weight": weight, "legacy": False})
    for space, cs in [("clean", [10.0, 100.0]), ("tfidf", [1.0, 10.0])]:
        for c in cs:
            configs.append({"name": f"{space}_two_stage_{c}", "family": f"{space}_two_stage",
                            "kind": "two_stage", "space": space, "C": c, "legacy": False})
    for weight in [.25, .5, .75]:
        configs.append({"name": f"hybrid_two_stage_{weight}", "family": "hybrid_two_stage",
                        "kind": "hybrid_two_stage", "weight": weight, "legacy": False})
    return configs


def assemble(config, predictions):
    if config["kind"] == "ridge":
        gain = (predictions[config["model"]] - 1) / 4
        return gain, gain
    if config["kind"] == "hybrid_ridge":
        w = config["weight"]
        gain = (w*predictions["clean_ridge_0.1"] + (1-w)*predictions["tfidf_ridge_1.0"] - 1) / 4
        return gain, gain
    if config["kind"] == "two_stage":
        space = config["space"]
        probability = predictions[f"{space}_prob_{config['C']}"]
        conditional_gain = np.clip(predictions[f"{space}_conditional_gain"], .25, 1)
        return probability*conditional_gain, probability
    w = config["weight"]
    pe, pt = predictions["clean_prob_100.0"], predictions["tfidf_prob_10.0"]
    ge = pe*np.clip(predictions["clean_conditional_gain"], .25, 1)
    gt = pt*np.clip(predictions["tfidf_conditional_gain"], .25, 1)
    return w*ge + (1-w)*gt, w*pe + (1-w)*pt


def make_priorities(gain, selection_value, threshold):
    gain, selection_value = np.asarray(gain, dtype=np.float64), np.asarray(selection_value)
    return np.where((selection_value > threshold) & (gain > 0), gain*10, 0)


def prepare_embeddings(data):
    """Reuse original vectors verbatim for unchanged inputs; re-encode repairs only."""
    import torch
    from adapters import AutoAdapterModel
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = OUT / "embeddings_clean.npz"
    manifest_path = OUT / "embedding_manifest.json"
    digest = baseline.input_digest(data)
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["input_sha256"] == digest, "Clean embedding cache is stale"
        return
    original, _ = baseline.load_data()
    original_manifest = json.loads((baseline.OUT / "embedding_manifest.json").read_text())
    assert original_manifest["input_sha256"] == baseline.input_digest(original)
    cached = np.load(baseline.OUT / "embeddings.npz")
    keys = (data.split + ":" + data.id).to_numpy(dtype=str)
    assert np.array_equal(keys, cached["keys"])
    changed = np.flatnonzero((data.title != original.title) | (data.abstract != original.abstract))
    vectors = cached["embeddings"].copy()
    tokenizer = AutoTokenizer.from_pretrained(baseline.BASE, revision=baseline.BASE_REV)
    base_path = snapshot_download(baseline.BASE, revision=baseline.BASE_REV, allow_patterns=["*.bin", "*.json", "vocab.txt"])
    adapter_path = snapshot_download(baseline.ADAPTER, revision=baseline.ADAPTER_REV, allow_patterns=["*.bin", "*.json"])
    model = AutoAdapterModel.from_pretrained(base_path)
    model.load_adapter(adapter_path, load_as="regression", set_active=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval().requires_grad_(False)
    torch.set_num_threads(4)
    text = (data.title + tokenizer.sep_token + data.abstract).tolist()
    for start in range(0, len(changed), 8):
        indices = changed[start:start+8]
        batch = tokenizer([text[i] for i in indices], padding=True, truncation=True,
                          max_length=512, return_token_type_ids=False, return_tensors="pt")
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=device == "cuda", dtype=torch.float16):
            vectors[indices] = model(**batch).last_hidden_state[:,0,:].float().cpu().numpy()
        print(f"Re-encoded {min(start+8,len(changed))}/{len(changed)} repaired inputs", flush=True)
    assert vectors.shape == (3090, 768) and np.isfinite(vectors).all()
    lengths = [len(tokens) for tokens in tokenizer(text, truncation=False)["input_ids"]]
    np.savez_compressed(path, embeddings=vectors, keys=keys)
    manifest = {**original_manifest, "input_sha256": digest, "device": device,
                "reencoded_rows": len(changed), "reencoded_keys": keys[changed].tolist(),
                "unchanged_vectors_reused_verbatim": len(data)-len(changed),
                "truncated_rows": int(sum(length > 512 for length in lengths)),
                "title_only_rows": int(data.abstract.eq("").sum())}
    manifest["baseline_encoding_elapsed_seconds"] = manifest.pop("elapsed_seconds", None)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def fit_pool(data, features, labels, fit_indices, prediction_indices):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98,
                                max_features=50000, sublinear_tf=True)
    text_fit = vectorizer.fit_transform(data.text.iloc[fit_indices])
    text_pred = vectorizer.transform(data.text.iloc[prediction_indices])
    spaces = {"raw": (features["raw"][fit_indices], features["raw"][prediction_indices]),
              "clean": (features["clean"][fit_indices], features["clean"][prediction_indices]),
              "tfidf": (text_fit, text_pred)}
    models, predictions = {}, {}
    y = labels[fit_indices]
    positive = y > 1
    for space, alphas in [("raw", RAW_ALPHAS), ("clean", CLEAN_ALPHAS), ("tfidf", TFIDF_ALPHAS)]:
        fit_x, pred_x = spaces[space]
        for alpha in alphas:
            key = f"{space}_ridge_{alpha}"
            model = Ridge(alpha=alpha, solver="lsqr").fit(fit_x, y)
            models[key] = model; predictions[key] = model.predict(pred_x)
    for space, cs, alpha in [("clean", [10.0,100.0], .1), ("tfidf", [1.0,10.0], 1.0)]:
        fit_x, pred_x = spaces[space]
        for c in cs:
            key = f"{space}_prob_{c}"
            model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs").fit(fit_x, positive)
            models[key] = model; predictions[key] = model.predict_proba(pred_x)[:,1]
        key = f"{space}_conditional_gain"
        model = Ridge(alpha=alpha, solver="lsqr").fit(fit_x[positive], (y[positive]-1)/4)
        models[key] = model; predictions[key] = model.predict(pred_x)
    return predictions, {"vectorizer": vectorizer, "models": models}


def predict_pool(bundle, data, features, indices):
    text = bundle["vectorizer"].transform(data.text.iloc[indices])
    spaces = {"raw": features["raw"][indices], "clean": features["clean"][indices], "tfidf": text}
    predictions = {}
    for key, model in bundle["models"].items():
        x = spaces[key.split("_", 1)[0]]
        predictions[key] = model.predict_proba(x)[:,1] if "_prob_" in key else model.predict(x)
    return predictions


def inner_select(data, features, labels, indices, groups, seed):
    configs = configurations()
    folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(
        np.zeros(len(indices)), labels[indices], groups))
    oof, fold_numbers = {}, np.full(len(indices), -1)
    for fold, (fit_idx, val_idx) in enumerate(folds):
        assert set(groups[fit_idx]).isdisjoint(groups[val_idx])
        predictions, _ = fit_pool(data, features, labels, indices[fit_idx], indices[val_idx])
        for key, values in predictions.items():
            oof.setdefault(key, np.full(len(indices), np.nan))[val_idx] = values
        fold_numbers[val_idx] = fold
    rows = []
    for config in configs:
        gain, selection = assemble(config, oof)
        assert np.isfinite(gain).all() and np.isfinite(selection).all()
        thresholds = np.arange(0, .626, .025) if config["kind"] in ["ridge", "hybrid_ridge"] else np.arange(0, .901, .05)
        best = None
        for threshold in thresholds:
            priority = make_priorities(gain, selection, threshold)
            scores = [evaluate(labels[indices[v]], priority[v], data.id.iloc[indices[v]])["score"] for _,v in folds]
            row = {**config, "threshold": float(threshold), "inner_score": float(np.mean(scores)),
                   "inner_std": float(np.std(scores))}
            if best is None or row["inner_score"] > best["inner_score"]:
                best = row
        rows.append(best)
    rows.sort(key=lambda r:r["inner_score"], reverse=True)
    family_best = {family:next(r for r in rows if r["family"] == family) for family in dict.fromkeys(r["family"] for r in rows)}
    winner = next(row for row in rows if not row["legacy"])
    return winner, family_best, rows, fold_numbers


def run(data, template):
    original_cache = np.load(baseline.OUT / "embeddings.npz")
    repaired_cache = np.load(OUT / "embeddings_clean.npz")
    assert np.array_equal(original_cache["keys"], repaired_cache["keys"])
    features = {"raw": normalize(original_cache["embeddings"]), "clean": normalize(repaired_cache["embeddings"])}
    train_indices = np.flatnonzero(data.split.eq("train"))
    test_indices = np.flatnonzero(data.split.eq("test"))
    labels = np.full(len(data), np.nan)
    labels[train_indices] = data.Label.iloc[train_indices].astype(int)
    train_meta = data.iloc[train_indices].reset_index(drop=True)
    groups = baseline.grouping(train_meta)
    configs = configurations()
    protocol = {"seed": SEED, "outer_folds":5, "inner_folds":4, "configurations":configs,
                "selection": "maximize mean inner-fold score; legacy baseline excluded from new-model selection",
                "data_sha256": baseline.input_digest(data), "clean_data_manifest": json.loads((OUT/"data_manifest.json").read_text())}
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2))
    outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        train_meta, labels[train_indices], groups))
    results, inner_results, split_rows = [], [], []
    oof = train_meta[["id", "title", "venue", "year", "Label"]].copy()
    oof["outer_fold"] = -1
    started = time.monotonic()
    for fold, (fit_idx, val_idx) in enumerate(outer):
        assert set(groups[fit_idx]).isdisjoint(groups[val_idx])
        print(f"OUTER {fold+1}/5: fit={len(fit_idx)} validation={len(val_idx)}", flush=True)
        selected, family_best, rows, fold_numbers = inner_select(
            data, features, labels, train_indices[fit_idx], groups[fit_idx], SEED+fold+1)
        inner_results.extend({"outer_fold":fold, **row} for row in rows)
        split_rows.extend({"outer_fold":fold,"id":train_meta.id.iloc[idx],"partition":"outer_train", "inner_fold":int(inner)}
                          for idx,inner in zip(fit_idx,fold_numbers))
        split_rows.extend({"outer_fold":fold,"id":train_meta.id.iloc[idx],"partition":"outer_validation", "inner_fold":-1}
                          for idx in val_idx)
        predictions, _ = fit_pool(data, features, labels, train_indices[fit_idx], train_indices[val_idx])
        choices = {**family_best, "selection_procedure":selected}
        for family, chosen in choices.items():
            gain, selection_value = assemble(chosen, predictions)
            priority = make_priorities(gain, selection_value, chosen["threshold"])
            metrics = evaluate(labels[train_indices[val_idx]], priority, data.id.iloc[train_indices[val_idx]])
            results.append({"outer_fold":fold,"family":family,"chosen_config":chosen["name"],
                            "threshold":chosen["threshold"],"inner_score":chosen["inner_score"], **metrics})
            oof.loc[val_idx, family+"_gain"] = gain
            oof.loc[val_idx, family+"_selection_value"] = selection_value
            oof.loc[val_idx, family+"_priority"] = priority
        oof.loc[val_idx,"outer_fold"] = fold
        print(f"  chosen={selected['name']} | inner={selected['inner_score']:.5f} | outer={results[-1]['score']:.5f}; elapsed={time.monotonic()-started:.1f}s",flush=True)
        pd.DataFrame(results).to_csv(OUT/"nested_fold_metrics.csv",index=False)
        oof.to_csv(OUT/"nested_oof_predictions.csv",index=False)
    pd.DataFrame(split_rows).to_csv(OUT/"nested_splits.csv",index=False)
    pd.DataFrame(inner_results).to_csv(OUT/"nested_inner_search.csv",index=False)
    fold_metrics = pd.DataFrame(results)
    summary = fold_metrics.groupby("family").agg(score_mean=("score","mean"),score_std=("score","std"),
        relevance_mean=("relevance","mean"),coverage_mean=("coverage","mean"),FP=("FP","sum"),FN=("FN","sum"),TP=("TP","sum"))
    summary.to_csv(OUT/"nested_summary.csv")
    print(summary.to_string(),flush=True)
    print("Selecting final configuration using inner OOF on all training labels",flush=True)
    selected, final_families, final_rows, _ = inner_select(data, features, labels, train_indices, groups, SEED+101)
    pd.DataFrame(final_rows).to_csv(OUT/"final_inner_search.csv",index=False)
    predictions, bundle = fit_pool(data, features, labels, train_indices, test_indices)
    gain, selection_value = assemble(selected, predictions)
    priority = make_priorities(gain, selection_value, selected["threshold"])
    submission = template[["id"]].merge(pd.DataFrame({"id":data.id.iloc[test_indices],"priority":priority}),
                                         on="id",how="left",validate="one_to_one")
    assert len(submission)==596 and submission.id.is_unique
    assert np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
    submission.to_csv(OUT/"submission_v2.csv",index=False)
    bundle.update(selected=selected,protocol=protocol,embedding_manifest=json.loads((OUT/"embedding_manifest.json").read_text()))
    joblib.dump(bundle,OUT/"model_v2.joblib")
    pd.DataFrame({"id":data.id.iloc[test_indices],"expected_gain":gain,"selection_value":selection_value,
                  "threshold":selected["threshold"],"priority":priority}).to_csv(OUT/"test_prediction_details.csv",index=False)
    final = {"selected":selected,"nested_summary":summary.reset_index().to_dict("records"),
             "test_selected":int(submission.priority.gt(0).sum()), "elapsed_seconds":time.monotonic()-started,
             "all_original_labels_preserved":True,"not_submitted_to_kaggle":True}
    (OUT/"results.json").write_text(json.dumps(final,indent=2))
    print("FINAL",json.dumps(final,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["all","embed","train"],nargs="?",default="all")
    args=parser.parse_args()
    data,template=load_clean_data()
    if args.stage in ["all","embed"]:prepare_embeddings(data)
    manifest=json.loads((OUT/"embedding_manifest.json").read_text())
    assert manifest["input_sha256"]==baseline.input_digest(data)
    if args.stage in ["all","train"]:
        with threadpool_limits(limits=4):run(data,template)


if __name__=="__main__":main()
