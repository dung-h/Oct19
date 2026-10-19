#!/usr/bin/env python3
"""Run the locked nested-disagreement-gate experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import ndg_lib as n


SOURCE_FILES = ["PLAN.md", "ndg_lib.py", "run.py", "verify.py", "report.py", "test_ndg.py"]


def _paths(ctx):
    paths = [n.PROJECT / "assignment-data-mining-261.zip", n.EXPERIMENTS / "specter2_v2/outputs/nested_splits.csv",
             n.EXPERIMENTS / "semantic_roles/outputs/embeddings.npz", n.EXPERIMENTS / "semantic_roles/outputs/texts.csv"]
    for fold, fit, pred, nums in n.outer_jobs(ctx):
        paths += [n.ac.source.pack_path(f"outer{fold}_validation"), n.ac.source.CLAIM / f"outer{fold}_validation_predictions.npz",
                  n.EXPERIMENTS / "specter2_preference/outputs" / f"outer{fold}_validation_adapted.npz"]
        for k in sorted(np.unique(nums)):
            paths += [n.ac.source.pack_path(f"outer{fold}_inner{k}"), n.ac.source.CLAIM / f"outer{fold}_inner{k}_predictions.npz",
                      n.EXPERIMENTS / "specter2_preference/outputs" / f"outer{fold}_inner{k}_adapted.npz"]
    assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
    return paths


def protocol(ctx):
    inputs = _paths(ctx)
    record = dict(
        plan_sha256=n.digest(n.HERE / "PLAN.md"),
        source_sha256={name: n.digest(n.HERE / name) for name in SOURCE_FILES},
        input_sha256={str(p.relative_to(n.PROJECT)): n.digest(p) for p in inputs},
        label_sha256=hashlib.sha256(ctx["labels"].astype(np.int8).tobytes()).hexdigest(),
        group_sha256=hashlib.sha256(ctx["groups"][:n.N_FIT].astype(np.int64).tobytes()).hexdigest(),
        sources=list(n.SOURCE_NAMES), neural_epoch=8,
        meta=dict(features=["incumbent_read_logit", "incumbent_gain", "title_read_logit", "title_gain", "neural_read_logit", "neural_gain",
                            "title_minus_incumbent_logit", "neural_minus_incumbent_logit", "neural_minus_title_logit"],
                  scaler="StandardScaler", model="LogisticRegression", penalty="l2", C=.2, solver="lbfgs", seed=n.SEED),
        thresholds=n.THRESHOLDS.tolist(), tie_rule="higher threshold", ranking="incumbent expected gain", outer_folds=5, inner_folds=4,
        title_model=dict(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8, seed=n.SEED),
        cached_sources=dict(incumbent="claim_stability nested cache; CatBoost parity replayed locally", neural_adapted="specter2_preference epoch8 nested cache; IDs/fit/pred verified"),
        source_refits=dict(incumbent=25, title_margin=25, neural_adapted=0), uploaded=False,
        packages={k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scikit-learn", "catboost", "joblib"]},
    )
    path = n.OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == record, "Locked protocol or input changed"
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def audit(ctx):
    n.OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(n.HERE / "test_ndg.py")], cwd=n.HERE, check=True)
    for fold, fit, pred, nums in n.outer_jobs(ctx):
        assert set(ctx["groups"][fit]).isdisjoint(ctx["groups"][pred])
        assert set(np.unique(nums)) == {0, 1, 2, 3}
        for k in range(4):
            a, b = fit[nums != k], fit[nums == k]
            assert set(ctx["groups"][a]).isdisjoint(ctx["groups"][b])
            n.incumbent_cache(f"outer{fold}_inner{k}", a, b)
            n.neural_cache(f"outer{fold}_inner{k}", a, b)
        n.incumbent_cache(f"outer{fold}_validation", fit, pred)
        n.neural_cache(f"outer{fold}_validation", fit, pred)
    locked = protocol(ctx)
    result = dict(status="PASSED", outer_folds=5, inner_partitions=20, sources=list(n.SOURCE_NAMES),
                  rows=n.N_FIT, protocol_sha256=n.digest(n.OUT / "protocol.json"), source_cache_alignment=True, uploaded=False)
    (n.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return locked


def _build_inner(ctx, fold, fit, nums, protocol_sha):
    result = {name: np.full((len(fit), 5), np.nan, dtype=np.float64) for name in n.SOURCE_NAMES}
    records = []
    for k in range(4):
        mask = nums == k; source = n.title_source(ctx, f"outer{fold}_inner{k}", fit[~mask], fit[mask], protocol_sha)
        for arm in n.SOURCE_NAMES:
            result[arm][mask] = source[arm]
        records.append(dict(inner_fold=k, n_fit=int((~mask).sum()), n_pred=int(mask.sum()),
                            fit_sha256=hashlib.sha256(np.ascontiguousarray(fit[~mask]).view(np.uint8)).hexdigest(),
                            pred_sha256=hashlib.sha256(np.ascontiguousarray(fit[mask]).view(np.uint8)).hexdigest()))
    assert all(np.isfinite(value).all() for value in result.values())
    return result, records


def _crossfit_meta(x, y, nums, fold):
    pred = np.full(len(x), np.nan, dtype=float); recipes = []
    for k in range(4):
        mask = nums == k
        recipe = n.fit_meta(x[~mask], y[~mask])
        pred[mask] = n.meta_probability(recipe, x[mask])
        joblib.dump(dict(scaler=recipe[0], model=recipe[1], train_positions=np.flatnonzero(~mask), pred_positions=np.flatnonzero(mask)),
                    n.OUT / f"outer{fold}_meta_inner{k}.joblib")
        recipes.append(dict(inner_fold=k, n_train=int((~mask).sum()), n_pred=int(mask.sum())))
    assert np.isfinite(pred).all()
    return pred, recipes


def _outer_metrics(ctx, fold, pred, incumbent, read, threshold):
    labels = ctx["labels"][pred]; ids = ctx["ids"][pred]
    p = n.priority(read, incumbent, threshold)
    row = n.metrics(labels, p, ids)
    row.update(outer_fold=int(fold), threshold=float(threshold), read_AUC=n.read_auc(labels, read),
               Brier=float(np.mean((read - (labels > 1)) ** 2)), priority=p)
    return row


def run_outer(ctx):
    record = protocol(ctx); protocol_sha = n.digest(n.OUT / "protocol.json")
    metric_rows, threshold_rows, oof_rows, calibration, lineage = [], [], [], [], []
    for fold, fit, pred, nums in n.outer_jobs(ctx):
        inner, rows = _build_inner(ctx, fold, fit, nums, protocol_sha)
        lineage.extend(dict(outer_fold=fold, partition="inner", **row) for row in rows)
        x_inner = n.meta_features(inner); y_inner = ctx["labels"][fit] > 1
        meta_inner, recipes = _crossfit_meta(x_inner, y_inner, nums, fold)
        choice, candidates = n.choose_threshold(meta_inner, inner["incumbent"], ctx["labels"][fit], ctx["ids"][fit], nums)
        threshold_rows.extend(dict(outer_fold=fold, selected=bool(row == choice), **row) for row in candidates)
        sources = n.title_source(ctx, f"outer{fold}_validation", fit, pred, protocol_sha)
        x_outer = n.meta_features(sources)
        outer_recipe = n.fit_meta(x_inner, y_inner)
        joblib.dump(dict(scaler=outer_recipe[0], model=outer_recipe[1], train_indices=fit, pred_indices=pred), n.OUT / f"outer{fold}_meta_final.joblib")
        read = n.meta_probability(outer_recipe, x_outer)
        meta = _outer_metrics(ctx, fold, pred, sources["incumbent"], read, choice["threshold"])
        baseline_read = 1 - sources["incumbent"][:, 0]
        baseline = _outer_metrics(ctx, fold, pred, sources["incumbent"], baseline_read, .55)
        for arm, result in [("incumbent", baseline), ("meta_gate", meta)]:
            priority = result.pop("priority")
            metric_rows.append(dict(arm=arm, **result))
            for pos, index in enumerate(pred):
                oof_rows.append(dict(id=ctx["ids"][index], outer_fold=fold, arm=arm, Label=int(ctx["labels"][index]),
                                     title=ctx["data"].title.iloc[index], venue=ctx["venues"][index], selected=bool(priority[pos] > 0),
                                     priority=float(priority[pos]), read_probability=float(read[pos] if arm == "meta_gate" else baseline_read[pos]),
                                     incumbent_read=float(baseline_read[pos]), title_read=float(1-sources["title_margin"][pos,0]),
                                     neural_read=float(1-sources["neural_adapted"][pos,0])))
            if arm == "meta_gate": calibration.extend(n.calibration_rows(ctx["labels"][pred], read, fold))
        lineage.append(dict(outer_fold=fold, partition="outer", n_fit=len(fit), n_pred=len(pred),
                            threshold=choice["threshold"], meta_inner_recipes=recipes,
                            fit_sha256=hashlib.sha256(np.ascontiguousarray(fit).view(np.uint8)).hexdigest(),
                            pred_sha256=hashlib.sha256(np.ascontiguousarray(pred).view(np.uint8)).hexdigest()))
        pd.DataFrame(metric_rows).to_csv(n.OUT / "fold_metrics.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(n.OUT / "inner_thresholds.csv", index=False)
        pd.DataFrame(oof_rows).to_csv(n.OUT / "outer_predictions.csv", index=False)
        print("outer", fold, "threshold", choice["threshold"], "H", round(meta["H"], 6), "FP/FN", meta["FP"], meta["FN"], flush=True)
    pd.DataFrame(lineage).to_json(n.OUT / "lineage.json", orient="records", indent=2)
    pd.DataFrame(calibration).to_csv(n.OUT / "calibration.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(n.OUT / "fold_metrics.csv", index=False)
    (n.OUT / "outer_run.json").write_text(json.dumps(dict(status="COMPLETE", protocol_sha256=protocol_sha, source_refits=50, cached_source_reuses=50, uploaded=False), indent=2) + "\n")


def main(stage):
    n.OUT.mkdir(parents=True, exist_ok=True)
    ctx = n.context()
    if stage == "audit": audit(ctx)
    elif stage == "outer":
        audit(ctx); run_outer(ctx)
    elif stage == "all":
        audit(ctx); run_outer(ctx)
    else: raise ValueError(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["audit", "outer", "all"])
    main(parser.parse_args().stage)
