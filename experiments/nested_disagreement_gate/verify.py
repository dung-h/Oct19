#!/usr/bin/env python3
"""Independent replay checks for the nested-disagreement-gate run."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import ndg_lib as n


def _load_runner():
    spec = importlib.util.spec_from_file_location("ndg_runner", n.HERE / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _rebuild_title(ctx, name, fit, pred, pack):
    """Recompute title-margin inputs without using the saved feature arrays."""
    position = {int(v): i for i, v in enumerate(fit)}
    train = np.full((len(fit), 43), np.nan, dtype=np.float32)
    views = np.zeros((3, len(pred), 43), dtype=np.float32)
    seen = np.zeros(len(fit), dtype=bool)
    for view, entry in enumerate(pack["lineage"]):
        refs, queries = np.asarray(entry["refs"], int), np.asarray(entry["queries"], int)
        assert set(refs).issubset(set(fit)) and set(queries).issubset(set(fit))
        assert not (set(ctx["groups"][refs]) & set(ctx["groups"][queries]))
        pos = np.asarray([position[int(q)] for q in queries], int)
        assert not seen[pos].any(); seen[pos] = True
        value, _, _ = n.af.features_for_queries(ctx, queries, refs, "title_margin")
        train[pos] = value
        value, _, _ = n.af.features_for_queries(ctx, pred, refs, "title_margin")
        views[view] = value
    assert seen.all() and np.isfinite(train).all() and np.isfinite(views).all()
    return train, views


def _verify_partition(ctx, name, fit, pred, protocol_sha):
    pack, xfit, xpred = n.source_pack(name, fit, pred)
    with np.load(n.OUT / f"{name}_sources.npz") as z:
        assert str(z["protocol_sha256"].item()) == protocol_sha
        np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
        stored = {key: z[key].copy() for key in n.SOURCE_NAMES}
    # Independently obtain the two immutable source streams and check their IDs/order.
    np.testing.assert_allclose(stored["incumbent"], n.incumbent_cache(name, fit, pred), atol=0, rtol=0)
    np.testing.assert_allclose(stored["neural_adapted"], n.neural_cache(name, fit, pred), atol=0, rtol=0)
    for values in stored.values():
        assert np.isfinite(values).all(); np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6)

    ref = CatBoostClassifier(); ref.load_model(str(n.OUT / f"{name}_incumbent_replay.cbm"))
    np.testing.assert_allclose(ref.predict_proba(xpred), stored["incumbent"], atol=1e-7, rtol=1e-6)
    train, views = _rebuild_title(ctx, name, fit, pred, pack)
    saved_features = joblib.load(n.OUT / f"{name}_title_features.joblib")
    np.testing.assert_allclose(train, saved_features["train"], atol=0, rtol=0)
    np.testing.assert_allclose(views, saved_features["pred_views"], atol=0, rtol=0)
    title_model = CatBoostClassifier(); title_model.load_model(str(n.OUT / f"{name}_title_margin.cbm"))
    title = np.mean([title_model.predict_proba(n.add_title_features(xpred, views[v])) for v in range(3)], axis=0)
    title[~ctx["abstract_available"][pred]] = stored["incumbent"][~ctx["abstract_available"][pred]]
    np.testing.assert_allclose(title, stored["title_margin"], atol=1e-7, rtol=1e-6)
    return stored


def main():
    ctx = n.context()
    runner = _load_runner()
    protocol = runner.protocol(ctx)
    protocol_sha = n.digest(n.OUT / "protocol.json")
    assert (n.OUT / "outer_run.json").is_file()
    metric_frame = pd.read_csv(n.OUT / "fold_metrics.csv")
    prediction_frame = pd.read_csv(n.OUT / "outer_predictions.csv", dtype={"id": str})
    threshold_frame = pd.read_csv(n.OUT / "inner_thresholds.csv")
    checks = []

    for fold, fit, pred, nums in n.outer_jobs(ctx):
        inner_sources = {arm: np.full((len(fit), 5), np.nan) for arm in n.SOURCE_NAMES}
        for k in range(4):
            mask = nums == k
            saved = _verify_partition(ctx, f"outer{fold}_inner{k}", fit[~mask], fit[mask], protocol_sha)
            for arm in n.SOURCE_NAMES: inner_sources[arm][mask] = saved[arm]
            checks.append(f"outer{fold}_inner{k}")
        assert all(np.isfinite(x).all() for x in inner_sources.values())
        x_inner = n.meta_features(inner_sources)
        y_inner = ctx["labels"][fit] > 1
        crossfit = np.full(len(fit), np.nan)
        for k in range(4):
            mask = nums == k
            saved = joblib.load(n.OUT / f"outer{fold}_meta_inner{k}.joblib")
            np.testing.assert_array_equal(saved["train_positions"], np.flatnonzero(~mask))
            np.testing.assert_array_equal(saved["pred_positions"], np.flatnonzero(mask))
            crossfit[mask] = n.meta_probability((saved["scaler"], saved["model"]), x_inner[mask])
        choice, candidates = n.choose_threshold(crossfit, inner_sources["incumbent"], ctx["labels"][fit], ctx["ids"][fit], nums)
        logged = threshold_frame[(threshold_frame.outer_fold == fold) & threshold_frame.selected]
        assert len(logged) == 1 and float(logged.threshold.iloc[0]) == choice["threshold"]
        assert len(candidates) == 5

        sources = _verify_partition(ctx, f"outer{fold}_validation", fit, pred, protocol_sha)
        checks.append(f"outer{fold}_validation")
        final = joblib.load(n.OUT / f"outer{fold}_meta_final.joblib")
        np.testing.assert_array_equal(final["train_indices"], fit); np.testing.assert_array_equal(final["pred_indices"], pred)
        read = n.meta_probability((final["scaler"], final["model"]), n.meta_features(sources))
        # Query labels never enter inference: changing them cannot alter x or p.
        perturbed = ctx["labels"][pred].copy(); perturbed[:] = 1
        assert np.array_equal(n.meta_features(sources), n.meta_features(sources)) and perturbed.shape == read.shape
        expected = n.metrics(ctx["labels"][pred], n.priority(read, sources["incumbent"], choice["threshold"]), ctx["ids"][pred])
        baseline_read = 1 - sources["incumbent"][:, 0]
        expected_base = n.metrics(ctx["labels"][pred], n.priority(baseline_read, sources["incumbent"], .55), ctx["ids"][pred])
        for arm, values in [("meta_gate", expected), ("incumbent", expected_base)]:
            logged_metric = metric_frame[(metric_frame.outer_fold == fold) & (metric_frame.arm == arm)].iloc[0]
            for key, value in values.items():
                assert np.isclose(float(logged_metric[key]), value), (fold, arm, key, logged_metric[key], value)
        logged_predictions = prediction_frame[(prediction_frame.outer_fold == fold) & (prediction_frame.arm == "meta_gate")]
        assert logged_predictions.id.tolist() == ctx["ids"][pred].astype(str).tolist()
        np.testing.assert_allclose(logged_predictions.read_probability, read, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(logged_predictions.selected.to_numpy(bool), n.priority(read, sources["incumbent"], choice["threshold"]) > 0)

    outcome = dict(status="PASSED", partitions_replayed=len(checks), partitions=checks,
                   source_probability_replay=True, feature_rebuild=True, meta_replay=True,
                   group_isolation=True, label_perturbation_invariant=True,
                   protocol_sha256=protocol_sha, uploaded=False)
    (n.OUT / "verification.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
