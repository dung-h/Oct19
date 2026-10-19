"""Locked nested-disagreement gate helpers.

The incumbent and SPECTER2-adapter sources are immutable nested caches from
their original recipes. The title-margin source is rebuilt locally for every
partition. No source cache generated on an outer validation row is used to
train a meta model for that row.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PROJECT = EXPERIMENTS.parent
OUT = HERE / "outputs"
N_FIT = 2494
SEED = 20261018
SOURCE_NAMES = ("incumbent", "title_margin", "neural_adapted")
THRESHOLDS = np.asarray([.45, .50, .55, .60, .65], dtype=float)
GAIN = np.arange(5, dtype=float) / 4

APM = EXPERIMENTS / "abstract_preference_margin"
if str(APM) not in sys.path:
    sys.path.insert(0, str(APM))
import apm_context as ac
import apm_features as af


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context():
    return ac.context()


def outer_jobs(ctx):
    return ac.outer_jobs(ctx)


def final_partition(ctx):
    membership = pd.read_csv(ac.source.INC / "final_inner_splits.csv", dtype={"id": str})
    assert membership.id.astype(str).tolist() == ctx["ids"][:N_FIT].astype(str).tolist()
    return np.arange(N_FIT, dtype=int), np.arange(N_FIT, len(ctx["ids"]), dtype=int), membership.inner_fold.to_numpy(int)


def source_pack(name: str, fit: np.ndarray, pred: np.ndarray):
    return ac.load_source_pack(name, fit, pred)


def incumbent_cache(name: str, fit: np.ndarray, pred: np.ndarray) -> np.ndarray:
    path = ac.source.CLAIM / f"{name}_predictions.npz"
    with np.load(path) as z:
        np.testing.assert_array_equal(z["fit"], fit)
        np.testing.assert_array_equal(z["pred"], pred)
        result = z["reference"].astype(np.float64, copy=True)
    assert result.shape == (len(pred), 5) and np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    return result


def neural_cache(name: str, fit: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # The SPECTER2 preference experiment fixed the adapted stream to epoch 8.
    stem = (f"{name}_adapted.npz" if name != "final_test" else "final_neural.npz")
    path = EXPERIMENTS / "specter2_preference" / "outputs" / stem
    with np.load(path) as z:
        np.testing.assert_array_equal(z["fit"], fit)
        np.testing.assert_array_equal(z["pred"], pred)
        result = z["epoch8"].astype(np.float64, copy=True)
    assert result.shape == (len(pred), 5) and np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    return result


def _title_feature_pack(ctx, name: str, fit: np.ndarray, pred: np.ndarray, pack: dict, protocol_sha: str):
    """Recreate only the 43 title-margin columns under the source lineage."""
    path = OUT / f"{name}_title_features.joblib"
    if path.exists():
        saved = joblib.load(path)
        assert saved["protocol_sha256"] == protocol_sha
        np.testing.assert_array_equal(saved["fit"], fit)
        np.testing.assert_array_equal(saved["pred"], pred)
        return saved
    lineage = pack["lineage"]
    assert len(lineage) == 3
    fit_position = {int(v): i for i, v in enumerate(fit)}
    train = np.full((len(fit), 43), np.nan, dtype=np.float32)
    pred_views = np.zeros((3, len(pred), 43), dtype=np.float32)
    seen = np.zeros(len(fit), dtype=bool)
    record = []
    for view, entry in enumerate(lineage):
        refs = np.asarray(entry["refs"], dtype=int)
        queries = np.asarray(entry["queries"], dtype=int)
        assert set(refs).issubset(set(fit)) and set(queries).issubset(set(fit))
        assert not (set(ctx["groups"][refs]) & set(ctx["groups"][queries]))
        pos = np.asarray([fit_position[int(q)] for q in queries], dtype=int)
        assert not seen[pos].any()
        seen[pos] = True
        value, _, _ = af.features_for_queries(ctx, queries, refs, "title_margin")
        train[pos] = value
        value, _, _ = af.features_for_queries(ctx, pred, refs, "title_margin")
        pred_views[view] = value
        record.append(dict(view=view, source_fold=int(entry["fold"]), refs=refs.tolist(), queries=queries.tolist(),
                           refs_sha256=hashlib.sha256(np.ascontiguousarray(refs).view(np.uint8)).hexdigest(),
                           queries_sha256=hashlib.sha256(np.ascontiguousarray(queries).view(np.uint8)).hexdigest()))
    assert seen.all() and np.isfinite(train).all() and np.isfinite(pred_views).all()
    saved = dict(protocol_sha256=protocol_sha, fit=fit, pred=pred, train=train, pred_views=pred_views, lineage=record)
    joblib.dump(saved, path, compress=3)
    return saved


def add_title_features(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    assert values.shape == (len(result), 43)
    for col, value in zip(af.feature_names(), values.T):
        assert col not in result.columns
        result[col] = value
    return result


def catboost_estimator() -> CatBoostClassifier:
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8,
                              loss_function="MultiClass", classes_count=5, random_seed=SEED,
                              thread_count=4, verbose=False, allow_writing_files=False)


def title_source(ctx, name: str, fit: np.ndarray, pred: np.ndarray, protocol_sha: str) -> dict[str, np.ndarray]:
    """Rebuild incumbent parity plus title-margin prediction for one partition."""
    path = OUT / f"{name}_sources.npz"
    if path.exists():
        with np.load(path) as z:
            assert str(z["protocol_sha256"].item()) == protocol_sha
            np.testing.assert_array_equal(z["fit"], fit)
            np.testing.assert_array_equal(z["pred"], pred)
            return {key: z[key].copy() for key in SOURCE_NAMES}
    pack, xfit, xpred = source_pack(name, fit, pred)
    incumbent = incumbent_cache(name, fit, pred)
    y = ctx["labels"][fit] - 1
    reference = catboost_estimator()
    reference.fit(xfit, y, cat_features=["venue"])
    replay = reference.predict_proba(xpred)
    np.testing.assert_allclose(replay, incumbent, atol=1e-7, rtol=1e-6)
    reference.save_model(str(OUT / f"{name}_incumbent_replay.cbm"))
    margins = _title_feature_pack(ctx, name, fit, pred, pack, protocol_sha)
    model = catboost_estimator()
    model.fit(add_title_features(xfit, margins["train"]), y, cat_features=["venue"])
    views = [model.predict_proba(add_title_features(xpred, margins["pred_views"][view])) for view in range(3)]
    title = np.mean(views, axis=0)
    available = ctx["abstract_available"][pred]
    title[~available] = incumbent[~available]
    assert np.isfinite(title).all()
    np.testing.assert_allclose(title.sum(axis=1), 1, atol=1e-6)
    model.save_model(str(OUT / f"{name}_title_margin.cbm"))
    neural = neural_cache(name, fit, pred)
    np.savez_compressed(path, protocol_sha256=np.asarray(protocol_sha), fit=fit, pred=pred,
                        incumbent=incumbent, title_margin=title, neural_adapted=neural)
    return dict(incumbent=incumbent, title_margin=title, neural_adapted=neural)


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, float), 1e-5, 1 - 1e-5)
    return np.log(values / (1 - values))


def meta_features(source_probabilities: dict[str, np.ndarray]) -> np.ndarray:
    inc = source_probabilities["incumbent"]
    title = source_probabilities["title_margin"]
    neural = source_probabilities["neural_adapted"]
    read = {name: 1 - value[:, 0] for name, value in [("inc", inc), ("title", title), ("neural", neural)]}
    logits = {name: _logit(value) for name, value in read.items()}
    gain = {name: value @ GAIN for name, value in [("inc", inc), ("title", title), ("neural", neural)]}
    result = np.column_stack([logits["inc"], gain["inc"], logits["title"], gain["title"], logits["neural"], gain["neural"],
                              logits["title"] - logits["inc"], logits["neural"] - logits["inc"], logits["neural"] - logits["title"]])
    assert result.shape == (len(inc), 9) and np.isfinite(result).all()
    return result


def fit_meta(x: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.2, solver="lbfgs", max_iter=1000, class_weight=None, random_state=SEED)
    model.fit(scaler.transform(x), y.astype(int))
    return scaler, model


def meta_probability(recipe, x: np.ndarray) -> np.ndarray:
    scaler, model = recipe
    p = model.predict_proba(scaler.transform(x))[:, 1]
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    return p


def priority(read: np.ndarray, incumbent_probability: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.asarray(read) > threshold, 1 + 9 * (incumbent_probability @ GAIN), 0.0)


def metrics(labels: np.ndarray, priorities: np.ndarray, ids: np.ndarray) -> dict:
    labels = np.asarray(labels, int); priorities = np.asarray(priorities, float); ids = np.asarray(ids, str)
    selected = priorities > 0; positive = labels > 1
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-priorities[i], ids[i])), int)
    weights = 1 / np.log2(np.arange(len(order), dtype=float) + 2)
    q = float(np.average((labels[order] - 1) / 4, weights=weights)) if len(order) else 0.0
    tp = int(np.sum(selected & positive)); fp = int(np.sum(selected & ~positive)); fn = int(np.sum(~selected & positive)); tn = int(np.sum(~selected & ~positive))
    r = tp / int(positive.sum()) if positive.any() else 0.0
    return dict(H=2*q*r/(q+r) if q+r else 0.0, Q=q, R=r, accepted=int(selected.sum()), dropped=int((~selected).sum()),
                TP=tp, FP=fp, FN=fn, TN=tn, total_error=fp+fn,
                missed_label4=int(np.sum((labels == 4) & ~selected)), missed_label5=int(np.sum((labels == 5) & ~selected)),
                missed_high=int(np.sum((labels >= 4) & ~selected)))


def choose_threshold(inner_read: np.ndarray, inner_incumbent: np.ndarray, labels: np.ndarray, ids: np.ndarray, nums: np.ndarray):
    rows = []
    for threshold in THRESHOLDS:
        fold_metrics = [metrics(labels[nums == fold], priority(inner_read[nums == fold], inner_incumbent[nums == fold], threshold), ids[nums == fold])
                        for fold in sorted(np.unique(nums))]
        rows.append(dict(threshold=float(threshold), inner_H=float(np.mean([x["H"] for x in fold_metrics])),
                         inner_Q=float(np.mean([x["Q"] for x in fold_metrics])), inner_R=float(np.mean([x["R"] for x in fold_metrics]))))
    # Stable sort and descending threshold explicitly implements the locked tie rule.
    return sorted(rows, key=lambda x: (x["inner_H"], x["threshold"]), reverse=True)[0], rows


def calibration_rows(labels: np.ndarray, prob: np.ndarray, outer_fold: int):
    result = []
    for low in np.arange(0, 1, .1):
        high = low + .1
        mask = (prob >= low) & ((prob < high) if high < 1 else (prob <= high))
        result.append(dict(outer_fold=int(outer_fold), lower=float(low), upper=float(high), n=int(mask.sum()),
                           mean_probability=float(prob[mask].mean()) if mask.any() else np.nan,
                           observed_positive_rate=float((labels[mask] > 1).mean()) if mask.any() else np.nan))
    return result


def read_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(labels) > 1, probability))

