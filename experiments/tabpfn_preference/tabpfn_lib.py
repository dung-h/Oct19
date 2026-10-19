"""Shared helpers for the locked TabPFN preference experiment."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from tabpfn import TabPFNClassifier

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PROJECT = EXPERIMENTS.parent
OUT = HERE / "outputs"
N_FIT = 2494
SEED = 20261018
THRESHOLDS = np.arange(0, .901, .05)
GAIN = np.arange(5, dtype=np.float64) / 4

APM = EXPERIMENTS / "abstract_preference_margin"
if str(APM) not in sys.path:
    sys.path.insert(0, str(APM))
import apm_context as ac


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context():
    return ac.context()


def jobs(ctx):
    return ac.outer_jobs(ctx)


def pack(name, fit, pred):
    return ac.load_source_pack(name, fit, pred)


def incumbent(name, fit, pred):
    path = ac.source.CLAIM / f"{name}_predictions.npz"
    with np.load(path) as z:
        np.testing.assert_array_equal(z["fit"], fit)
        np.testing.assert_array_equal(z["pred"], pred)
        result = z["reference"].astype(np.float64)
    assert result.shape == (len(pred), 5)
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    return result


def matrix(train: pd.DataFrame, query: pd.DataFrame):
    """Encode venue from train categories; retain the other 276 values exactly."""
    assert train.columns.equals(query.columns) and train.columns[0] == "venue"
    categories = sorted(train.venue.astype(str).unique().tolist())
    lookup = {v: i for i, v in enumerate(categories)}

    def one(frame):
        rest = frame.drop(columns="venue").to_numpy(dtype=np.float32, copy=True)
        assert rest.shape[1] == 276 and np.isfinite(rest).all()
        venue = frame.venue.astype(str).map(lookup).to_numpy(dtype=np.float32)
        # Any unseen category is represented as missing, as required by the contract.
        venue[~frame.venue.astype(str).isin(lookup).to_numpy()] = np.nan
        values = np.column_stack([venue, rest]).astype(np.float32, copy=False)
        assert values.shape[1] == 277
        return values
    return one(train), one(query), categories


def tabpfn():
    return TabPFNClassifier(n_estimators=4, categorical_features_indices=[0], device="cuda",
                             random_state=SEED, fit_mode="fit_with_cache", inference_precision="auto",
                             n_jobs=4)


def catboost():
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8,
                              loss_function="MultiClass", classes_count=5, random_seed=SEED,
                              thread_count=4, verbose=False, allow_writing_files=False)


def probability(model, xfit, yfit, xpred):
    model.fit(xfit, yfit)
    result = model.predict_proba(xpred).astype(np.float64)
    assert result.shape == (len(xpred), 5) and np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(model.classes_, np.arange(5))
    return result


def priority(probability, threshold):
    probability = np.asarray(probability, float)
    read = 1 - probability[:, 0]
    gain = probability @ GAIN
    return np.where(read > threshold, 1 + 9 * gain, 0.0)


def metric(labels, priority_values, ids):
    labels = np.asarray(labels, int); priorities = np.asarray(priority_values, float); ids = np.asarray(ids, str)
    selected = priorities > 0; positive = labels > 1
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-priorities[i], ids[i])), int)
    weights = 1 / np.log2(np.arange(len(order), dtype=float) + 2)
    q = float(np.average((labels[order] - 1) / 4, weights=weights)) if len(order) else 0.0
    r = float((selected & positive).sum() / positive.sum()) if positive.any() else 0.0
    return dict(H=2*q*r/(q+r) if q+r else 0., Q=q, R=r,
                accepted=int(selected.sum()), dropped=int((~selected).sum()),
                TP=int((selected & positive).sum()), FP=int((selected & ~positive).sum()),
                FN=int((~selected & positive).sum()), TN=int((~selected & ~positive).sum()),
                total_error=int((selected & ~positive).sum() + (~selected & positive).sum()),
                missed_label2=int(((labels == 2) & ~selected).sum()),
                missed_label3=int(((labels == 3) & ~selected).sum()),
                missed_label4=int(((labels == 4) & ~selected).sum()),
                missed_label5=int(((labels == 5) & ~selected).sum()))


def diagnostics(labels, probability):
    target = np.asarray(labels) > 1
    read = 1 - np.asarray(probability)[:, 0]
    return dict(read_AUC=float(roc_auc_score(target, read)), multiclass_logloss=float(log_loss(labels - 1, probability, labels=np.arange(5))),
                Brier=float(np.mean((read - target) ** 2)))


def choose(probability, labels, ids, nums):
    rows = []
    for threshold in THRESHOLDS:
        fold_rows = [metric(labels[nums == k], priority(probability[nums == k], threshold), ids[nums == k])
                     for k in sorted(np.unique(nums))]
        rows.append(dict(threshold=float(threshold), inner_H=float(np.mean([r["H"] for r in fold_rows])),
                         inner_Q=float(np.mean([r["Q"] for r in fold_rows])), inner_R=float(np.mean([r["R"] for r in fold_rows]))))
    # Exact tie rule is lower threshold, declared before the run.
    return sorted(rows, key=lambda r: (-r["inner_H"], r["threshold"]))[0], rows


def calibration(labels, probability, outer_fold, arm):
    target = np.asarray(labels) > 1; read = 1 - np.asarray(probability)[:, 0]
    rows = []
    for lo in np.arange(0, 1, .1):
        hi = lo + .1; mask = (read >= lo) & ((read < hi) if hi < 1 else (read <= hi))
        rows.append(dict(outer_fold=outer_fold, arm=arm, lower=float(lo), upper=float(hi), n=int(mask.sum()),
                         mean_probability=float(read[mask].mean()) if mask.any() else np.nan,
                         observed_read_rate=float(target[mask].mean()) if mask.any() else np.nan))
    return rows


def save_prediction(path, fit, pred, probability, categories, elapsed_s, protocol_sha):
    np.savez_compressed(path, fit=fit, pred=pred, probability=probability.astype(np.float32),
                        categories=np.asarray(categories, dtype=str), elapsed_s=np.asarray(float(elapsed_s)),
                        protocol_sha=np.asarray(protocol_sha))


def load_prediction(path, fit, pred, protocol_sha):
    with np.load(path) as z:
        np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
        assert str(z["protocol_sha"].item()) == protocol_sha
        result = z["probability"].astype(np.float64); cats = z["categories"].astype(str).tolist()
    assert result.shape == (len(pred), 5); np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)
    return result, cats
