"""Assignment ranking metrics and fixed-gate classification diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

GAIN = np.arange(5, dtype=float) / 4.0


def priorities(read_probability, incumbent_probability, threshold=0.55):
    read_probability = np.asarray(read_probability, dtype=float)
    incumbent_probability = np.asarray(incumbent_probability, dtype=float)
    incumbent_gain = incumbent_probability @ GAIN
    selected = read_probability > threshold
    priority = np.where(selected, 1.0 + 9.0 * incumbent_gain, 0.0)
    return selected, priority


def official_metrics(labels, priority, ids):
    labels = np.asarray(labels, dtype=int)
    priority = np.asarray(priority, dtype=float)
    ids = np.asarray(ids, dtype=str)
    positive = labels > 1
    selected = priority > 0
    tp = int((positive & selected).sum())
    fp = int(((labels == 1) & selected).sum())
    fn = int((positive & ~selected).sum())
    tn = int(((labels == 1) & ~selected).sum())
    order = np.asarray(sorted(np.flatnonzero(selected),
                              key=lambda ix: (-priority[ix], ids[ix])), dtype=int)
    discounts = 1 / np.log2(np.arange(len(order), dtype=float) + 2)
    gain = (labels[order] - 1) / 4.0
    q = float(np.average(gain, weights=discounts)) if len(order) else 0.0
    r = tp / int(positive.sum()) if positive.any() else 0.0
    h = 2 * q * r / (q + r) if q + r else 0.0
    return dict(total=len(labels), accepted=int(selected.sum()), dropped=int((~selected).sum()),
                TP=tp, FP=fp, FN=fn, TN=tn, total_error=fp + fn,
                missed_high=int(((labels >= 4) & ~selected).sum()),
                missed_label4=int(((labels == 4) & ~selected).sum()),
                missed_label5=int(((labels == 5) & ~selected).sum()),
                H=float(h), Q=float(q), R=float(r))


def probability_metrics(labels, probability):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    read = 1.0 - probability[:, 0]
    return dict(AUC=float(roc_auc_score(labels > 1, read)),
                Brier=float(np.mean((read - (labels > 1)) ** 2)),
                logloss=float(log_loss(labels - 1, probability, labels=np.arange(5))))


def calibration_bins(labels, read_probability, arm, fold):
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(read_probability, dtype=float)
    rows = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (values >= lo) & ((values < hi) if hi < 1 else (values <= hi))
        rows.append(dict(arm=arm, outer_fold=int(fold), lower=float(lo), upper=float(hi),
                         n=int(mask.sum()), mean_read=float(values[mask].mean()) if mask.any() else np.nan,
                         observed_positive_rate=float((labels[mask] > 1).mean()) if mask.any() else np.nan))
    return rows

