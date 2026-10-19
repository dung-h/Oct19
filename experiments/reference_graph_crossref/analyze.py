#!/usr/bin/env python3
"""Separate graph-feature effects from threshold-policy effects after A/B/C CV."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
sys.path.insert(0, str(ROOT / "experiments" / "abstract_preference_margin"))
import apm_context as ac


def transitions(labels, before, after):
    pos = labels > 1
    return dict(FN_rescued=int((~before & after & pos).sum()), FP_removed=int((before & ~after & ~pos).sum()),
        new_FP=int((~before & after & ~pos).sum()), lost_TP=int((before & ~after & pos).sum()))


def official(labels, priority, ids):
    labels, priority, ids = np.asarray(labels, int), np.asarray(priority, float), np.asarray(ids, str)
    selected = priority > 0; positive = labels > 1
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-priority[i], ids[i])), int)
    weights = 1 / np.log2(np.arange(len(order), dtype=float) + 2)
    q = float(np.average((labels[order] - 1) / 4, weights=weights)) if len(order) else 0.
    r = float((labels[order] > 1).sum() / positive.sum()) if positive.any() else 0.
    return dict(H=2*q*r/(q+r) if q+r else 0., Q=q, R=r, selected=int(selected.sum()),
        TP=int((selected & positive).sum()), FP=int((selected & ~positive).sum()), FN=int((~selected & positive).sum()),
        missed_high=int(((labels >= 4) & ~selected).sum()))


def main():
    ctx = ac.context()
    thresholds = pd.read_csv(OUT / "thresholds.csv").pivot(index="outer_fold", columns="arm", values="threshold")
    results, changes = [], []
    for fold, fit, pred, _ in ac.outer_jobs(ctx):
        with np.load(OUT / f"B_metadata_outer{fold}_validation_predictions.npz") as b, np.load(OUT / f"C_overlap_outer{fold}_validation_predictions.npz") as c:
            np.testing.assert_array_equal(b["fit"], fit); np.testing.assert_array_equal(b["pred"], pred)
            np.testing.assert_array_equal(c["fit"], fit); np.testing.assert_array_equal(c["pred"], pred)
            probability = {"A_incumbent": b["incumbent"], "B_metadata": b["probability"], "C_overlap": c["probability"]}
        labels, ids = ctx["labels"][pred], ctx["ids"][pred]
        for policy in ["A_incumbent", "B_metadata", "C_overlap"]:
            threshold = float(thresholds.loc[fold, policy])
            priority = {arm: np.where((1 - p[:, 0]) > threshold, 1 + 9 * (p @ np.arange(5) / 4), 0.) for arm, p in probability.items()}
            for arm, value in priority.items():
                row = official(labels, value, ids)
                results.append(dict(outer_fold=int(fold), threshold_policy=policy, threshold=threshold, arm=arm,
                    **row))
            for before, after in [("A_incumbent", "C_overlap"), ("B_metadata", "C_overlap")]:
                changes.append(dict(outer_fold=int(fold), threshold_policy=policy, comparison=f"{after}_minus_{before}",
                    **transitions(labels, priority[before] > 0, priority[after] > 0)))
    result = pd.DataFrame(results); result.to_csv(OUT / "fixed_threshold_metrics.csv", index=False)
    delta_rows = []
    for policy, part in result.groupby("threshold_policy", sort=False):
        wide = part.pivot(index="outer_fold", columns="arm", values="H")
        delta_rows.append(dict(threshold_policy=policy, B_minus_A=float((wide.B_metadata - wide.A_incumbent).mean()),
            C_minus_B=float((wide.C_overlap - wide.B_metadata).mean()), C_minus_A=float((wide.C_overlap - wide.A_incumbent).mean()),
            C_vs_B_fold_wins=int((wide.C_overlap > wide.B_metadata).sum())))
    delta = pd.DataFrame(delta_rows); delta.to_csv(OUT / "fixed_threshold_deltas.csv", index=False)
    transition = pd.DataFrame(changes); transition.to_csv(OUT / "fixed_threshold_transitions.csv", index=False)
    text = "# Fixed-threshold diagnostic\n\n" + delta.to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += "Each row fixes every arm to the named arm's inner-selected threshold. This isolates score/feature changes from threshold choice.\n"
    (OUT / "FIXED_THRESHOLD_RESULTS.md").write_text(text)
    print(delta.to_string(index=False)); print(transition.groupby(["threshold_policy", "comparison"], sort=False).sum(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
