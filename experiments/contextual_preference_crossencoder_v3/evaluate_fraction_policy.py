#!/usr/bin/env python3
"""Nested, scale-invariant selection policy over already completed V3 caches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CV = HERE / "outputs" / "nested_cv"
OUT = HERE / "outputs" / "diagnosis"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
import apm_context as ac

ARMS = ("query_only", "contextual")
FRACTIONS = np.arange(.10, .901, .05)
GAIN = np.arange(5) / 4


def select_top_fraction(read, ids, fraction):
    count = int(np.ceil(fraction * len(read)))
    order = np.asarray(sorted(range(len(read)), key=lambda i: (-float(read[i]), str(ids[i]))), int)
    selected = np.zeros(len(read), dtype=bool); selected[order[:count]] = True
    return selected


def metric(labels, ids, probability, fraction):
    selected = select_top_fraction(1-probability[:, 0], ids, fraction)
    gain = probability @ GAIN
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-float(gain[i]), str(ids[i]))), int)
    q = float(np.average((labels[order]-1)/4, weights=1/np.log2(np.arange(len(order))+2))) if len(order) else 0.
    r = float(((labels > 1) & selected).sum() / (labels > 1).sum())
    h = float(2*q*r/(q+r)) if q+r else 0.
    return dict(H=h, Q=q, R=r, accepted=int(selected.sum()), TP=int(((labels>1)&selected).sum()),
                FP=int(((labels==1)&selected).sum()), FN=int(((labels>1)&~selected).sum()),
                missed_label4=int(((labels==4)&~selected).sum()), missed_label5=int(((labels==5)&~selected).sum()))


def load(path, fit, pred):
    with np.load(path) as z:
        np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
        return z["probability"].astype(float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = ac.context(); rows = []; choices = []; curves = []
    for fold, fit, pred, nums in ac.outer_jobs(ctx):
        for arm in ARMS:
            inner = np.full((len(fit), 5), np.nan)
            for k in range(4):
                mask = nums == k
                inner[mask] = load(CV / f"outer{fold}_inner{k}_{arm}.npz", fit[~mask], fit[mask])
            candidates = []
            for fraction in FRACTIONS:
                per_inner = [metric(ctx["labels"][fit[nums == k]], ctx["ids"][fit[nums == k]], inner[nums == k], fraction) for k in range(4)]
                candidate = dict(outer_fold=fold, arm=arm, fraction=float(fraction),
                                 inner_H=float(np.mean([x["H"] for x in per_inner])),
                                 inner_Q=float(np.mean([x["Q"] for x in per_inner])),
                                 inner_R=float(np.mean([x["R"] for x in per_inner])))
                candidates.append(candidate); curves.append(candidate)
            choice = max(candidates, key=lambda x: x["inner_H"]); choices.append(choice)
            outer = load(CV / f"outer{fold}_validation_{arm}.npz", fit, pred)
            result = metric(ctx["labels"][pred], ctx["ids"][pred], outer, choice["fraction"])
            rows.append(dict(**choice, **result))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "fraction_policy_outer_metrics.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "fraction_policy_choices.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "fraction_policy_inner_curves.csv", index=False)
    mean = result.groupby("arm", sort=False)[["H", "Q", "R", "accepted", "TP", "FP", "FN", "missed_label4", "missed_label5"]].mean()
    mean.to_csv(OUT / "fraction_policy_summary.csv")
    text = ["# Scale-invariant top-fraction selection", "",
            "For each outer fold and arm, choose one fraction from 0.10 to 0.90 in steps of 0.05 by mean H across its four inner folds. The outer model then selects that fraction of papers by P(read) and ranks them by its own expected gain. No outer labels enter the fraction choice.", "",
            "## Outer results", "", result.to_markdown(index=False, floatfmt=".6f"), "",
            "## Mean across outer folds", "", mean.reset_index().to_markdown(index=False, floatfmt=".6f"), ""]
    (OUT / "FRACTION_POLICY_RESULTS.md").write_text("\n".join(text), encoding="utf8")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(mean.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
