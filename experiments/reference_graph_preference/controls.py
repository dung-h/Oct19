#!/usr/bin/env python3
"""A/B/C control for reference availability/count versus shared references.

Run only after `run.py outer` in the same REFERENCE_GRAPH_OUTPUT directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("REFERENCE_GRAPH_OUTPUT", "outputs")
spec = importlib.util.spec_from_file_location("reference_graph_run", HERE / "run.py")
assert spec and spec.loader
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
sys_out = g.OUT
assert sys_out == OUT

# B sees only whether a verified, nonempty outgoing list exists and its size.
# C sees the full reference-overlap family already evaluated by run.py.
B_COLUMNS = ["graph_available", "graph_log_reference_count"]


def protocol_sha() -> str:
    value = json.loads((OUT / "protocol.json").read_text())
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def b_probability(ctx, graph, name, fit, pred, sha):
    path = OUT / f"control_B_{name}.npz"
    if path.exists():
        with np.load(path) as z:
            np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
            return z["probability"].copy(), z["incumbent"].copy()
    pack, xfit, xpred = g.source_pack(name, fit, pred)
    features = g.partition_features(ctx, graph, name, fit, pred, pack, sha)
    columns = [g.GRAPH_NAMES.index(x) for x in B_COLUMNS]
    train = pd.concat([xfit.reset_index(drop=True), pd.DataFrame(features["train"][:, columns], columns=B_COLUMNS)], axis=1)
    model = g.catboost()
    model.fit(train, ctx["labels"][fit] - 1, cat_features=["venue"])
    views = []
    for values in features["pred_views"]:
        frame = pd.concat([xpred.reset_index(drop=True), pd.DataFrame(values[:, columns], columns=B_COLUMNS)], axis=1)
        views.append(model.predict_proba(frame))
    result = np.mean(views, axis=0)
    incumbent = g.ndg.incumbent_cache(name, fit, pred)
    result[~features["graph_available"].astype(bool)] = incumbent[~features["graph_available"].astype(bool)]
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    np.savez_compressed(path, fit=fit, pred=pred, probability=result, incumbent=incumbent)
    return result, incumbent


def c_probability(name, fit, pred, sha):
    path = OUT / f"{name}_predictions.npz"
    with np.load(path) as z:
        assert str(z["protocol_sha256"].item()) == sha
        np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
        return z["probability"].copy(), z["incumbent"].copy()


def threshold(probability, labels, ids, nums):
    return g.choose_threshold(probability, labels, ids, nums)[0]


def main():
    ctx = g.ac.context(); graph = g.graph_context(ctx); sha = protocol_sha()
    rows, choices, oof = [], [], []
    with threadpool_limits(4):
        for fold, fit, pred, nums in g.ac.outer_jobs(ctx):
            inner = {name: np.full((len(fit), 5), np.nan) for name in ["A_incumbent", "B_metadata", "C_overlap"]}
            for part in sorted(np.unique(nums)):
                mask = nums == part; name = f"outer{fold}_inner{part}"
                b, a = b_probability(ctx, graph, name, fit[~mask], fit[mask], sha)
                c, a2 = c_probability(name, fit[~mask], fit[mask], sha)
                np.testing.assert_allclose(a, a2, atol=1e-7)
                inner["A_incumbent"][mask] = a; inner["B_metadata"][mask] = b; inner["C_overlap"][mask] = c
            selected_thresholds = {arm: threshold(prob, ctx["labels"][fit], ctx["ids"][fit], nums) for arm, prob in inner.items()}
            b, a = b_probability(ctx, graph, f"outer{fold}_validation", fit, pred, sha)
            c, a2 = c_probability(f"outer{fold}_validation", fit, pred, sha)
            np.testing.assert_allclose(a, a2, atol=1e-7)
            outer = {"A_incumbent": a, "B_metadata": b, "C_overlap": c}
            for arm, prob in outer.items():
                priority = g.priority(prob, selected_thresholds[arm])
                values = g.metrics(ctx["labels"][pred], priority, ctx["ids"][pred])
                values.update(outer_fold=int(fold), arm=arm, threshold=float(selected_thresholds[arm]),
                              AUC=float(roc_auc_score(ctx["labels"][pred] > 1, 1 - prob[:, 0])))
                rows.append(values); choices.append(dict(outer_fold=int(fold), arm=arm, threshold=float(selected_thresholds[arm])))
                for position, index in enumerate(pred):
                    oof.append(dict(index=int(index), id=str(ctx["ids"][index]), outer_fold=int(fold), Label=int(ctx["labels"][index]),
                        graph_covered=bool(graph["available"][index]), arm=arm, priority=float(priority[position])))
            print(f"control outer={fold} thresholds={selected_thresholds}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "control_fold_metrics.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "control_thresholds.csv", index=False)
    detail = pd.DataFrame(oof); detail.to_csv(OUT / "control_oof.csv", index=False)
    summary = pd.DataFrame(rows).groupby("arm", sort=False).agg(H=("H", "mean"), Q=("Q", "mean"), R=("R", "mean"), AUC=("AUC", "mean"),
        selected=("selected", "sum"), TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"), missed_high=("missed_high", "sum"))
    summary.to_csv(OUT / "control_summary.csv")
    pivot = pd.DataFrame(rows).pivot(index="outer_fold", columns="arm", values="H")
    deltas = {"B_minus_A": float(summary.loc["B_metadata", "H"] - summary.loc["A_incumbent", "H"]),
              "C_minus_B": float(summary.loc["C_overlap", "H"] - summary.loc["B_metadata", "H"]),
              "C_minus_A": float(summary.loc["C_overlap", "H"] - summary.loc["A_incumbent", "H"]),
              "C_vs_B_fold_wins": int((pivot.C_overlap > pivot.B_metadata).sum())}
    (OUT / "control_decision.json").write_text(json.dumps(deltas, indent=2) + "\n")
    text = "# A/B/C reference controls\n\n" + summary.reset_index().to_markdown(index=False, floatfmt=".6f") + "\n\n" + json.dumps(deltas, indent=2) + "\n"
    (OUT / "CONTROL_RESULTS.md").write_text(text)
    print(summary.to_string()); print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
