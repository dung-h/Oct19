#!/usr/bin/env python3
"""Grouped A/B/C test of Crossref DOI bibliographic-coupling features."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
GRAPH_SOURCE = OUT / "doi_reference_ledger.jsonl"
sys.path[:0] = [str(ROOT / "experiments" / "abstract_preference_margin"), str(ROOT / "experiments" / "nested_disagreement_gate")]
import apm_context as ac
import ndg_lib as ndg

_spec = importlib.util.spec_from_file_location("reference_graph_helpers", ROOT / "experiments" / "reference_graph_preference" / "run.py")
assert _spec and _spec.loader
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)
g.OUT = OUT  # Reuse only leakage-safe feature construction with an isolated cache.

B_COLUMNS = ["graph_available", "graph_log_reference_count"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_context(ctx: dict) -> dict:
    rows = [json.loads(line) for line in GRAPH_SOURCE.read_text(encoding="utf8").splitlines()]
    lookup = {(row["split"], str(row["id"])): row for row in rows if row["identity_status"] == "ACCEPTED"}
    refs = []
    for key in ctx["keys"].astype(str):
        split, identifier = key.split(":", 1)
        row = lookup.get((split, identifier), {})
        refs.append(frozenset(row.get("references", [])))
    available = np.asarray([bool(value) for value in refs], bool)
    return dict(refs=refs, available=available, source_sha256=digest(GRAPH_SOURCE))


def protocol(ctx: dict, graph: dict) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(plan_sha256=digest(HERE / "PLAN.md"), source_sha256=digest(HERE / "evaluate.py"),
        graph_source_sha256=graph["source_sha256"], graph_feature_names=g.GRAPH_NAMES, metadata_columns=B_COLUMNS,
        data_rows=len(ctx["ids"]), split_sha256=digest(ac.split_path()), seed=g.SEED,
        model=dict(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8, loss="MultiClass"),
        selection="each arm selects threshold independently using only its four inner folds",
        fallback="queries without verified nonempty Crossref references copy the exact incumbent probability")
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == value, "Protocol/source changed; use a new output directory."
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def features(ctx, graph, name, fit, pred, pack, sha):
    # This invokes existing group-safe construction: training rows are OOF within
    # the outer fit set; prediction rows use the three matching fit-reference views.
    return g.partition_features(ctx, graph, name, fit, pred, pack, sha)


def model_probability(ctx, graph, name, fit, pred, sha, arm):
    path = OUT / f"{arm}_{name}_predictions.npz"
    if path.exists():
        with np.load(path) as z:
            np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
            return z["probability"].copy(), z["incumbent"].copy(), z["covered"].astype(bool)
    pack, xfit, xpred = ac.load_source_pack(name, fit, pred)
    bundle = features(ctx, graph, name, fit, pred, pack, sha)
    if arm == "B_metadata":
        columns = B_COLUMNS
        take = [g.GRAPH_NAMES.index(column) for column in columns]
    elif arm == "C_overlap":
        columns = g.GRAPH_NAMES
        take = list(range(len(columns)))
    else:
        raise ValueError(arm)
    train = pd.concat([xfit.reset_index(drop=True), pd.DataFrame(bundle["train"][:, take], columns=columns)], axis=1)
    model = g.catboost()
    model.fit(train, ctx["labels"][fit] - 1, cat_features=["venue"])
    views = []
    for value in bundle["pred_views"]:
        frame = pd.concat([xpred.reset_index(drop=True), pd.DataFrame(value[:, take], columns=columns)], axis=1)
        views.append(model.predict_proba(frame))
    result = np.mean(views, axis=0)
    incumbent = ndg.incumbent_cache(name, fit, pred)
    covered = bundle["graph_available"].astype(bool)
    result[~covered] = incumbent[~covered]
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    np.savez_compressed(path, fit=fit, pred=pred, probability=result, incumbent=incumbent, covered=covered)
    return result, incumbent, covered


def choose(probability, labels, ids, inner_fold):
    return g.choose_threshold(probability, labels, ids, inner_fold)[0]


def main():
    ctx = ac.context(); graph = graph_context(ctx); sha = protocol(ctx, graph)
    rows, threshold_rows, detail = [], [], []
    with threadpool_limits(4):
        for fold, fit, pred, numbers in ac.outer_jobs(ctx):
            inner = {key: np.full((len(fit), 5), np.nan) for key in ["A_incumbent", "B_metadata", "C_overlap"]}
            for part in sorted(np.unique(numbers)):
                mask = numbers == part; name = f"outer{fold}_inner{part}"
                b, incumbent, _ = model_probability(ctx, graph, name, fit[~mask], fit[mask], sha, "B_metadata")
                c, incumbent_c, _ = model_probability(ctx, graph, name, fit[~mask], fit[mask], sha, "C_overlap")
                np.testing.assert_allclose(incumbent, incumbent_c, atol=1e-7)
                inner["A_incumbent"][mask] = incumbent
                inner["B_metadata"][mask] = b
                inner["C_overlap"][mask] = c
            thresholds = {arm: choose(prob, ctx["labels"][fit], ctx["ids"][fit], numbers) for arm, prob in inner.items()}
            b, incumbent, covered = model_probability(ctx, graph, f"outer{fold}_validation", fit, pred, sha, "B_metadata")
            c, incumbent_c, covered_c = model_probability(ctx, graph, f"outer{fold}_validation", fit, pred, sha, "C_overlap")
            np.testing.assert_allclose(incumbent, incumbent_c, atol=1e-7); np.testing.assert_array_equal(covered, covered_c)
            for arm, prob in [("A_incumbent", incumbent), ("B_metadata", b), ("C_overlap", c)]:
                priority = g.priority(prob, thresholds[arm])
                metric = g.metrics(ctx["labels"][pred], priority, ctx["ids"][pred])
                metric.update(outer_fold=int(fold), arm=arm, threshold=float(thresholds[arm]),
                    AUC=float(roc_auc_score(ctx["labels"][pred] > 1, 1 - prob[:, 0])), graph_covered=int(covered.sum()))
                rows.append(metric); threshold_rows.append(dict(outer_fold=int(fold), arm=arm, threshold=float(thresholds[arm])))
                for pos, index in enumerate(pred):
                    detail.append(dict(index=int(index), id=str(ctx["ids"][index]), Label=int(ctx["labels"][index]), outer_fold=int(fold),
                        graph_covered=bool(covered[pos]), arm=arm, priority=float(priority[pos])))
            print(f"outer={fold} coverage={covered.sum()}/{len(pred)} thresholds={thresholds}", flush=True)
    metrics = pd.DataFrame(rows); metrics.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT / "thresholds.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "oof_predictions.csv", index=False)
    summary = metrics.groupby("arm", sort=False).agg(H=("H", "mean"), Q=("Q", "mean"), R=("R", "mean"), AUC=("AUC", "mean"),
        selected=("selected", "sum"), TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"), missed_high=("missed_high", "sum"))
    summary.to_csv(OUT / "summary.csv")
    piv = metrics.pivot(index="outer_fold", columns="arm", values="H")
    decision = dict(B_minus_A=float(summary.loc["B_metadata", "H"] - summary.loc["A_incumbent", "H"]),
        C_minus_B=float(summary.loc["C_overlap", "H"] - summary.loc["B_metadata", "H"]),
        C_minus_A=float(summary.loc["C_overlap", "H"] - summary.loc["A_incumbent", "H"]),
        C_vs_B_fold_wins=int((piv.C_overlap > piv.B_metadata).sum()), coverage_train=int(graph["available"][:2494].sum()),
        coverage_test=int(graph["available"][2494:].sum()))
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (OUT / "RESULTS.md").write_text("# Crossref DOI graph A/B/C\n\n" + summary.reset_index().to_markdown(index=False, floatfmt=".6f") + "\n\n" + json.dumps(decision, indent=2) + "\n")
    print(summary.to_string()); print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
