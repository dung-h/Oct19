#!/usr/bin/env python3
"""OOF audit of Crossref graph evidence; no preference labels are created."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE))
import evaluate as e


def safe_logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def graph_view(bundle):
    return np.asarray(bundle["pred_views"], float).mean(axis=0)


def graph_columns(frame):
    # Fixed, interpretable summary. No C/feature search uses outer labels.
    names = e.g.GRAPH_NAMES
    raw = frame[names].to_numpy(float)
    return np.column_stack([raw[:, names.index(name)] for name in [
        "graph_available", "graph_log_reference_count", "graph_overlap_support",
        "graph_PminusN_max", "graph_PminusN_mean5", "graph_GminusN_max", "graph_GminusN_mean5",
    ]])


def outer_partitions(ctx):
    for fold, fit, pred, nums in e.ac.outer_jobs(ctx):
        train_x, train_y, train_inc = [], [], []
        for part in sorted(np.unique(nums)):
            mask = nums == part
            bundle = joblib.load(OUT / f"outer{fold}_inner{part}_graph_features.joblib")
            with np.load(OUT / f"B_metadata_outer{fold}_inner{part}_predictions.npz") as values:
                np.testing.assert_array_equal(values["pred"], fit[mask])
                inc = values["incumbent"]
            train_x.append(graph_view(bundle)); train_y.append(ctx["labels"][fit[mask]] > 1); train_inc.append(1 - inc[:, 0])
        val_bundle = joblib.load(OUT / f"outer{fold}_validation_graph_features.joblib")
        with np.load(OUT / f"B_metadata_outer{fold}_validation_predictions.npz") as values:
            np.testing.assert_array_equal(values["pred"], pred)
            val_inc = values["incumbent"]
        yield fold, fit, pred, np.vstack(train_x), np.concatenate(train_y), np.concatenate(train_inc), graph_view(val_bundle), 1 - val_inc[:, 0]


def residual_test(ctx):
    rows = []
    for fold, fit, pred, gx, target, inc, val_gx, val_inc in outer_partitions(ctx):
        # Incumbent-only and incumbent + fixed graph summary use the identical
        # grouped-crossfit training rows. This is a diagnostic, not a challenger.
        x0 = safe_logit(inc).reshape(-1, 1); z0 = safe_logit(val_inc).reshape(-1, 1)
        x1 = np.column_stack([x0, graph_columns(pd.DataFrame(gx, columns=e.g.GRAPH_NAMES))])
        z1 = np.column_stack([z0, graph_columns(pd.DataFrame(val_gx, columns=e.g.GRAPH_NAMES))])
        for arm, a, b in [("incumbent_logit", x0, z0), ("plus_graph_summary", x1, z1)]:
            scaler = StandardScaler().fit(a)
            model = LogisticRegression(C=.1, solver="lbfgs", max_iter=1000, random_state=20261018).fit(scaler.transform(a), target)
            probability = model.predict_proba(scaler.transform(b))[:, 1]
            observed = ctx["labels"][pred] > 1
            rows.append(dict(outer_fold=int(fold), arm=arm, AUC=float(roc_auc_score(observed, probability)),
                brier=float(brier_score_loss(observed, probability)), logloss=float(log_loss(observed, probability)),
                n=len(pred), covered=int(val_gx[:, 0].sum())))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "audit_residual_test.csv", index=False)
    pivot = result.pivot(index="outer_fold", columns="arm", values=["AUC", "brier", "logloss"])
    summary = result.groupby("arm", sort=False)[["AUC", "brier", "logloss"]].mean()
    summary.to_csv(OUT / "audit_residual_summary.csv")
    delta = dict(AUC=float(summary.loc["plus_graph_summary", "AUC"] - summary.loc["incumbent_logit", "AUC"]),
        brier=float(summary.loc["plus_graph_summary", "brier"] - summary.loc["incumbent_logit", "brier"]),
        logloss=float(summary.loc["plus_graph_summary", "logloss"] - summary.loc["incumbent_logit", "logloss"]),
        AUC_fold_wins=int((pivot["AUC"]["plus_graph_summary"] > pivot["AUC"]["incumbent_logit"]).sum()))
    return summary, delta


def association_table(ctx):
    rows = []
    for fold, _, pred, _, _, _, gx, inc in outer_partitions(ctx):
        for pos, index in enumerate(pred):
            rows.append(dict(index=int(index), id=str(ctx["ids"][index]), outer_fold=int(fold), Label=int(ctx["labels"][index]),
                venue=str(ctx["venues"][index]), incumbent_read=float(inc[pos]), **dict(zip(e.g.GRAPH_NAMES, gx[pos]))))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "audit_oof_graph_features.csv", index=False)
    frame["positive"] = frame.Label > 1
    frame["graph_margin_quintile"] = pd.qcut(frame.graph_PminusN_mean5.rank(method="first"), 5, labels=False)
    grouped = frame.groupby(["venue", "graph_margin_quintile"], observed=True).agg(n=("id", "size"),
        positive_rate=("positive", "mean"), mean_label=("Label", "mean"), mean_incumbent_read=("incumbent_read", "mean"),
        graph_coverage=("graph_available", "mean"), mean_support=("graph_overlap_support", "mean"))
    grouped.to_csv(OUT / "audit_association_by_venue_margin.csv")
    return frame


def neighbor_cases(ctx, frame):
    # Fixed, deterministic cases: changes from incumbent to C at one common .55
    # threshold, then nearest graph neighbours from the matching outer fit only.
    events = []
    for fold, fit, pred, _, _, _, gx, _ in outer_partitions(ctx):
        with np.load(OUT / f"C_overlap_outer{fold}_validation_predictions.npz") as c:
            old, new = 1 - c["incumbent"][:, 0] > .55, 1 - c["probability"][:, 0] > .55
        for pos, index in enumerate(pred):
            positive = ctx["labels"][index] > 1
            event = "FN_rescued" if not old[pos] and new[pos] and positive else "FP_removed" if old[pos] and not new[pos] and not positive else "lost_TP" if old[pos] and not new[pos] and positive else "new_FP" if not old[pos] and new[pos] and not positive else ""
            if event:
                events.append(dict(event=event, outer_fold=fold, query_index=int(index), id=str(ctx["ids"][index]), Label=int(ctx["labels"][index]),
                    title=str(ctx["role_texts"].title.iloc[index]), abstract=str(ctx["role_texts"].abstract.iloc[index])))
    picked = pd.DataFrame(events).sort_values(["event", "outer_fold", "id"], kind="stable").groupby("event", group_keys=False).head(8)
    graph = e.graph_context(ctx); cases = []
    for query in picked.itertuples():
        fold = int(query.outer_fold)
        fit = next(value[1] for value in e.ac.outer_jobs(ctx) if value[0] == fold)
        query_index = int(query.query_index)
        qrefs = graph["refs"][query_index]
        neighbors = []
        for ref in fit:
            if ctx["groups"][ref] == ctx["groups"][query_index] or not graph["refs"][ref]:
                continue
            common = len(qrefs & graph["refs"][ref])
            if common:
                union = len(qrefs | graph["refs"][ref])
                neighbors.append((common / union, common, str(ctx["ids"][ref]), int(ref)))
        for rank, (score, common, _, ref) in enumerate(sorted(neighbors, key=lambda x: (-x[0], -x[1], x[2]))[:5], 1):
            cases.append(dict(event=query.event, outer_fold=fold, query_id=query.id, query_label=query.Label,
                query_title=query.title, query_abstract=query.abstract[:700], rank=rank, neighbor_id=str(ctx["ids"][ref]),
                neighbor_label=int(ctx["labels"][ref]), neighbor_venue=str(ctx["venues"][ref]), neighbor_title=str(ctx["role_texts"].title.iloc[ref]),
                neighbor_abstract=str(ctx["role_texts"].abstract.iloc[ref])[:700], jaccard=score, shared_references=common))
    pd.DataFrame(cases).to_csv(OUT / "audit_neighbor_cases.csv", index=False)
    picked.to_csv(OUT / "audit_case_queries.csv", index=False)
    return picked, pd.DataFrame(cases)


def main():
    ctx = e.ac.context()
    summary, residual = residual_test(ctx)
    features = association_table(ctx)
    picked, neighbors = neighbor_cases(ctx, features)
    meta = dict(residual_test=residual, fixed_cases=len(picked), neighbor_rows=len(neighbors),
        source_sha256=hashlib.sha256((OUT / "doi_reference_ledger.jsonl").read_bytes()).hexdigest(),
        note="Case selection uses OOF prediction transitions only; text is descriptive and no labels are changed.")
    (OUT / "audit_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    text = "# Crossref graph audit\n\n" + summary.reset_index().to_markdown(index=False, floatfmt=".6f") + "\n\n```json\n" + json.dumps(meta, indent=2) + "\n```\n"
    (OUT / "AUDIT_RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
