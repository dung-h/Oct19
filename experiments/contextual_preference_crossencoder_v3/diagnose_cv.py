#!/usr/bin/env python3
"""Read-only diagnostics over the completed V3 and incumbent outer OOF caches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
CV = HERE / "outputs" / "nested_cv"
OUT = HERE / "outputs" / "diagnosis"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
import apm_context as ac
import model_v3 as m

ARMS = ("query_only", "contextual")
GAIN = np.arange(5) / 4


def rank(selected, score, ids):
    result = np.full(len(ids), np.nan)
    order = sorted(np.flatnonzero(selected), key=lambda i: (-float(score[i]), str(ids[i])))
    result[order] = np.arange(1, len(order) + 1)
    return result


def scored_metrics(labels, ids, selected, score):
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-float(score[i]), str(ids[i]))), int)
    q = float(np.average((labels[order] - 1) / 4, weights=1 / np.log2(np.arange(len(order)) + 2))) if len(order) else 0.
    r = float(((labels > 1) & selected).sum() / (labels > 1).sum())
    h = float(2 * q * r / (q + r)) if q + r else 0.
    return dict(H=h, Q=q, R=r, accepted=int(selected.sum()), TP=int(((labels > 1) & selected).sum()),
                FP=int(((labels == 1) & selected).sum()), FN=int(((labels > 1) & ~selected).sum()),
                missed_label4=int(((labels == 4) & ~selected).sum()), missed_label5=int(((labels == 5) & ~selected).sum()))


def top_m(read, ids, m):
    selected = np.zeros(len(read), dtype=bool)
    order = sorted(range(len(read)), key=lambda i: (-float(read[i]), str(ids[i])))
    selected[np.asarray(order[:m], int)] = True
    return selected


def probabilities(ctx, fold, fit, pred, arm):
    with np.load(CV / f"outer{fold}_validation_{arm}.npz") as z:
        np.testing.assert_array_equal(z["fit"], fit)
        np.testing.assert_array_equal(z["pred"], pred)
        return z["probability"].astype(float)


def incumbent(ctx, fold, fit, pred):
    value = ac.source_probabilities(f"outer{fold}_validation", fit, pred, ctx)
    return value[0] if isinstance(value, tuple) else value


def add_model_columns(frame, name, probability, threshold):
    read = 1 - probability[:, 0]
    gain = probability @ GAIN
    selected = read > threshold
    frame[f"{name}_read_probability"] = read
    frame[f"{name}_expected_gain"] = gain
    frame[f"{name}_selected"] = selected
    frame[f"{name}_threshold"] = threshold
    frame[f"{name}_rank_if_selected"] = rank(selected, gain, frame.id.to_numpy())
    for k in range(5):
        frame[f"{name}_p{k+1}"] = probability[:, k]


def transition(old, new, labels):
    return np.select(
        [~old & new & (labels > 1), old & ~new & (labels == 1), ~old & new & (labels == 1), old & ~new & (labels > 1)],
        ["FN_rescued", "FP_removed", "new_FP", "lost_TP"], default="unchanged")


def score_shift_rows(probability, labels, ids, fold, arm, split, threshold):
    read = 1 - probability[:, 0]
    selected = read > threshold
    y = labels > 1
    return dict(outer_fold=fold, arm=arm, split=split, n=len(labels), threshold=threshold,
                read_mean=float(read.mean()), read_std=float(read.std()),
                read_p05=float(np.quantile(read, .05)), read_p25=float(np.quantile(read, .25)),
                read_p50=float(np.quantile(read, .50)), read_p75=float(np.quantile(read, .75)), read_p95=float(np.quantile(read, .95)),
                selected=int(selected.sum()), selected_fraction=float(selected.mean()), positive_rate=float(y.mean()),
                AUC=float(roc_auc_score(y, read)), logloss=float(log_loss(labels - 1, probability, labels=np.arange(5))),
                brier=float(brier_score_loss(y, read)), selected_positive_precision=float(y[selected].mean()) if selected.any() else 0.)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = ac.context()
    m.prepare(ctx)
    full_rows = []; transitions = []; diagnostics = []; counterfactuals = []
    choices = {fold: json.loads((CV / f"outer{fold}_choices.json").read_text()) for fold in range(5)}
    for fold, fit, pred, nums in ac.outer_jobs(ctx):
        labels = ctx["labels"][pred]; ids = ctx["ids"][pred]
        meta = ctx["data"].iloc[pred]
        frame = pd.DataFrame(dict(row_index=pred, id=ids.astype(str), outer_fold=fold, Label=labels,
                                 title=meta.title.astype(str).to_numpy(), venue=meta.venue.astype(str).to_numpy(),
                                 abstract_available=meta.abstract.fillna("").astype(str).str.strip().ne("").to_numpy(),
                                 raw_token_length=ctx["raw_token_lengths"][pred],
                                 side_truncated=ctx["raw_token_lengths"][pred] > 126))
        inc = incumbent(ctx, fold, fit, pred); add_model_columns(frame, "incumbent", inc, .55)
        arm_probabilities = {}
        for arm in ARMS:
            probability = probabilities(ctx, fold, fit, pred, arm); arm_probabilities[arm] = probability
            add_model_columns(frame, arm, probability, choices[fold][arm]["threshold"])
        full_rows.append(frame)

        inc_selected = frame.incumbent_selected.to_numpy(bool)
        inc_gain = frame.incumbent_expected_gain.to_numpy(float)
        for arm, probability in arm_probabilities.items():
            candidate_selected = frame[f"{arm}_selected"].to_numpy(bool)
            candidate_gain = frame[f"{arm}_expected_gain"].to_numpy(float)
            model_rows = frame[["id", "row_index", "outer_fold", "Label", "title", "venue", "abstract_available", "side_truncated",
                                "incumbent_selected", f"{arm}_selected", "incumbent_read_probability", f"{arm}_read_probability",
                                "incumbent_expected_gain", f"{arm}_expected_gain"]].copy()
            model_rows["arm"] = arm
            model_rows["transition"] = transition(inc_selected, candidate_selected, labels)
            transitions.append(model_rows)

            accepted_count = int(inc_selected.sum())
            fixed_budget = top_m(1-probability[:, 0], ids, accepted_count)
            policies = {
                "incumbent_policy": (inc_selected, inc_gain),
                "candidate_native": (candidate_selected, candidate_gain),
                "candidate_top_m_incumbent_budget": (fixed_budget, candidate_gain),
                "incumbent_selection_candidate_ranking": (inc_selected, candidate_gain),
                "candidate_selection_incumbent_ranking": (candidate_selected, inc_gain),
            }
            for policy, (selected, score) in policies.items():
                counterfactuals.append(dict(outer_fold=fold, arm=arm, policy=policy,
                                            **scored_metrics(labels, ids, selected, score)))

        if fold == 4:
            for arm in ARMS:
                threshold = choices[fold][arm]["threshold"]
                for k in range(4):
                    mask = nums == k
                    with np.load(CV / f"outer4_inner{k}_{arm}.npz") as z:
                        np.testing.assert_array_equal(z["pred"], fit[mask])
                        diagnostics.append(score_shift_rows(z["probability"].astype(float), ctx["labels"][fit[mask]], ctx["ids"][fit[mask]], fold, arm, f"inner{k}", threshold))
                diagnostics.append(score_shift_rows(arm_probabilities[arm], labels, ids, fold, arm, "outer_validation", threshold))

    comparison = pd.concat(full_rows, ignore_index=True)
    comparison.to_csv(OUT / "oof_comparison.csv", index=False)
    transition_rows = pd.concat(transitions, ignore_index=True)
    transition_rows.to_csv(OUT / "failure_transitions.csv", index=False)
    counterfactual = pd.DataFrame(counterfactuals)
    counterfactual.to_csv(OUT / "selection_ranking_diagnostics.csv", index=False)
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame.to_csv(OUT / "fold4_score_shift.csv", index=False)

    transition_summary = transition_rows.groupby(["arm", "transition"], sort=False).size().unstack(fill_value=0)
    for column in ["FN_rescued", "FP_removed", "new_FP", "lost_TP", "unchanged"]:
        if column not in transition_summary:
            transition_summary[column] = 0
    transition_summary = transition_summary[["FN_rescued", "FP_removed", "new_FP", "lost_TP", "unchanged"]]
    transition_summary["net_TP_change"] = transition_summary.FN_rescued - transition_summary.lost_TP
    transition_summary["net_FP_change"] = transition_summary.new_FP - transition_summary.FP_removed
    transition_summary.to_csv(OUT / "transition_summary.csv")

    summary = counterfactual.groupby(["arm", "policy"], sort=False)[["H", "Q", "R", "accepted", "TP", "FP", "FN", "missed_label4", "missed_label5"]].mean()
    summary.to_csv(OUT / "selection_ranking_summary.csv")

    results = ["# V3 selection and ranking diagnosis", "",
               "All figures use the completed outer OOF cache. Counterfactual policies are diagnostics; they are not candidate submissions or threshold choices.", "",
               "## Selection/ranking policies", "", summary.reset_index().to_markdown(index=False, floatfmt=".6f"), "",
               "## Per-paper selection transitions", "", transition_summary.reset_index().to_markdown(index=False), "",
               "## Fold 4 score shift", "", diagnostics_frame.to_markdown(index=False, floatfmt=".6f"), "",
               "`candidate_top_m_incumbent_budget` gives the candidate the same count accepted by the incumbent. `incumbent_selection_candidate_ranking` isolates ranking, while `candidate_selection_incumbent_ranking` isolates selection.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(results), encoding="utf8")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print(transition_summary.to_string())
    print(diagnostics_frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
