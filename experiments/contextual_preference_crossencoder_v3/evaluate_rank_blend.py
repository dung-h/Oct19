#!/usr/bin/env python3
"""Nested fixed-budget rank blend of incumbent and contextual V3 read scores."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CV = HERE / "outputs" / "nested_cv"
OUT = HERE / "outputs" / "diagnosis" / "rank_blend"
SOURCE_CACHE = HERE.parent / "nested_disagreement_gate" / "outputs"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
import apm_context as ac

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
GAIN = np.arange(5) / 4


def load_probability(path, fit, pred):
    with np.load(path) as z:
        np.testing.assert_array_equal(z["fit"], fit)
        np.testing.assert_array_equal(z["pred"], pred)
        return z["probability"].astype(float)


def incumbent(name, fit, pred):
    """Reuse the exact grouped CatBoost replay saved by the verified gate experiment."""
    with np.load(SOURCE_CACHE / f"{name}_sources.npz") as z:
        np.testing.assert_array_equal(z["fit"], fit)
        np.testing.assert_array_equal(z["pred"], pred)
        probability = z["incumbent"].astype(float)
    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
    return probability


def percentile_rank(score, ids):
    """Rank in [0,1], highest score=1; string ID resolves score ties."""
    order = np.asarray(sorted(range(len(score)), key=lambda i: (-float(score[i]), str(ids[i]))), int)
    result = np.empty(len(score), dtype=float)
    result[order] = 1 - np.arange(len(score)) / max(1, len(score)-1)
    return result


def select(incumbent_probability, contextual_probability, ids, weight):
    incumbent_read = 1 - incumbent_probability[:, 0]
    contextual_read = 1 - contextual_probability[:, 0]
    count = int((incumbent_read > .55).sum())
    score = (1-weight)*percentile_rank(incumbent_read, ids) + weight*percentile_rank(contextual_read, ids)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: (-float(score[i]), str(ids[i]))), int)
    selected = np.zeros(len(ids), dtype=bool); selected[order[:count]] = True
    return selected, score, count


def metrics(labels, ids, selected, rank_probability):
    gain = rank_probability @ GAIN
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-float(gain[i]), str(ids[i]))), int)
    q = float(np.average((labels[order]-1)/4, weights=1/np.log2(np.arange(len(order))+2))) if len(order) else 0.
    r = float(((labels > 1) & selected).sum() / (labels > 1).sum())
    h = float(2*q*r/(q+r)) if q+r else 0.
    return dict(H=h, Q=q, R=r, accepted=int(selected.sum()), TP=int(((labels>1)&selected).sum()),
                FP=int(((labels==1)&selected).sum()), FN=int(((labels>1)&~selected).sum()),
                missed_label4=int(((labels==4)&~selected).sum()), missed_label5=int(((labels==5)&~selected).sum()))


def transition(baseline, candidate, labels):
    return np.select(
        [~baseline & candidate & (labels > 1), baseline & ~candidate & (labels == 1),
         ~baseline & candidate & (labels == 1), baseline & ~candidate & (labels > 1)],
        ["FN_rescued", "FP_removed", "new_FP", "lost_TP"], default="unchanged")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = ac.context(); metric_rows = []; curve_rows = []; choice_rows = []; transition_rows = []
    for fold, fit, pred, nums in ac.outer_jobs(ctx):
        inner_incumbent = np.full((len(fit), 5), np.nan)
        inner_contextual = np.full((len(fit), 5), np.nan)
        for k in range(4):
            mask = nums == k; train, heldout = fit[~mask], fit[mask]
            inner_incumbent[mask] = incumbent(f"outer{fold}_inner{k}", train, heldout)
            inner_contextual[mask] = load_probability(CV / f"outer{fold}_inner{k}_contextual.npz", train, heldout)
        candidates = []
        for weight in WEIGHTS:
            per_inner = []
            for k in range(4):
                mask = nums == k
                chosen, _, _ = select(inner_incumbent[mask], inner_contextual[mask], ctx["ids"][fit[mask]], weight)
                per_inner.append(metrics(ctx["labels"][fit[mask]], ctx["ids"][fit[mask]], chosen, inner_incumbent[mask]))
            item = dict(outer_fold=fold, weight=weight, inner_H=float(np.mean([x["H"] for x in per_inner])),
                        inner_Q=float(np.mean([x["Q"] for x in per_inner])), inner_R=float(np.mean([x["R"] for x in per_inner])))
            candidates.append(item); curve_rows.append(item)
        choice = max(candidates, key=lambda x: (x["inner_H"], -x["weight"]))
        choice_rows.append(choice)
        outer_incumbent = incumbent(f"outer{fold}_validation", fit, pred)
        source_outer = ac.source_probabilities(f"outer{fold}_validation", fit, pred, ctx)
        source_outer = source_outer[0] if isinstance(source_outer, tuple) else source_outer
        np.testing.assert_allclose(outer_incumbent, source_outer, atol=1e-7, rtol=1e-6)
        outer_contextual = load_probability(CV / f"outer{fold}_validation_contextual.npz", fit, pred)
        baseline_selected, _, _ = select(outer_incumbent, outer_contextual, ctx["ids"][pred], 0.)
        actual_baseline = 1-outer_incumbent[:, 0] > .55
        np.testing.assert_array_equal(baseline_selected, actual_baseline)
        for weight in WEIGHTS:
            selected, _, _ = select(outer_incumbent, outer_contextual, ctx["ids"][pred], weight)
            metric_rows.append(dict(outer_fold=fold, selected_weight=choice["weight"], candidate_weight=weight,
                                    is_inner_selected=weight == choice["weight"], **metrics(ctx["labels"][pred], ctx["ids"][pred], selected, outer_incumbent)))
        selected, score, _ = select(outer_incumbent, outer_contextual, ctx["ids"][pred], choice["weight"])
        changes = transition(baseline_selected, selected, ctx["labels"][pred])
        for i, index in enumerate(pred):
            transition_rows.append(dict(id=str(ctx["ids"][index]), row_index=int(index), outer_fold=fold,
                Label=int(ctx["labels"][index]), selected_weight=choice["weight"], transition=str(changes[i]),
                incumbent_selected=bool(baseline_selected[i]), rank_blend_selected=bool(selected[i]),
                incumbent_read=float(1-outer_incumbent[i,0]), contextual_read=float(1-outer_contextual[i,0]),
                blend_score=float(score[i]), title=str(ctx["data"].title.iloc[index]), venue=str(ctx["venues"][index])))
    curve = pd.DataFrame(curve_rows); selected = pd.DataFrame(metric_rows).query("is_inner_selected")
    transitions = pd.DataFrame(transition_rows)
    curve.to_csv(OUT / "inner_weight_curves.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(OUT / "choices.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "outer_weight_metrics.csv", index=False)
    selected.to_csv(OUT / "selected_outer_metrics.csv", index=False)
    transitions.to_csv(OUT / "transitions.csv", index=False)
    summary = selected[["H", "Q", "R", "accepted", "TP", "FP", "FN", "missed_label4", "missed_label5"]].mean().to_frame("rank_blend")
    baseline = pd.DataFrame(metric_rows).query("candidate_weight == 0")[summary.index].mean().to_frame("incumbent")
    comparison = baseline.join(summary); comparison["delta"] = comparison.rank_blend-comparison.incumbent
    comparison.to_csv(OUT / "summary.csv")
    counts = transitions.groupby("transition").size().to_frame("count")
    counts.to_csv(OUT / "transition_summary.csv")
    wins = int((selected.H.to_numpy() > pd.DataFrame(metric_rows).query("candidate_weight == 0").sort_values("outer_fold").H.to_numpy()).sum())
    eligible = bool(comparison.loc["H", "delta"] >= .005 and wins >= 3 and comparison.loc["FP", "delta"] <= 0 and comparison.loc["FN", "delta"] <= 0 and comparison.loc["missed_label4", "delta"] <= 0 and comparison.loc["missed_label5", "delta"] <= 0)
    decision = dict(status="PROMISING" if eligible else "REJECT_RANK_BLEND", weights=WEIGHTS, tie_rule="smaller weight",
                    mean_H_delta=float(comparison.loc["H", "delta"]), outer_H_wins=wins, criteria=dict(
                        H_gain_at_least_005=bool(comparison.loc["H", "delta"] >= .005), wins_at_least_3=bool(wins >= 3),
                        FP_no_higher=bool(comparison.loc["FP", "delta"] <= 0), FN_no_higher=bool(comparison.loc["FN", "delta"] <= 0),
                        missed_label4_no_higher=bool(comparison.loc["missed_label4", "delta"] <= 0),
                        missed_label5_no_higher=bool(comparison.loc["missed_label5", "delta"] <= 0)))
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    report = ["# Fixed-budget incumbent/contextual rank blend", "",
              "Each partition keeps exactly the incumbent gate's selection count. Five fixed blend weights are evaluated; one is selected by mean H on the four inner folds, breaking ties toward the smaller contextual weight. Selected papers retain incumbent expected-gain ranking.", "",
              "## Chosen outer results", "", selected.to_markdown(index=False, floatfmt=".6f"), "",
              "## Mean comparison", "", comparison.reset_index().to_markdown(index=False, floatfmt=".6f"), "",
              "## Transitions", "", counts.reset_index().to_markdown(index=False), "",
              "## Decision", "", "```json", json.dumps(decision, indent=2), "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(report), encoding="utf8")
    print(selected.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(comparison.to_string(float_format=lambda x: f"{x:.6f}"))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
