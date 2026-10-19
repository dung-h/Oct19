#!/usr/bin/env python3
"""Audit complementarity of saved OOF decisions; this script trains no model.

Every source must use the same pre-existing grouped outer-fold row assignment.
The result is a diagnostic of already inspected OOF predictions. It neither
chooses a model nor creates a submission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
OUT = HERE / "outputs"

# Chosen before inspecting this audit: one representative decision stream per
# prior modelling direction. These are alternatives for error diagnosis, not a
# leaderboard of models or a blend search.
SOURCES = {
    "title_margin": (EXPERIMENTS / "abstract_preference_margin/outputs/oof_predictions.csv", "long", "title_margin", "priority"),
    "abstract_margin": (EXPERIMENTS / "abstract_preference_margin/outputs/oof_predictions.csv", "long", "abstract_margin", "priority"),
    "content_flags": (EXPERIMENTS / "preference_augmentation/outputs/oof_predictions.csv", "wide", None, "content_priority"),
    "history_neighbors": (EXPERIMENTS / "preference_augmentation/outputs/oof_predictions.csv", "wide", None, "neighbors_priority"),
    "ordinal": (EXPERIMENTS / "ordinal_preference/outputs/oof_predictions.csv", "wide", None, "ordinal_priority"),
    "specter_title": (EXPERIMENTS / "input_ablation/outputs/oof_predictions.csv", "wide", None, "specter_title_priority"),
    "venue_metadata": (EXPERIMENTS / "input_ablation/metadata_outputs/oof_predictions.csv", "wide", None, "venue_priority"),
    "adapter_random": (EXPERIMENTS / "preference_adapter/outputs/oof_predictions.csv", "wide", None, "adapter_random_priority"),
    "neural_adapted": (EXPERIMENTS / "specter2_preference/outputs/oof_predictions.csv", "wide", None, "adapted_priority"),
    "retrieval_rbf": (EXPERIMENTS / "diagnose_bottleneck/outputs/oof_predictions.csv", "wide", None, "retrieval_rbf_priority"),
    "claims": (EXPERIMENTS / "claim_stability/outputs/oof_predictions.csv", "wide", None, "claims_seed20261018_priority"),
}
INCUMBENT = (EXPERIMENTS / "abstract_preference_margin/outputs/oof_predictions.csv", "reference")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_metrics(frame: pd.DataFrame, priority: np.ndarray) -> dict[str, float | int]:
    """Official metric evaluated separately in every hidden-like outer fold."""
    fold_values = []
    total = dict(accepted=0, TP=0, FP=0, FN=0, total_error=0, missed_high=0)
    for fold in sorted(frame.outer_fold.unique()):
        mask = frame.outer_fold.to_numpy(int) == int(fold)
        metrics = _single_fold_metrics(frame.loc[mask], priority[mask])
        fold_values.append(metrics)
        for key in total:
            total[key] += int(metrics[key])
    return dict(H=float(np.mean([x["H"] for x in fold_values])),
                Q=float(np.mean([x["Q"] for x in fold_values])),
                R=float(np.mean([x["R"] for x in fold_values])), **total)


def _single_fold_metrics(frame: pd.DataFrame, priority: np.ndarray) -> dict[str, float | int]:
    y = frame.Label.to_numpy(int)
    ids = frame.id.astype(str).to_numpy()
    selected = priority > 0
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-float(priority[i]), ids[i])), int)
    weights = 1.0 / np.log2(np.arange(len(order), dtype=float) + 2)
    q = float(np.average((y[order] - 1) / 4, weights=weights)) if len(order) else 0.0
    tp = int(np.sum(selected & (y > 1)))
    fp = int(np.sum(selected & (y == 1)))
    fn = int(np.sum(~selected & (y > 1)))
    r = float(tp / np.sum(y > 1))
    h = float(2 * q * r / (q + r)) if q + r else 0.0
    return dict(H=h, Q=q, R=r, accepted=int(selected.sum()), TP=tp, FP=fp, FN=fn,
                total_error=fp + fn, missed_high=int(np.sum(~selected & (y >= 4))))


def load_baseline() -> tuple[pd.DataFrame, dict[str, str]]:
    path, arm = INCUMBENT
    raw = pd.read_csv(path, dtype={"id": str})
    frame = raw.loc[raw.arm.eq(arm), ["id", "outer_fold", "Label", "title", "venue", "priority", "selected"]].copy()
    assert len(frame) == 2494 and frame.id.is_unique and frame.Label.between(1, 5).all()
    assert np.array_equal(frame.priority.to_numpy() > 0, frame.selected.to_numpy(bool))
    frame = frame.sort_values("id").reset_index(drop=True)
    return frame, {str(path.relative_to(ROOT)): sha256(path)}


def load_candidate(path: Path, kind: str, arm: str | None, col: str, baseline: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"id": str})
    if kind == "long":
        assert arm is not None
        raw = raw.loc[raw.arm.eq(arm)]
    fields = ["id", "outer_fold", "Label", col]
    assert set(fields).issubset(raw.columns), (path, fields, raw.columns.tolist())
    out = raw[fields].rename(columns={col: "priority"}).copy()
    assert len(out) == 2494 and out.id.is_unique and np.isfinite(out.priority).all() and (out.priority >= 0).all()
    out = out.sort_values("id").reset_index(drop=True)
    np.testing.assert_array_equal(out.id.to_numpy(), baseline.id.to_numpy())
    np.testing.assert_array_equal(out.outer_fold.to_numpy(int), baseline.outer_fold.to_numpy(int))
    np.testing.assert_array_equal(out.Label.to_numpy(int), baseline.Label.to_numpy(int))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base, hashes = load_baseline()
    base_priority = base.priority.to_numpy(float)
    base_selected = base.selected.to_numpy(bool)
    labels = base.Label.to_numpy(int)
    positive = labels > 1
    old_fp = base_selected & ~positive
    old_tp = base_selected & positive
    old_fn = ~base_selected & positive
    assert old_fp.sum() == 338 and old_tp.sum() == 1278 and old_fn.sum() == 313
    expected = official_metrics(base, base_priority)
    np.testing.assert_allclose([expected["H"], expected["Q"]], [0.6402759969159176, 0.5324997993087505], rtol=0, atol=3e-7)
    assert 0.80 < expected["R"] < 0.81

    candidate_priority: dict[str, np.ndarray] = {}
    source_meta = []
    for name, (path, kind, arm, col) in SOURCES.items():
        view = load_candidate(path, kind, arm, col, base)
        candidate_priority[name] = view.priority.to_numpy(float)
        relative = str(path.relative_to(ROOT))
        hashes[relative] = sha256(path)
        source_meta.append(dict(model=name, source=relative, source_sha256=hashes[relative], field=col, arm=arm or ""))

    rows, detailed = [], []
    for name, priority in candidate_priority.items():
        selected = priority > 0
        metrics = official_metrics(base, priority)
        # A veto is a counterfactual diagnostic: it can only deselect papers
        # selected by incumbent. It has no new positive selection, so its exact
        # cost/benefit is easy to inspect without tuning a blend weight.
        veto = base_selected & selected
        veto_priority = np.where(veto, base_priority, 0.0)
        veto_metrics = official_metrics(base, veto_priority)
        removed_fp = int(np.sum(old_fp & ~selected))
        lost_tp = int(np.sum(old_tp & ~selected))
        rescued_fn = int(np.sum(old_fn & selected))
        new_fp = int(np.sum(~base_selected & ~positive & selected))
        moved = int(np.sum(base_selected != selected))
        denom = removed_fp + lost_tp
        rows.append(dict(model=name, **metrics, delta_H=metrics["H"] - expected["H"],
                         incumbent_FP_removed=removed_fp, incumbent_TP_lost=lost_tp,
                         incumbent_FN_rescued=rescued_fn, new_FP=new_fp, decision_changes=moved,
                         veto_H=veto_metrics["H"], veto_Q=veto_metrics["Q"], veto_R=veto_metrics["R"],
                         veto_FP_removed=removed_fp, veto_TP_lost=lost_tp,
                         veto_net_correct=removed_fp - lost_tp,
                         veto_precision=float(removed_fp / denom) if denom else np.nan))
        for i, paper_id in enumerate(base.id):
            if old_fp[i] or old_fn[i]:
                detailed.append(dict(model=name, id=paper_id, outer_fold=int(base.outer_fold.iloc[i]),
                                     Label=int(labels[i]), venue=base.venue.iloc[i], title=base.title.iloc[i],
                                     incumbent_error="FP" if old_fp[i] else "FN",
                                     candidate_selected=bool(selected[i]), candidate_priority=float(priority[i]),
                                     change=("FP_removed" if old_fp[i] and not selected[i] else
                                             "FN_rescued" if old_fn[i] and selected[i] else "unchanged")))
    summary = pd.DataFrame(rows).sort_values(["veto_net_correct", "delta_H"], ascending=False)
    summary.to_csv(OUT / "candidate_veto_summary.csv", index=False)
    pd.DataFrame(source_meta).to_csv(OUT / "source_manifest.csv", index=False)
    detail = pd.DataFrame(detailed)
    detail.to_csv(OUT / "incumbent_error_by_model.csv", index=False)

    # Across different model families, count whether an incumbent FP remains
    # selected. This quantifies consensus rather than assuming one model's
    # selected set is ground truth.
    select_matrix = np.column_stack([candidate_priority[name] > 0 for name in SOURCES])
    names = list(SOURCES)
    consensus = base.loc[old_fp, ["id", "outer_fold", "Label", "title", "venue"]].copy()
    fp_indices = np.flatnonzero(old_fp)
    consensus["models_selecting"] = select_matrix[fp_indices].sum(axis=1)
    consensus["models_rejecting"] = len(names) - consensus.models_selecting
    for j, name in enumerate(names):
        consensus[f"selected_{name}"] = select_matrix[fp_indices, j]
    consensus = consensus.sort_values(["models_selecting", "venue", "id"], ascending=[False, True, True])
    consensus.to_csv(OUT / "incumbent_fp_consensus.csv", index=False)
    bucket = consensus.groupby(["models_rejecting", "venue"], dropna=False).size().rename("count").reset_index()
    bucket.to_csv(OUT / "fp_consensus_by_venue.csv", index=False)
    histogram = consensus.models_rejecting.value_counts().sort_index().rename_axis("models_rejecting").reset_index(name="incumbent_FP_count")
    histogram.to_csv(OUT / "fp_consensus_histogram.csv", index=False)
    venue_consensus = consensus.groupby("venue", sort=False).agg(
        incumbent_FP=("id", "size"),
        selected_by_all_11=("models_rejecting", lambda x: int((x == 0).sum())),
        rejected_by_at_least_5=("models_rejecting", lambda x: int((x >= 5).sum())),
        mean_models_rejecting=("models_rejecting", "mean"),
    )
    venue_consensus["persistent_share"] = venue_consensus.selected_by_all_11 / venue_consensus.incumbent_FP
    venue_consensus = venue_consensus.sort_values("incumbent_FP", ascending=False)
    venue_consensus.to_csv(OUT / "fp_consensus_venue_summary.csv")
    # The threshold sequence is deliberately fully reported rather than using
    # its best OOF row. It describes how much shared disagreement exists; a
    # threshold selected here would be an outer-label-tuned ensemble.
    reject_count = (~select_matrix).sum(axis=1)
    vote_rows = []
    for needed_rejections in range(1, len(names) + 1):
        voted = base_selected & (reject_count < needed_rejections)
        voted_priority = np.where(voted, base_priority, 0.0)
        metrics = official_metrics(base, voted_priority)
        removed_fp = int(np.sum(old_fp & ~voted))
        lost_tp = int(np.sum(old_tp & ~voted))
        vote_rows.append(dict(rejections_needed_to_veto=needed_rejections, **metrics,
                              delta_H=metrics["H"] - expected["H"],
                              incumbent_FP_removed=removed_fp,
                              incumbent_TP_lost=lost_tp,
                              net_correct_decisions=removed_fp - lost_tp,
                              veto_precision=float(removed_fp / (removed_fp + lost_tp))))
    votes = pd.DataFrame(vote_rows)
    votes.to_csv(OUT / "multi_model_vote_diagnostic.csv", index=False)

    # Pairwise agreement on decisions among heterogeneous saved candidates.
    agreement_rows = []
    for left_i, left in enumerate(names):
        for right_i in range(left_i + 1, len(names)):
            right = names[right_i]
            a, b = select_matrix[:, left_i], select_matrix[:, right_i]
            agreement_rows.append(dict(left=left, right=right, agreement=float(np.mean(a == b)),
                                       both_reject_incumbent_fp=int(np.sum(old_fp & ~a & ~b)),
                                       both_drop_incumbent_tp=int(np.sum(old_tp & ~a & ~b))))
    pd.DataFrame(agreement_rows).to_csv(OUT / "candidate_pair_agreement.csv", index=False)

    # Source inputs must not change during calculation.
    for rel, digest in hashes.items():
        assert sha256(ROOT / rel) == digest, rel
    verifier = dict(status="PASSED", models=len(SOURCES), rows=2494,
                    incumbent_FP=338, incumbent_TP=1278, incumbent_FN=313,
                    exact_outer_fold_and_label_alignment=True,
                    source_hashes_unchanged=True, training_fits=0,
                    final_model_selection=False, uploaded=False)
    (OUT / "verification.json").write_text(json.dumps(verifier, indent=2) + "\n")
    report = [
        "# Cross-model false-positive audit",
        "",
        "This is a diagnostic of saved out-of-fold predictions from already examined folds. It trains no model, searches no ensemble weight, selects no final candidate, and does not use test labels.",
        "",
        "## Baseline",
        "",
        "The incumbent has 338 selected Label1 papers (FP), 1,278 selected worthwhile papers (TP), and 313 dropped worthwhile papers (FN). Every candidate was required to have exactly the same 2,494 IDs, labels and outer-fold assignment before comparison.",
        "",
        "## Can another model act as a veto?",
        "",
        "`incumbent_FP_removed` is the number of incumbent FP that the candidate rejects. `incumbent_TP_lost` is the cost: incumbent TP that it also rejects. `veto_*` scores retain the incumbent order and select only papers selected by both models; this is an explanatory counterfactual, not a tuned ensemble.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Persistent incumbent false positives",
        "",
        "The histogram below counts, for each incumbent FP, how many of the 11 distinct candidate streams reject it. A high rejection count indicates cross-model disagreement with the incumbent; zero means every tested stream still selects the paper. It does not prove a paper should be rejected, because candidates can share blind spots or use different thresholds.",
        "",
        histogram.to_markdown(index=False),
        "",
        "The shared errors are uneven by venue. This is descriptive: venue is already a model feature and the table does not establish a causal venue effect.",
        "",
        venue_consensus.reset_index().to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Untuned multi-model vote diagnostic",
        "",
        "For transparency, the table reports every possible vote threshold. A paper is removed from the incumbent list only if at least `rejections_needed_to_veto` of the 11 streams reject it; incumbent ranking stays unchanged. The table is diagnostic only: choosing a threshold by its highest row would tune on the same outer labels and is therefore invalid for promotion.",
        "",
        votes.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Case-level IDs/titles are in `incumbent_fp_consensus.csv`; per-model outcomes for all incumbent FP/FN are in `incumbent_error_by_model.csv`. Pairwise agreement is in `candidate_pair_agreement.csv`.",
        "",
        "## Interpretation boundary",
        "",
        "The result can justify a pre-registered, nested-CV ensemble experiment only when a candidate removes materially more incumbent FP than incumbent TP and the pattern is not merely one threshold tradeoff. It cannot justify choosing the best row above or writing ID-specific rules: these OOF labels and folds have already been inspected.",
        "",
        "Verification passed: all sources align exactly on ID/label/fold and their hashes did not change during analysis. No Kaggle upload.",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf8")
    print(summary[["model", "incumbent_FP_removed", "incumbent_TP_lost", "veto_net_correct", "veto_H", "delta_H"]].to_string(index=False))
    print(json.dumps(verifier, indent=2))


if __name__ == "__main__":
    main()
