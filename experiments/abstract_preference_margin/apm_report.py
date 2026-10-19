"""OOF counts, assignment metrics, transitions and predeclared decision."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from apm_context import (ARMS, N_FIT, OUT, PRIMARY, READ_THRESHOLD, context,
                         source_probabilities, outer_jobs)
from apm_metrics import calibration_bins, official_metrics, priorities, probability_metrics


def _top_m(selected_scores, m, ids):
    order = np.asarray(sorted(range(len(ids)), key=lambda ix: (-selected_scores[ix], str(ids[ix]))), int)
    selected = np.zeros(len(ids), dtype=bool)
    selected[order[:m]] = True
    return selected


def generate():
    verification_path = OUT/"verification.json"
    assert verification_path.is_file()
    verification = json.loads(verification_path.read_text())
    assert verification["status"] == "PASSED"
    ctx = context()
    fold_rows, calibration, oof_rows, count_labels, count_venues = [], [], [], [], []
    transition_rows, matched_rows = [], []

    for fold, fit, pred, _ in outer_jobs(ctx):
        name = f"outer{fold}_validation"
        with np.load(OUT/f"{name}_predictions.npz") as z:
            saved_fit, saved_pred = z["fit"], z["pred"]
            np.testing.assert_array_equal(saved_fit, fit)
            np.testing.assert_array_equal(saved_pred, pred)
            probs = {arm:z[arm].copy() for arm in ARMS}
        labels = ctx["labels"][pred]
        ids = ctx["ids"][pred]
        venues = ctx["venues"][pred]
        incumbent = source_probabilities(name, fit, pred, ctx)
        base_selected = 1.0-incumbent[:,0] > READ_THRESHOLD
        m = int(base_selected.sum())
        fold_selected = {}
        for arm in ARMS:
            read = 1.0-probs[arm][:,0]
            selected, priority = priorities(read, incumbent)
            fold_selected[arm] = selected
            row = dict(outer_fold=int(fold),arm=arm,threshold=READ_THRESHOLD,
                       **official_metrics(labels,priority,ids),**probability_metrics(labels,probs[arm]))
            fold_rows.append(row)
            calibration.extend(calibration_bins(labels,read,arm,fold))

            for label in range(1,6):
                mask = labels == label
                count_labels.append(dict(outer_fold=int(fold),arm=arm,Label=label,total=int(mask.sum()),
                    selected=int((selected&mask).sum()),dropped=int((~selected&mask).sum())))
            for venue in sorted(set(venues)):
                mask = venues == venue
                positives = labels > 1
                count_venues.append(dict(outer_fold=int(fold),arm=arm,venue=venue,total=int(mask.sum()),
                    selected=int((selected&mask).sum()),FP=int((selected&mask&(labels==1)).sum()),
                    FN=int((~selected&mask&positives).sum()),
                    missed_high=int((~selected&mask&(labels>=4)).sum())))

            for ix, idx in enumerate(pred):
                oof_rows.append(dict(id=ctx["ids"][idx],outer_fold=int(fold),arm=arm,
                    Label=int(labels[ix]),title=ctx["data"].title.iloc[idx],venue=venues[ix],
                    abstract_available=bool(ctx["abstract_available"][idx]),
                    abstract_truncated=bool(ctx["abstract_truncated"][idx]),
                    read_probability=float(read[ix]),selected=bool(selected[ix]),
                    priority=float(priority[ix]),**{f"p{j+1}":float(probs[arm][ix,j]) for j in range(5)}))

            matched = _top_m(read,m,ids)
            matched_priority = np.where(matched,1+9*(incumbent@(np.arange(5)/4)),0)
            matched_row = dict(outer_fold=int(fold),arm=arm,reference_selected=m,
                               **official_metrics(labels,matched_priority,ids))
            matched_rows.append(matched_row)

        old = fold_selected["reference"]
        for arm in ARMS[1:]:
            now = fold_selected[arm]
            positive = labels > 1
            transition = np.select(
                [~old&now&positive, old&~now&~positive, ~old&now&~positive, old&~now&positive],
                ["FN_rescued","FP_removed","new_FP","lost_TP"], default="unchanged")
            for ix, idx in enumerate(pred):
                transition_rows.append(dict(id=ctx["ids"][idx],outer_fold=int(fold),arm=arm,
                    Label=int(labels[ix]),transition=str(transition[ix]),
                    incumbent_selected=bool(old[ix]),candidate_selected=bool(now[ix]),
                    title=ctx["data"].title.iloc[idx],venue=venues[ix]))

    folds = pd.DataFrame(fold_rows)
    assert len(folds)==20
    folds.to_csv(OUT/"fold_metrics.csv",index=False)
    pd.DataFrame(calibration).to_csv(OUT/"calibration_bins.csv",index=False)
    pd.DataFrame(oof_rows).to_csv(OUT/"oof_predictions.csv",index=False)
    pd.DataFrame(count_labels).to_csv(OUT/"counts_by_label.csv",index=False)
    pd.DataFrame(count_venues).to_csv(OUT/"counts_by_venue.csv",index=False)
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(OUT/"failure_transitions.csv",index=False)
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(OUT/"matched_budget_diagnostics.csv",index=False)

    summary = folds.groupby("arm",sort=False).agg(
        accepted=("accepted","sum"),dropped=("dropped","sum"),FP=("FP","sum"),FN=("FN","sum"),
        total_error=("total_error","sum"),missed_high=("missed_high","sum"),
        missed_label4=("missed_label4","sum"),missed_label5=("missed_label5","sum"),
        H=("H","mean"),Q=("Q","mean"),R=("R","mean"),AUC=("AUC","mean"),
        Brier=("Brier","mean"),logloss=("logloss","mean"))
    summary.to_csv(OUT/"summary.csv")
    transitions_summary = transitions.groupby(["arm","transition"],sort=False).size().unstack(fill_value=0)
    for col in ["FN_rescued","FP_removed","new_FP","lost_TP","unchanged"]:
        if col not in transitions_summary: transitions_summary[col] = 0
    transitions_summary = transitions_summary[["FN_rescued","FP_removed","new_FP","lost_TP","unchanged"]]
    transitions_summary["delta_error"] = transitions_summary.new_FP + transitions_summary.lost_TP - transitions_summary.FN_rescued - transitions_summary.FP_removed
    transitions_summary.to_csv(OUT/"transition_summary.csv")

    paired = folds.pivot(index="outer_fold",columns="arm",values="total_error")
    primary, baseline = summary.loc[PRIMARY],summary.loc["reference"]
    wins = int((paired[PRIMARY] < paired["reference"]).sum())
    criteria = dict(
        fp_no_higher=bool(primary.FP<=338), fn_no_higher=bool(primary.FN<=313),
        one_of_fp_fn_strict=bool(primary.FP<338 or primary.FN<313),
        label4_fn_no_higher=bool(primary.missed_label4<=33),
        label5_fn_no_higher=bool(primary.missed_label5<=5),
        lower_error_in_at_least_four_folds=bool(wins>=4),
        mean_fold_H_no_lower=bool(primary.H>=0.6402759969),
        independent_verification_passed=True,
    )
    passed = all(criteria.values())
    decision = dict(status="promising" if passed else "negative",
        primary=PRIMARY,criteria=criteria,folds_with_lower_error=wins,
        incumbent=dict(FP=int(baseline.FP),FN=int(baseline.FN),missed_label4=int(baseline.missed_label4),
                       missed_label5=int(baseline.missed_label5),total_error=int(baseline.total_error),H=float(baseline.H)),
        primary_result=dict(FP=int(primary.FP),FN=int(primary.FN),missed_label4=int(primary.missed_label4),
                       missed_label5=int(primary.missed_label5),total_error=int(primary.total_error),H=float(primary.H)),
        incumbent_preserved=True,uploaded=False)
    decision["message"] = ("Primary đạt tiêu chí screening; có thể refit final theo protocol."
                            if passed else "Primary không đạt đầy đủ tiêu chí đã khóa; giữ incumbent và không refit final.")
    (OUT/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")

    matched_summary=matched.groupby("arm",sort=False).agg(accepted=("accepted","sum"),FP=("FP","sum"),FN=("FN","sum"),
        missed_label4=("missed_label4","sum"),missed_label5=("missed_label5","sum"),total_error=("total_error","sum"),
        H=("H","mean"),Q=("Q","mean"),R=("R","mean"))
    matched_summary.to_csv(OUT/"matched_budget_summary.csv")
    _write_results(summary,transitions_summary,matched_summary,decision)
    print(summary.to_string(float_format=lambda value:f"{value:.5f}"),flush=True)
    print(transitions_summary.to_string(),flush=True)
    print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
    return decision


def _write_results(summary,transitions,matched,decision):
    result = ["# Abstract preference margin: results", "", decision["message"], "",
        "H/Q/R là trung bình của năm outer folds. Counts/FP/FN là tổng OOF trên2.494 train rows. "
        "Read gate strict >0.55; ranking giữ incumbent expected gain.", "",
        "## Tất cả các nhánh", "", summary.reset_index().to_markdown(index=False,floatfmt=".5f"), "",
        "## Thay đổi lỗi so với incumbent", "", transitions.reset_index().to_markdown(index=False), "",
        "## Chẩn đoán với cùng số lượng paper được chọn", "",
        matched.reset_index().to_markdown(index=False,floatfmt=".5f"), "",
        "## Quyết định đã khóa", "", "```json", json.dumps(decision,ensure_ascii=False,indent=2), "```", "",
        "Các outer folds đã được xem trong nghiên cứu trước; kết quả paired là exploratory, "
        "không phải holdout độc lập. Không dùng test labels hoặc upload Kaggle. Missing-abstract "
        "queries dùng nguyên incumbent probability. Xem `failure_transitions.csv`, "
        "`counts_by_label.csv`, `counts_by_venue.csv` và `matched_budget_diagnostics.csv` để audit case-level.", ""]
    (OUT/"RESULTS.md").write_text("\n".join(result),encoding="utf8")

