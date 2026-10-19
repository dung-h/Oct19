#!/usr/bin/env python3
"""Produce the preregistered decision report from verified outer OOF results."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

import ndg_lib as n


BASELINE = dict(H=.6402759969, FP=338, FN=313, missed_label4=33, missed_label5=5)


def main():
    verification = json.loads((n.OUT / "verification.json").read_text())
    assert verification["status"] == "PASSED"
    fold = pd.read_csv(n.OUT / "fold_metrics.csv")
    oof = pd.read_csv(n.OUT / "outer_predictions.csv", dtype={"id": str})
    assert len(fold) == 10 and set(fold.arm) == {"incumbent", "meta_gate"}
    summary = fold.groupby("arm").agg(
        mean_H=("H", "mean"), mean_Q=("Q", "mean"), mean_R=("R", "mean"),
        accepted=("accepted", "sum"), dropped=("dropped", "sum"), TP=("TP", "sum"), FP=("FP", "sum"),
        FN=("FN", "sum"), TN=("TN", "sum"), total_error=("total_error", "sum"),
        missed_label4=("missed_label4", "sum"), missed_label5=("missed_label5", "sum"),
        mean_AUC=("read_AUC", "mean"), mean_Brier=("Brier", "mean")).reset_index()
    summary.to_csv(n.OUT / "summary.csv", index=False)
    incumbent = oof[oof.arm == "incumbent"].set_index("id").sort_index()
    candidate = oof[oof.arm == "meta_gate"].set_index("id").sort_index()
    assert incumbent.index.equals(candidate.index) and len(incumbent) == n.N_FIT
    transition = pd.DataFrame(dict(id=incumbent.index, Label=incumbent.Label, outer_fold=incumbent.outer_fold,
                                   incumbent_selected=incumbent.selected, meta_selected=candidate.selected,
                                   incumbent_priority=incumbent.priority, meta_priority=candidate.priority,
                                   incumbent_read=incumbent.read_probability, meta_read=candidate.read_probability))
    positive = transition.Label > 1
    transition["transition"] = np.select(
        [transition.incumbent_selected & ~transition.meta_selected & ~positive,
         ~transition.incumbent_selected & transition.meta_selected & positive,
         ~transition.incumbent_selected & transition.meta_selected & ~positive,
         transition.incumbent_selected & ~transition.meta_selected & positive],
        ["FP_removed", "FN_rescued", "new_FP", "lost_TP"], default="unchanged")
    transition.to_csv(n.OUT / "transitions.csv", index=False)
    counts = transition.transition.value_counts().to_dict()
    base = summary[summary.arm == "incumbent"].iloc[0]
    meta = summary[summary.arm == "meta_gate"].iloc[0]
    h_wins = int((fold[fold.arm == "meta_gate"].set_index("outer_fold").H > fold[fold.arm == "incumbent"].set_index("outer_fold").H).sum())
    criteria = dict(
        mean_H=bool(meta.mean_H >= BASELINE["H"]), h_wins=bool(h_wins >= 3),
        fp=bool(meta.FP <= BASELINE["FP"]), fn=bool(meta.FN <= BASELINE["FN"]),
        one_error_improves=bool(meta.FP < BASELINE["FP"] or meta.FN < BASELINE["FN"]),
        high_labels=bool(meta.missed_label4 <= BASELINE["missed_label4"] and meta.missed_label5 <= BASELINE["missed_label5"]),
        total_error=bool(meta.total_error <= BASELINE["FP"] + BASELINE["FN"]), verifier=True)
    promote = all(criteria.values())
    decision = dict(status="PROMISING_CREATE_FINAL_CANDIDATE" if promote else "REJECT_SCORE_ONLY_GATE",
                    promote=promote, criteria=criteria, h_wins=h_wins,
                    baseline={k: float(base[k]) for k in ["mean_H", "mean_Q", "mean_R", "accepted", "TP", "FP", "FN", "total_error", "missed_label4", "missed_label5"]},
                    candidate={k: float(meta[k]) for k in ["mean_H", "mean_Q", "mean_R", "accepted", "TP", "FP", "FN", "total_error", "missed_label4", "missed_label5"]},
                    transitions={str(k): int(v) for k, v in counts.items()}, uploaded=False)
    (n.OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (n.HERE / "DECISION.md").write_text(
        "# Quyết định nested disagreement gate\n\n"
        f"**{decision['status']}**. Mean outer H: incumbent {base.mean_H:.6f}, meta gate {meta.mean_H:.6f}; "
        f"meta thắng {h_wins}/5 folds. FP/FN: {int(base.FP)}/{int(base.FN)} → {int(meta.FP)}/{int(meta.FN)}.\n\n"
        f"Tiêu chí: `{json.dumps(criteria, ensure_ascii=False)}`.\n\n"
        "Không upload Kaggle. Nếu reject, dừng các biến thể score-only đúng theo PLAN.md.\n")
    (n.HERE / "RESULTS.md").write_text(
        "# Nested disagreement gate — verified outer OOF\n\n"
        "| arm | mean H | mean Q | mean R | accepted | TP | FP | FN | errors | missed L4 | missed L5 | AUC | Brier |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(f"| {r.arm} | {r.mean_H:.6f} | {r.mean_Q:.6f} | {r.mean_R:.6f} | {int(r.accepted)} | {int(r.TP)} | {int(r.FP)} | {int(r.FN)} | {int(r.total_error)} | {int(r.missed_label4)} | {int(r.missed_label5)} | {r.mean_AUC:.6f} | {r.mean_Brier:.6f} |" for r in summary.itertuples()) +
        "\n\n" + f"Meta gate thắng H ở {h_wins}/5 outer folds. Chuyển trạng thái: `{json.dumps(counts)}`.\n\n"
        f"Decision: **{decision['status']}**. Independent verifier: PASSED (25 source partitions, feature/model/meta replay).\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
