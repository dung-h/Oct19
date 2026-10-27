#!/usr/bin/env python3
"""Evaluate held-out predictions only after BOTH fixed partitions finish."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

from common import HERE, check_probability, digest, fingerprint, jsonl, load_inputs, official, write_json


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction(root, tag, arm, indices, docs):
    directory = root / tag
    complete = json.loads((directory / "COMPLETE.json").read_text())
    lock = json.loads((directory / "runtime_lock.json").read_text())
    assert lock["fingerprint"] == fingerprint() and lock["partition"] == tag
    assert lock["mode"] == "train"
    assert complete["status"] == "PREDICTIONS_COMPLETE" and complete["lock_sha"] == digest(lock)
    rows = jsonl(directory / f"{arm}.jsonl")
    assert [r["index"] for r in rows] == indices
    assert all(r["id"] == docs[r["index"]]["id"] and r["lock_sha"] == digest(lock) for r in rows)
    return check_probability([r["probabilities"] for r in rows], len(indices))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/pilot_v1")
    args = parser.parse_args()
    root = args.output.resolve()
    config = json.loads((HERE / "config.json").read_text())
    docs, targets, parts, base = load_inputs()
    # Validate and load every prediction arm before evaluating any held-out target.
    predictions = {tag: {arm: prediction(root, tag, arm, parts[tag]["pred"], docs)
                         for arm in ("frozen", "trained")} for tag in config["partitions"]}
    metrics, confusions, cases, transitions, class_metrics = [], [], [], [], []
    gain = np.arange(5) / 4
    cutoff = config["evaluation"]["read_cutoff"]
    for tag in config["partitions"]:
        pred = parts[tag]["pred"]
        ids = np.array([docs[i]["id"] for i in pred], str)
        y = np.array([targets[i] for i in pred])
        incumbent = np.array(base[tag]["probabilities"])
        selected_base = 1-incumbent[:, 0] > cutoff
        incumbent_gain = incumbent @ gain
        for arm, p in dict(incumbent=incumbent, **predictions[tag]).items():
            read, expected_gain = 1-p[:, 0], p @ gain
            selected = read > cutoff
            matched = np.zeros(len(pred), bool)
            matched[np.lexsort((ids, -read))[:int(selected_base.sum())]] = True
            policies = dict(native=np.where(selected, 1+9*expected_gain, 0),
                            selection_only=np.where(selected, 1+9*incumbent_gain, 0),
                            ranking_only=np.where(selected_base, 1+9*expected_gain, 0),
                            matched_budget=np.where(matched, 1+9*incumbent_gain, 0))
            for policy, priority in policies.items():
                metrics.append(dict(partition=tag, arm=arm, policy=policy,
                    **official(y, priority, ids), read_AUC=float(roc_auc_score(y > 1, read)),
                    read_Brier=float(np.mean((read-(y > 1))**2)),
                    rating_logloss=float(log_loss(y, p, labels=[1, 2, 3, 4, 5]))))
            cm = confusion_matrix(y, p.argmax(1)+1, labels=[1, 2, 3, 4, 5])
            for a in range(5):
                for b in range(5):
                    confusions.append(dict(partition=tag, arm=arm, true_label=a+1, predicted_label=b+1,
                                           count=int(cm[a, b]), fraction_of_true=float(cm[a, b]/cm[a].sum())))
                tp = int(cm[a, a])
                class_metrics.append(dict(partition=tag, arm=arm, label=a+1, support=int(cm[a].sum()),
                    predicted_count=int(cm[:, a].sum()), recall=float(tp/cm[a].sum()),
                    precision=float(tp/cm[:, a].sum()) if cm[:, a].sum() else 0.0))
            for policy, chosen in (("native", selected), ("matched_budget", matched)):
                for label in range(1, 6):
                    mask = y == label
                    transitions.append(dict(partition=tag, arm=arm, policy=policy, label=label,
                        added=int((mask & ~selected_base & chosen).sum()),
                        dropped=int((mask & selected_base & ~chosen).sum()),
                        incumbent_selected=int((mask & selected_base).sum()), new_selected=int((mask & chosen).sum())))
            for position, index in enumerate(pred):
                cases.append(dict(partition=tag, arm=arm, id=ids[position], true_label=int(y[position]),
                    predicted_label=int(p[position].argmax()+1), selected=bool(selected[position]),
                    incumbent_selected=bool(selected_base[position]), matched_selected=bool(matched[position]),
                    read_probability=float(read[position]), title=docs[index]["title"],
                    venue=docs[index]["venue"], authors=docs[index]["authors"],
                    **{f"p{k+1}": float(p[position, k]) for k in range(5)}))
    tables = dict(partition_metrics=metrics, confusion=confusions, class_metrics=class_metrics,
                  label_transitions=transitions, cases=cases)
    for name, rows in tables.items():
        save_csv(root / f"{name}.csv", rows)
    lookup = {(r["partition"], r["arm"], r["policy"]): r for r in metrics}
    summary = []
    for arm in ("incumbent", "frozen", "trained"):
        for policy in ("native", "selection_only", "ranking_only", "matched_budget"):
            rows = [lookup[t, arm, policy] for t in config["partitions"]]
            means = {k: float(np.mean([r[k] for r in rows])) for k in
                     ("H", "Q", "R", "read_AUC", "read_Brier", "rating_logloss")}
            counts = {k: sum(r[k] for r in rows) for k in
                      ("selected", "TP", "FP", "FN", "TN", "missed_label4", "missed_label5")}
            summary.append(dict(arm=arm, policy=policy, **means, **counts))
    save_csv(root / "summary.csv", summary)
    summary_map = {(r["arm"], r["policy"]): r for r in summary}
    trained, frozen, inc = [summary_map[a, "native"] for a in ("trained", "frozen", "incumbent")]
    deltas = {t: lookup[t, "trained", "native"]["H"]-lookup[t, "incumbent", "native"]["H"]
              for t in config["partitions"]}
    gate = config["evaluation"]
    fp_saved = summary_map["incumbent", "native"]["FP"]-summary_map["trained", "matched_budget"]["FP"]
    missed_changes = {f"{policy}_{key}": summary_map["trained", policy][key]-inc[key]
                      for policy in ("native", "matched_budget") for key in ("missed_label4", "missed_label5")}
    learned = trained["rating_logloss"] < frozen["rating_logloss"] and trained["read_AUC"] >= frozen["read_AUC"]
    performance = (trained["H"]-inc["H"] >= gate["mean_H_improvement_min"] and
                   min(deltas.values()) >= gate["each_partition_H_change_min"] and
                   fp_saved >= gate["matched_budget_FP_reduction_min"])
    # Explicit high-rating checks avoid conflating class-4 and class-5 misses.
    performance = performance and all(
        value <= gate["missed_label4_change_max" if key.endswith("label4") else "missed_label5_change_max"]
        for key, value in missed_changes.items())
    decision = dict(interpretation="Exploratory screen on repeatedly inspected inherited folds; not promotion",
                    learned_signal=bool(learned), performance_gate=bool(performance),
                    continue_confirmation=bool(learned and performance), per_partition_H_delta=deltas,
                    mean_H_delta=trained["H"]-inc["H"], matched_budget_FP_saved=fp_saved,
                    missed_changes=missed_changes, next_action=("Run locked confirmation protocol" if learned and performance
                        else "Diagnose selection/ranking and calibration; do not train full-data or replace incumbent"))
    write_json(root / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
