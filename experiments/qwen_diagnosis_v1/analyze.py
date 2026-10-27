"""Diagnose completed Qwen predictions without changing the locked pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "qwen_remote_v1"
sys.path.insert(0, str(PILOT))
from common import digest, fingerprint, jsonl, load_inputs, official, sha, write_json
from evaluate import prediction


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(spearmanr(x, y).statistic) if len(x) > 2 and np.ptp(x) and np.ptp(y) else None


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def metrics(y, p):
    p = np.asarray(p, float)
    p = p / p.sum(1, keepdims=True)
    read = 1 - p[:, 0]
    gain = p @ (np.arange(5) / 4)
    positive = y > 1
    return dict(n=len(y), rating_logloss=float(log_loss(y, p, labels=[1, 2, 3, 4, 5])),
                read_AUC=auc(positive, read), read_Brier=float(np.mean((read-positive)**2)),
                predicted_read=float(read.mean()), actual_read=float(positive.mean()),
                ordinal_spearman=correlation(y, gain),
                positive_ordinal_spearman=correlation(y[positive], gain[positive]),
                positive_conditional_gain_spearman=correlation(y[positive], (gain/read)[positive]),
                positive_rating45_AUC=auc(y[positive] >= 4, gain[positive]),
                positive_rating5_AUC=auc(y[positive] == 5, gain[positive]))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--source-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=HERE / "outputs/diagnosis_v1")
    args = ap.parse_args()
    source, out = args.source_output.resolve(), args.output.resolve()
    if (out / "analysis.json").exists():
        raise RuntimeError("Analysis exists; use a new output namespace")
    docs, targets, parts, base = load_inputs()
    predictions = {tag: {arm: prediction(source, tag, arm, part["pred"], docs)
                         for arm in ("frozen", "trained")} for tag, part in parts.items()}
    out.mkdir(parents=True, exist_ok=True)
    report = dict(source=str(source), source_fingerprint=fingerprint(),
                  protocol_sha256=sha(HERE / "PROTOCOL.md"), partitions={})
    metric_rows, calibration, subgroup_rows, changes = [], [], [], []
    for tag, part in parts.items():
        y = np.array([targets[i] for i in part["pred"]])
        fit_y = np.array([targets[i] for i in part["fit"]])
        prior = np.bincount(fit_y, minlength=6)[1:] / len(fit_y)
        arms = dict(fit_prior=np.tile(prior, (len(y), 1)),
                    incumbent=np.array(base[tag]["probabilities"]), **predictions[tag])
        record = dict(fit_prior=prior.tolist(), heldout_prior=(np.bincount(y, minlength=6)[1:]/len(y)).tolist(), arms={})
        present = np.array([bool(docs[i]["abstract"].strip()) for i in part["pred"]])
        audit = {r["index"]: r for r in json.loads((source/tag/"prompt_audit.json").read_text())}
        truncated = np.array([audit[i]["abstract"]["raw"] > audit[i]["abstract"]["kept"] for i in part["pred"]])
        for arm, p in arms.items():
            read, gain = 1-p[:, 0], p @ (np.arange(5)/4)
            m = metrics(y, p)
            metric_rows.append(dict(partition=tag, arm=arm, **m))
            centered = p-p.mean(0)
            eigen = np.linalg.svd(centered, compute_uv=False)**2
            cond = p[:, 1:] / read[:, None]
            record["arms"][arm] = dict(**m, probability_mean=p.mean(0).tolist(),
                probability_std=p.std(0).tolist(), conditional_positive_mean=cond.mean(0).tolist(),
                conditional_positive_std=cond.std(0).tolist(),
                argmax_counts=np.bincount(p.argmax(1), minlength=5).tolist(),
                max_probability_sum_error=float(abs(p.sum(1)-1).max()),
                gain_read_Pearson=float(np.corrcoef(read, gain)[0,1]) if np.ptp(read) > 1e-12 else None,
                gain_read_Spearman=correlation(read, gain),
                first_probability_PC_fraction=float(eigen[0]/eigen.sum()) if eigen.sum()>1e-20 else None)
            edges = [0, .2, .4, .55, .65, .8, 1.000001]
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask=(read >= lo) & (read < hi)
                if mask.any():
                    calibration.append(dict(partition=tag, arm=arm, bin_lower=lo,
                        bin_upper=min(hi,1), n=int(mask.sum()), predicted_read=float(read[mask].mean()),
                        actual_read=float((y[mask]>1).mean())))
            for name, mask in (("abstract_present",present),("abstract_missing",~present),
                               ("abstract_truncated",truncated),("abstract_not_truncated",~truncated)):
                if mask.any():
                    subgroup_rows.append(dict(partition=tag, arm=arm, subgroup=name, **metrics(y[mask], p[mask])))
        trained, inc = arms["trained"], arms["incumbent"]
        chosen, old = 1-trained[:,0]>.55, 1-inc[:,0]>.55
        for pos, index in enumerate(part["pred"]):
            if chosen[pos] == old[pos]:
                continue
            label = int(y[pos])
            kind = ("rescued_FN" if label>1 else "new_FP") if chosen[pos] else ("lost_TP" if label>1 else "removed_FP")
            changes.append(dict(partition=tag,index=index,id=docs[index]["id"],true_label=label,
                transition=kind,trained_read=float(1-trained[pos,0]),incumbent_read=float(1-inc[pos,0]),
                abstract_present=bool(present[pos]),title=docs[index]["title"]))
        logs = jsonl(source/tag/"training.jsonl")
        assert len(logs)==94 and all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in logs)
        elapsed=np.array([0]+[r["elapsed_seconds"] for r in logs])
        record["training_segments"]=[]
        for lo,hi in ((1,5),(6,20),(21,50),(51,94)):
            subset=logs[lo-1:hi]
            record["training_segments"].append(dict(first_update=lo,last_update=hi,
                mean_loss=float(np.mean([r["loss"] for r in subset])),
                mean_seconds_per_update=float(np.diff(elapsed)[lo-1:hi].mean())))
        report["partitions"][tag]=record
    write_json(out/"analysis.json",report)
    for name, rows in (("diagnostic_metrics",metric_rows),("read_reliability",calibration),
                       ("subgroups",subgroup_rows),("selection_changes",changes)):
        write_csv(out/f"{name}.csv",rows)
    print(json.dumps(dict(output=str(out),rows=len(metric_rows),source_unchanged=True),indent=2))


if __name__ == "__main__":
    main()
