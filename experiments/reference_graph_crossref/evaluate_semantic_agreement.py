#!/usr/bin/env python3
"""One locked D-arm test: Crossref overlap weighted by frozen SPECTER2 similarity.

The hypothesis follows the OOF audit: bibliographic coupling should be trusted
more when the two documents are also semantically close.  SPECTER2 retrieval
vectors are a frozen, pre-existing input; no document, preference label, split,
or threshold is changed.  Every training feature is group cross-fitted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs_semantic_agreement_v2"
BASE_OUT = HERE / "outputs"
sys.path.insert(0, str(HERE))
import evaluate as e


EXTRA_NAMES = [
    "agreement_N_max", "agreement_N_mean5", "agreement_P_max", "agreement_P_mean5",
    "agreement_G_max", "agreement_G_mean5", "agreement_PminusN_max",
    "agreement_PminusN_mean5", "agreement_GminusN_max", "agreement_GminusN_mean5",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(ctx: dict, graph: dict) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(
        script_sha256=digest(HERE / "evaluate_semantic_agreement.py"),
        crossref_ledger_sha256=digest(BASE_OUT / "doi_reference_ledger.jsonl"),
        split_sha256=e.ac.digest(e.ac.split_path()),
        retrieval_manifest_sha256=digest(ROOT / "experiments" / "diagnose_bottleneck" / "outputs" / "embeddings_retrieval.json"),
        base_feature_names=e.g.GRAPH_NAMES,
        added_feature_names=EXTRA_NAMES,
        agreement="Jaccard(outgoing DOI references) * frozen SPECTER2-retrieval cosine",
        model=dict(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8, loss="MultiClass"),
        selection="Each arm selects its threshold from only its four inner folds.",
        fallback="A query with no verified nonempty Crossref reference list copies the exact incumbent probability.",
    )
    target = OUT / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == value, "Protocol changed: use a new output directory."
    else:
        target.write_text(json.dumps(value, indent=2) + "\n")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def semantic_graph_features(ctx: dict, graph: dict, queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Classwise top graph scores after a continuous semantic-agreement weight."""
    queries, refs = np.asarray(queries, int), np.asarray(refs, int)
    result = np.zeros((len(queries), len(EXTRA_NAMES)), dtype=np.float32)
    vectors, labels, groups, ids = ctx["spaces"]["retrieval_margin"], ctx["labels"], ctx["groups"], ctx["ids"]
    for pos, query in enumerate(queries):
        qrefs = graph["refs"][query]
        if not qrefs:
            continue
        allowed = np.asarray([r for r in refs if graph["refs"][r] and groups[r] != groups[query]], int)
        if not len(allowed):
            continue
        overlap = np.asarray([len(qrefs & graph["refs"][r]) / len(qrefs | graph["refs"][r]) for r in allowed], float)
        # Both quantities are non-negative.  An unrelated graph neighbour is
        # therefore down-weighted continuously rather than removed by a tuned cut-off.
        agreement = overlap * np.asarray(vectors[allowed] @ vectors[query], float)
        y = labels[allowed]
        pairs = {}
        for code, condition, offset in [("N", y == 1, 0), ("P", y > 1, 2), ("G", y >= 4, 4)]:
            pairs[code] = e.g._stats(agreement[condition], ids[allowed][condition])
            result[pos, offset:offset + 2] = pairs[code]
        result[pos, 6:8] = np.subtract(pairs["P"], pairs["N"])
        result[pos, 8:10] = np.subtract(pairs["G"], pairs["N"])
    assert np.isfinite(result).all()
    return result


def feature_bundle(ctx: dict, graph: dict, name: str, fit: np.ndarray, pred: np.ndarray, sha: str) -> dict:
    target = OUT / f"{name}_agreement_features.joblib"
    if target.exists():
        saved = joblib.load(target)
        assert saved["protocol_sha256"] == sha
        np.testing.assert_array_equal(saved["fit"], fit); np.testing.assert_array_equal(saved["pred"], pred)
        return saved
    base = joblib.load(BASE_OUT / f"{name}_graph_features.joblib")
    np.testing.assert_array_equal(base["fit"], fit); np.testing.assert_array_equal(base["pred"], pred)
    pack, _, _ = e.ac.load_source_pack(name, fit, pred)
    location = {int(x): i for i, x in enumerate(fit)}
    train = np.full((len(fit), len(EXTRA_NAMES)), np.nan, dtype=np.float32)
    pred_views, seen = [], np.zeros(len(fit), bool)
    for item in pack["lineage"]:
        refs, queries = np.asarray(item["refs"], int), np.asarray(item["queries"], int)
        pos = np.asarray([location[int(x)] for x in queries], int)
        assert not seen[pos].any()
        train[pos] = semantic_graph_features(ctx, graph, queries, refs)
        pred_views.append(semantic_graph_features(ctx, graph, pred, refs))
        seen[pos] = True
    assert seen.all() and np.isfinite(train).all()
    saved = dict(protocol_sha256=sha, fit=fit, pred=pred, base_train=base["train"], base_pred_views=base["pred_views"],
        train=train, pred_views=np.asarray(pred_views), covered=base["graph_available"].astype(bool))
    joblib.dump(saved, target, compress=3)
    return saved


def probability(ctx: dict, graph: dict, name: str, fit: np.ndarray, pred: np.ndarray, sha: str):
    target = OUT / f"D_semantic_agreement_{name}_predictions.npz"
    if target.exists():
        with np.load(target) as z:
            assert str(z["protocol_sha256"].item()) == sha
            np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
            return z["probability"].copy(), z["incumbent"].copy(), z["covered"].astype(bool)
    bundle = feature_bundle(ctx, graph, name, fit, pred, sha)
    _, xfit, xpred = e.ac.load_source_pack(name, fit, pred)
    columns = e.g.GRAPH_NAMES + EXTRA_NAMES
    train = pd.concat([xfit.reset_index(drop=True), pd.DataFrame(np.column_stack([bundle["base_train"], bundle["train"]]), columns=columns)], axis=1)
    model = e.g.catboost(); model.fit(train, ctx["labels"][fit] - 1, cat_features=["venue"])
    views = []
    for old, extra in zip(bundle["base_pred_views"], bundle["pred_views"]):
        frame = pd.concat([xpred.reset_index(drop=True), pd.DataFrame(np.column_stack([old, extra]), columns=columns)], axis=1)
        views.append(model.predict_proba(frame))
    probability = np.mean(views, axis=0)
    incumbent = e.ndg.incumbent_cache(name, fit, pred)
    probability[~bundle["covered"]] = incumbent[~bundle["covered"]]
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
    np.savez_compressed(target, protocol_sha256=np.asarray(sha), fit=fit, pred=pred, probability=probability, incumbent=incumbent, covered=bundle["covered"])
    return probability, incumbent, bundle["covered"]


def base_c(name: str, fit: np.ndarray, pred: np.ndarray):
    with np.load(BASE_OUT / f"C_overlap_{name}_predictions.npz") as z:
        np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
        return z["probability"].copy(), z["incumbent"].copy(), z["covered"].astype(bool)


def main() -> None:
    ctx = e.ac.context(); graph = e.graph_context(ctx); sha = protocol(ctx, graph)
    rows, thresholds, details = [], [], []
    with threadpool_limits(4):
        for fold, fit, pred, numbers in e.ac.outer_jobs(ctx):
            inner = {key: np.full((len(fit), 5), np.nan) for key in ["A_incumbent", "C_overlap", "D_semantic_agreement"]}
            for part in sorted(np.unique(numbers)):
                mask = numbers == part; name = f"outer{fold}_inner{part}"
                d, inc, _ = probability(ctx, graph, name, fit[~mask], fit[mask], sha)
                c, inc_c, _ = base_c(name, fit[~mask], fit[mask])
                np.testing.assert_allclose(inc, inc_c, atol=1e-7)
                inner["A_incumbent"][mask] = inc
                inner["C_overlap"][mask] = c
                inner["D_semantic_agreement"][mask] = d
            selected = {arm: e.choose(prob, ctx["labels"][fit], ctx["ids"][fit], numbers) for arm, prob in inner.items()}
            d, inc, covered = probability(ctx, graph, f"outer{fold}_validation", fit, pred, sha)
            c, inc_c, covered_c = base_c(f"outer{fold}_validation", fit, pred)
            np.testing.assert_allclose(inc, inc_c, atol=1e-7); np.testing.assert_array_equal(covered, covered_c)
            for arm, prob in [("A_incumbent", inc), ("C_overlap", c), ("D_semantic_agreement", d)]:
                pri = e.g.priority(prob, selected[arm])
                metric = e.g.metrics(ctx["labels"][pred], pri, ctx["ids"][pred])
                metric.update(outer_fold=int(fold), arm=arm, threshold=float(selected[arm]),
                    AUC=float(roc_auc_score(ctx["labels"][pred] > 1, 1 - prob[:, 0])), graph_covered=int(covered.sum()))
                rows.append(metric); thresholds.append(dict(outer_fold=int(fold), arm=arm, threshold=float(selected[arm])))
                for pos, index in enumerate(pred):
                    details.append(dict(index=int(index), id=str(ctx["ids"][index]), Label=int(ctx["labels"][index]), outer_fold=int(fold),
                        graph_covered=bool(covered[pos]), arm=arm, probability_read=float(1-prob[pos,0]), priority=float(pri[pos])))
            print(f"outer={fold} coverage={covered.sum()}/{len(pred)} thresholds={selected}", flush=True)
    metrics = pd.DataFrame(rows); metrics.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "thresholds.csv", index=False); pd.DataFrame(details).to_csv(OUT / "oof_predictions.csv", index=False)
    summary = metrics.groupby("arm", sort=False).agg(H=("H", "mean"), Q=("Q", "mean"), R=("R", "mean"), AUC=("AUC", "mean"),
        selected=("selected", "sum"), TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"), missed_high=("missed_high", "sum"))
    summary.to_csv(OUT / "summary.csv")
    pivot = metrics.pivot(index="outer_fold", columns="arm", values="H")
    decision = dict(D_minus_C=float(summary.loc["D_semantic_agreement", "H"] - summary.loc["C_overlap", "H"]),
        D_minus_A=float(summary.loc["D_semantic_agreement", "H"] - summary.loc["A_incumbent", "H"]),
        D_vs_C_fold_wins=int((pivot.D_semantic_agreement > pivot.C_overlap).sum()),
        coverage_train=int(graph["available"][:2494].sum()), coverage_test=int(graph["available"][2494:].sum()))
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (OUT / "RESULTS.md").write_text("# Crossref graph × frozen SPECTER2 agreement\n\n" + summary.reset_index().to_markdown(index=False, floatfmt=".6f") + "\n\n```json\n" + json.dumps(decision, indent=2) + "\n```\n")
    print(summary.to_string()); print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
