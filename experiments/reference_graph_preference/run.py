#!/usr/bin/env python3
"""Resumable public-reference collection and leakage-safe CatBoost ablation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Set REFERENCE_GRAPH_OUTPUT to create a new, isolated model run while reusing
# the append-only acquisition cache below.  Never overwrite a completed result.
OUT = HERE / os.environ.get("REFERENCE_GRAPH_OUTPUT", "outputs")
CACHE = HERE / "outputs" / "openalex_raw.jsonl"
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "reference_graph_audit"),
                str(ROOT / "experiments" / "abstract_preference_margin"),
                str(ROOT / "experiments" / "nested_disagreement_gate")]
from collect_remaining_abstracts import load_sources
_audit_spec = importlib.util.spec_from_file_location("reference_graph_audit_run", ROOT / "experiments" / "reference_graph_audit" / "run.py")
assert _audit_spec and _audit_spec.loader
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)
import apm_context as ac
import ndg_lib as ndg

SEED = 20261018
THRESHOLDS = np.asarray([.45, .50, .55, .60, .65], float)
GAIN = np.arange(5, dtype=float) / 4
FIELDS = audit.FIELDS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_records() -> pd.DataFrame:
    sources = load_sources()
    frame = pd.DataFrame([dict(split=split, **row) for (split, _), row in sources.items()])
    frame["id"] = frame.id.astype(str)
    frame["year"] = pd.to_numeric(frame.year, errors="raise").astype(int)
    frame["doi_canonical"] = frame.doi.map(audit.canonical_doi)
    return frame.sort_values(["split", "id"], kind="stable").reset_index(drop=True)


def cache_entries() -> dict[tuple[str, str], dict]:
    result = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf8").splitlines():
            item = json.loads(line)
            result[(item["split"], str(item["id"]))] = item
    return result


def request(filter_value: str, kind: str) -> dict:
    item = dict(retrieved_at=datetime.now(timezone.utc).isoformat(), kind=kind, filter=filter_value)
    for attempt in range(2):
        try:
            response = requests.get("https://api.openalex.org/works", params={"filter": filter_value, "per_page": 100, "select": FIELDS},
                headers={"User-Agent": "Personalized-reading-reference-ablation/1.0"}, timeout=30)
            item["http_status"] = response.status_code
            item["rate_headers"] = {k: v for k, v in response.headers.items() if "ratelimit" in k.lower()}
            if response.ok:
                item["response"] = response.json()
                return item
            item["error"] = response.text[:300]
            if response.status_code in (401, 403, 429):
                return item
        except requests.RequestException as exc:
            item.update(http_status=0, error=f"{type(exc).__name__}: {str(exc)[:180]}")
        if attempt == 0:
            time.sleep(1.5)
    return item


def write_batch(sink, rows: list[dict], response: dict, query: str) -> None:
    for row in rows:
        item = dict(response, split=row["split"], id=str(row["id"]), requested_title=row["title"],
                    requested_doi=row["doi_canonical"], query=query)
        sink.write(json.dumps(item, ensure_ascii=False) + "\n")
    sink.flush()


def collect(max_title_batches: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = all_records().to_dict("records")
    cached = cache_entries()
    direct = [r for r in records if r["doi_canonical"] and (r["split"], r["id"]) not in cached]
    print(f"records={len(records)} cached={len(cached)} pending_doi={len(direct)}", flush=True)
    with CACHE.open("a", encoding="utf8") as sink:
        for start in range(0, len(direct), 80):
            batch = direct[start:start + 80]
            query = "doi:" + "|".join("https://doi.org/" + r["doi_canonical"] for r in batch)
            response = request(query, "doi_batch")
            write_batch(sink, batch, response, query)
            print(f"doi {min(start + len(batch), len(direct))}/{len(direct)} status={response['http_status']} {response.get('rate_headers', {})}", flush=True)
            if response["http_status"] in (0, 401, 403, 429):
                return
            time.sleep(1.1)
        cached = cache_entries()
        remaining = [r for r in records if (r["split"], r["id"]) not in cached]
        print(f"pending_title={len(remaining)} max_title_batches={max_title_batches}", flush=True)
        for batch_no, start in enumerate(range(0, len(remaining), 20), 1):
            if batch_no > max_title_batches:
                return
            batch = remaining[start:start + 20]
            # Preserve the OR delimiter; normalising this whole query would turn OR into AND.
            query = "title.search:" + "|".join(audit.norm(r["title"]) for r in batch)
            response = request(query, "title_batch")
            write_batch(sink, batch, response, query)
            print(f"title {batch_no}/{max_title_batches} rows={len(batch)} status={response['http_status']} {response.get('rate_headers', {})}", flush=True)
            if response["http_status"] in (0, 401, 403, 429):
                return
            time.sleep(1.1)


def verified_authors() -> dict[tuple[str, str], str]:
    path = ROOT / "experiments" / "input_ablation" / "outputs" / "metadata_audit.csv"
    data = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    return {(r.split, str(r.id)): r.authors_verified for r in data.itertuples()
            if str(r.author_verified).lower() == "true" and r.authors_verified}


def materialize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = all_records(); cached = cache_entries(); known_authors = verified_authors()
    # Reuse the immutable raw responses from the earlier audited pilot.  They are
    # still subjected to exactly the same identity rule below; this only avoids
    # spending a second API request for a paper already queried in this session.
    pilot_cache = ROOT / "experiments" / "reference_graph_audit" / "outputs" / "openalex_raw.jsonl"
    reused_pilot = 0
    if pilot_cache.exists():
        pilot_latest = {}
        for line in pilot_cache.read_text(encoding="utf8").splitlines():
            item = json.loads(line)
            pilot_latest[(item["split"], str(item["id"]))] = item
        for key, item in pilot_latest.items():
            if key not in cached:
                cached[key] = item
                reused_pilot += 1
    rows, works = [], {}
    for raw in records.to_dict("records"):
        key = (raw["split"], raw["id"])
        row = dict(raw, authors=known_authors.get(key, raw["authors"]))
        evidence, work = audit.identify(row, cached.get(key))
        verified = work is not None
        refs = (work or {}).get("referenced_works") or []
        if verified:
            works[f"{key[0]}:{key[1]}"] = dict(openalex_id=work["id"], references=refs)
        rows.append(dict(split=key[0], id=key[1], title=raw["title"], venue=raw["venue"], year=int(raw["year"]),
            identity_status=evidence["identity_status"], verified=verified,
            references_available=bool(refs), references_count=len(refs) if verified else np.nan,
            work_id=(work or {}).get("id", ""), identity_method=evidence.get("identity_method", ""),
            title_similarity=evidence.get("title_similarity", np.nan), token_jaccard=evidence.get("token_jaccard", np.nan),
            year_gap=evidence.get("year_gap", np.nan), author_check_available=evidence.get("author_check_available", False),
            author_overlap=evidence.get("author_overlap", False)))
    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "identity_ledger.csv", index=False)
    (OUT / "verified_works.json").write_text(json.dumps(works, ensure_ascii=False), encoding="utf8")
    summary = ledger.groupby("split").agg(n=("id", "size"), verified=("verified", "sum"),
        with_references=("references_available", "sum"), median_references=("references_count", "median"))
    summary["verified_fraction"] = summary.verified / summary.n
    summary["reference_fraction"] = summary.with_references / summary.n
    summary.to_csv(OUT / "coverage_summary.csv")
    status = dict(total=len(ledger), cached=len(cached), reused_pilot_responses=reused_pilot, verified=int(ledger.verified.sum()),
        with_references=int(ledger.references_available.sum()), identity_status_counts=ledger.identity_status.value_counts().to_dict(),
        source="OpenAlex outgoing references; failed requests remain REQUEST_FAILED", models_trained=0)
    (OUT / "collection_summary.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    print(summary.to_string(), flush=True)


GRAPH_NAMES = ["graph_available", "graph_log_reference_count", "graph_overlap_support", "graph_N_max", "graph_N_mean5",
    "graph_P_max", "graph_P_mean5", "graph_G_max", "graph_G_mean5", "graph_PminusN_max", "graph_PminusN_mean5",
    "graph_GminusN_max", "graph_GminusN_mean5"]


def graph_context(ctx: dict) -> dict:
    ledger = pd.read_csv(OUT / "identity_ledger.csv", dtype={"id": str})
    works = json.loads((OUT / "verified_works.json").read_text(encoding="utf8"))
    keys = ctx["keys"].astype(str)
    assert set(keys) == set(ledger.split.astype(str) + ":" + ledger.id.astype(str))
    refs = []
    for key in keys:
        item = works.get(str(key), {})
        refs.append(frozenset(item.get("references", [])))
    available = np.asarray([bool(x) for x in refs], bool)
    return dict(refs=refs, available=available, source_sha256=digest(OUT / "verified_works.json"))


def _stats(values: np.ndarray, ids: np.ndarray) -> tuple[float, float]:
    if not len(values):
        return 0.0, 0.0
    order = np.lexsort((ids.astype(str), -values))
    ranked = values[order]
    return float(ranked[0]), float(ranked[:min(5, len(ranked))].mean())


def graph_features(ctx: dict, graph: dict, queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
    queries, refs = np.asarray(queries, int), np.asarray(refs, int)
    values = np.zeros((len(queries), len(GRAPH_NAMES)), np.float32)
    labels, groups, ids = ctx["labels"], ctx["groups"], ctx["ids"]
    for pos, query in enumerate(queries):
        qrefs = graph["refs"][query]
        if not qrefs:
            continue
        allowed = np.asarray([r for r in refs if graph["refs"][r] and groups[r] != groups[query]], int)
        if not len(allowed):
            values[pos, 1] = np.log1p(len(qrefs)); continue
        overlaps = np.asarray([len(qrefs & graph["refs"][r]) / len(qrefs | graph["refs"][r]) for r in allowed], float)
        y = labels[allowed]; overlap = overlaps > 0
        values[pos, 0] = 1.0
        values[pos, 1] = np.log1p(len(qrefs))
        values[pos, 2] = float(overlap.mean())
        pairs = {}
        for code, condition, offset in [("N", y == 1, 3), ("P", y > 1, 5), ("G", y >= 4, 7)]:
            maximum, mean5 = _stats(overlaps[condition], ids[allowed][condition])
            values[pos, offset:offset + 2] = maximum, mean5
            pairs[code] = maximum, mean5
        values[pos, 9:11] = np.subtract(pairs["P"], pairs["N"])
        values[pos, 11:13] = np.subtract(pairs["G"], pairs["N"])
    assert np.isfinite(values).all()
    return values


def source_pack(name: str, fit: np.ndarray, pred: np.ndarray):
    return ac.load_source_pack(name, fit, pred)


def partition_features(ctx: dict, graph: dict, name: str, fit: np.ndarray, pred: np.ndarray, pack: dict, protocol_sha: str) -> dict:
    path = OUT / f"{name}_graph_features.joblib"
    if path.exists():
        saved = joblib.load(path)
        assert saved["protocol_sha256"] == protocol_sha
        np.testing.assert_array_equal(saved["fit"], fit); np.testing.assert_array_equal(saved["pred"], pred)
        return saved
    location = {int(x): i for i, x in enumerate(fit)}
    train = np.full((len(fit), len(GRAPH_NAMES)), np.nan, np.float32)
    pred_views = []; seen = np.zeros(len(fit), bool); views = []
    for entry in pack["lineage"]:
        refs, queries = np.asarray(entry["refs"], int), np.asarray(entry["queries"], int)
        positions = np.asarray([location[int(x)] for x in queries], int)
        assert set(refs).issubset(set(fit)) and set(queries).issubset(set(fit))
        assert not seen[positions].any() and not (set(ctx["groups"][refs]) & set(ctx["groups"][queries]))
        train[positions] = graph_features(ctx, graph, queries, refs)
        pred_views.append(graph_features(ctx, graph, pred, refs))
        seen[positions] = True
        views.append(dict(fold=int(entry["fold"]), refs=len(refs), queries=len(queries)))
    assert seen.all() and np.isfinite(train).all()
    result = dict(protocol_sha256=protocol_sha, fit=fit, pred=pred, train=train,
        pred_views=np.asarray(pred_views), graph_available=graph["available"][pred], views=views)
    joblib.dump(result, path, compress=3)
    return result


def catboost() -> CatBoostClassifier:
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8, loss_function="MultiClass",
        classes_count=5, random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False)


def priority(probability: np.ndarray, threshold: float) -> np.ndarray:
    return np.where((1 - probability[:, 0]) > threshold, 1 + 9 * (probability @ GAIN), 0.0)


def metrics(labels: np.ndarray, priorities: np.ndarray, ids: np.ndarray) -> dict:
    labels, priorities, ids = np.asarray(labels, int), np.asarray(priorities, float), np.asarray(ids, str)
    selected, pos = priorities > 0, labels > 1
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-priorities[i], ids[i])), int)
    weights = 1 / np.log2(np.arange(len(order), dtype=float) + 2)
    q = float(np.average((labels[order] - 1) / 4, weights=weights)) if len(order) else 0.0
    tp, fp, fn = int((selected & pos).sum()), int((selected & ~pos).sum()), int((~selected & pos).sum())
    r = tp / int(pos.sum()) if pos.any() else 0.0
    return dict(H=2*q*r/(q+r) if q+r else 0., Q=q, R=r, selected=int(selected.sum()), TP=tp, FP=fp, FN=fn,
        missed_label4=int(((labels == 4) & ~selected).sum()), missed_label5=int(((labels == 5) & ~selected).sum()),
        missed_high=int(((labels >= 4) & ~selected).sum()))


def choose_threshold(inner: np.ndarray, labels: np.ndarray, ids: np.ndarray, numbers: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for threshold in THRESHOLDS:
        fold = [metrics(labels[numbers == k], priority(inner[numbers == k], threshold), ids[numbers == k]) for k in sorted(np.unique(numbers))]
        rows.append(dict(threshold=float(threshold), inner_H=float(np.mean([x["H"] for x in fold])), inner_Q=float(np.mean([x["Q"] for x in fold])), inner_R=float(np.mean([x["R"] for x in fold]))))
    chosen = sorted(rows, key=lambda r: (r["inner_H"], r["threshold"]), reverse=True)[0]
    return chosen["threshold"], rows


def probability(ctx: dict, graph: dict, name: str, fit: np.ndarray, pred: np.ndarray, protocol_sha: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = OUT / f"{name}_predictions.npz"
    if path.exists():
        with np.load(path) as z:
            assert str(z["protocol_sha256"].item()) == protocol_sha
            np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
            return z["probability"].copy(), z["incumbent"].copy(), {"covered": z["covered"].astype(bool)}
    pack, xfit, xpred = source_pack(name, fit, pred)
    features = partition_features(ctx, graph, name, fit, pred, pack, protocol_sha)
    incumbent = ndg.incumbent_cache(name, fit, pred)
    model = catboost()
    graph_columns = pd.DataFrame(features["train"], columns=GRAPH_NAMES)
    train = pd.concat([xfit.reset_index(drop=True), graph_columns], axis=1)
    model.fit(train, ctx["labels"][fit] - 1, cat_features=["venue"])
    views = []
    for view in features["pred_views"]:
        frame = pd.concat([xpred.reset_index(drop=True), pd.DataFrame(view, columns=GRAPH_NAMES)], axis=1)
        views.append(model.predict_proba(frame))
    result = np.mean(views, axis=0)
    covered = features["graph_available"].astype(bool)
    result[~covered] = incumbent[~covered]
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    np.savez_compressed(path, protocol_sha256=np.asarray(protocol_sha), fit=fit, pred=pred, probability=result, incumbent=incumbent, covered=covered)
    return result, incumbent, {"covered": covered}


def protocol(ctx: dict, graph: dict) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(plan_sha256=digest(HERE / "PLAN.md"), run_sha256=digest(HERE / "run.py"), data_sha256=ac.digest(ac.source.INC / "test_prediction_details.csv"),
        split_sha256=digest(ac.split_path()), graph_sha256=graph["source_sha256"], graph_features=GRAPH_NAMES, seed=SEED,
        catboost=dict(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8), thresholds=THRESHOLDS.tolist(),
        fallback="unverified or empty-query references use exact incumbent probability")
    path = OUT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old == value, "Protocol/source/graph changed; create a new experiment directory."
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_outer(ctx: dict, graph: dict, protocol_sha: str) -> None:
    rows, search, oof = [], [], []
    for fold, fit, pred, nums in ac.outer_jobs(ctx):
        inner = np.full((len(fit), 5), np.nan)
        for part in sorted(np.unique(nums)):
            mask = nums == part; name = f"outer{fold}_inner{part}"
            p, _, _ = probability(ctx, graph, name, fit[~mask], fit[mask], protocol_sha)
            inner[mask] = p
        assert np.isfinite(inner).all()
        threshold, candidates = choose_threshold(inner, ctx["labels"][fit], ctx["ids"][fit], nums)
        p, base, coverage = probability(ctx, graph, f"outer{fold}_validation", fit, pred, protocol_sha)
        for family, prob in [("incumbent", base), ("graph", p)]:
            score = metrics(ctx["labels"][pred], priority(prob, threshold), ctx["ids"][pred])
            score.update(outer_fold=int(fold), family=family, threshold=threshold,
                AUC=float(roc_auc_score(ctx["labels"][pred] > 1, 1 - prob[:, 0])), graph_covered=int(coverage["covered"].sum()))
            rows.append(score)
        search.extend(dict(outer_fold=int(fold), **r) for r in candidates)
        for pos, index in enumerate(pred):
            oof.append(dict(index=int(index), id=str(ctx["ids"][index]), outer_fold=int(fold), Label=int(ctx["labels"][index]),
                graph_covered=bool(coverage["covered"][pos]), incumbent_read=float(1-base[pos, 0]), graph_read=float(1-p[pos, 0]),
                incumbent_priority=float(priority(base, threshold)[pos]), graph_priority=float(priority(p, threshold)[pos])))
        pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv", index=False)
        pd.DataFrame(search).to_csv(OUT / "inner_search.csv", index=False)
        pd.DataFrame(oof).sort_values("index").to_csv(OUT / "oof_predictions.csv", index=False)
        print(f"outer {fold}: {rows[-2]['H']:.5f} -> {rows[-1]['H']:.5f}; covered={rows[-1]['graph_covered']}/{len(pred)}", flush=True)
    report_outer()


def report_outer() -> None:
    metrics_frame = pd.read_csv(OUT / "fold_metrics.csv")
    summary = metrics_frame.groupby("family", sort=False).agg(H=("H", "mean"), Q=("Q", "mean"), R=("R", "mean"), AUC=("AUC", "mean"),
        selected=("selected", "sum"), TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"), missed_high=("missed_high", "sum"))
    summary.to_csv(OUT / "summary.csv")
    oof = pd.read_csv(OUT / "oof_predictions.csv", dtype={"id": str})
    base, graph = oof.incumbent_priority.gt(0), oof.graph_priority.gt(0)
    positive = oof.Label.gt(1)
    transitions = dict(FN_rescued=int((~base & graph & positive).sum()), FP_removed=int((base & ~graph & ~positive).sum()),
        new_FP=int((~base & graph & ~positive).sum()), lost_TP=int((base & ~graph & positive).sum()))
    transitions["delta_error"] = transitions["new_FP"] + transitions["lost_TP"] - transitions["FN_rescued"] - transitions["FP_removed"]
    pd.DataFrame([transitions]).to_csv(OUT / "transitions.csv", index=False)
    covered = oof[oof.graph_covered]
    coverage = dict(total=len(oof), graph_covered=len(covered), fraction=float(len(covered)/len(oof)),
        covered_label_counts=covered.Label.value_counts().sort_index().to_dict())
    (OUT / "coverage_on_oof.json").write_text(json.dumps(coverage, indent=2) + "\n")
    gain = float(summary.loc["graph", "H"] - summary.loc["incumbent", "H"])
    byfold = metrics_frame.pivot(index="outer_fold", columns="family", values="H")
    wins = int((byfold.graph > byfold.incumbent).sum())
    promoted = gain >= .005 and wins >= 3 and summary.loc["graph", "missed_high"] <= summary.loc["incumbent", "missed_high"]
    decision = dict(mean_H_delta=gain, fold_wins=wins, promoted=bool(promoted), rule="delta H >= .005; >=3/5 wins; missed Label4-5 not larger")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    text = "# Bibliographic-coupling ablation\n\n" + summary.reset_index().to_markdown(index=False, floatfmt=".6f") + "\n\n" + json.dumps(dict(decision=decision, transitions=transitions, coverage=coverage), indent=2) + "\n"
    (OUT / "RESULTS.md").write_text(text, encoding="utf8")
    print(summary.to_string(), "\n", json.dumps(decision, indent=2), flush=True)


def verify() -> None:
    ledger = pd.read_csv(OUT / "identity_ledger.csv", dtype={"id": str})
    records = all_records()
    assert len(ledger) == len(records) == 3090
    assert ledger[["split", "id"]].duplicated().sum() == 0
    works = json.loads((OUT / "verified_works.json").read_text())
    assert all(isinstance(v.get("references"), list) for v in works.values())
    if (OUT / "summary.csv").exists():
        oof = pd.read_csv(OUT / "oof_predictions.csv", dtype={"id": str})
        assert len(oof) == 2494 and oof.id.nunique() == 2494
        assert np.isfinite(oof[["incumbent_read", "graph_read"]].to_numpy()).all()
    result = dict(status="PASSED", records=len(ledger), verified=int(ledger.verified.sum()), references=int(ledger.references_available.sum()), works=len(works))
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["collect", "materialize", "outer", "verify"])
    parser.add_argument("--max-title-batches", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "collect": collect(args.max_title_batches); return
    if args.stage == "materialize": materialize(); return
    if args.stage == "verify": verify(); return
    ctx = ac.context(); graph = graph_context(ctx); sha = protocol(ctx, graph)
    with threadpool_limits(4): run_outer(ctx, graph, sha)


if __name__ == "__main__":
    main()
