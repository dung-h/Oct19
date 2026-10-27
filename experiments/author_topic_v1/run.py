"""Label-free B0 author identity and author×topic support feasibility audit."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = ROOT / "experiments/qwen_remote_v1"
INPUTS = REMOTE / "inputs"
PARTITIONS = ("outer1_inner0", "outer1_inner1")
SAMPLE_SEED = 20261102
INDEX_RE = re.compile(r'"index"\s*:\s*(\d+)')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def canonical_name(value):
    normalized = unicodedata.normalize("NFC", str(value))
    return " ".join(normalized.split()).casefold()


def author_name_is_uncertain(name):
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", name), flags=re.UNICODE)
    return len(tokens) < 2 or sum(len(token) > 1 for token in tokens) < 2


def parse_authors(raw):
    if not str(raw or "").strip():
        return {"status": "missing", "authors": [], "uncertain": []}
    authors, uncertain, seen = [], [], set()
    for item in re.split(r"\s*,\s*", str(raw).strip()):
        display = " ".join(unicodedata.normalize("NFC", item).split())
        if not display:
            continue
        key = canonical_name(display)
        if key in seen:
            continue
        seen.add(key)
        row = {"key": key, "display": display}
        (uncertain if author_name_is_uncertain(display) else authors).append(row)
    status = "known" if authors else "uncertain" if uncertain else "missing"
    return {"status": status, "authors": authors, "uncertain": uncertain}


def official_metadata_query(doc, retries=4):
    params = urlencode({"query.title": doc["title"], "rows": 3,
                        "select": "DOI,title,author,publisher,URL"})
    url = "https://api.crossref.org/works?" + params
    request = Request(url, headers={"User-Agent": "Oct19-author-topic-feasibility/1.0"})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=18) as response:
                payload = response.read()
                body = json.loads(payload)
                rate_headers = {name: response.headers.get(name) for name in
                    ("x-rate-limit-limit", "x-rate-limit-interval", "x-concurrency-limit", "x-api-pool")}
            items = body.get("message", {}).get("items", [])
            raw_names = parse_authors(doc.get("authors", ""))["authors"]
            candidates = []
            for item in items:
                title = item.get("title", [""])[0] if item.get("title") else ""
                ratio = difflib.SequenceMatcher(None, canonical_name(title), canonical_name(doc["title"])).ratio()
                official_names = [" ".join(filter(None, (a.get("given", ""), a.get("family", "")))).strip()
                                  for a in item.get("author", [])]
                norm_official = {canonical_name(name) for name in official_names if name}
                matches = [row["display"] for row in raw_names if row["key"] in norm_official]
                candidates.append({"doi": item.get("DOI"), "title": title, "title_similarity": ratio,
                    "publisher": item.get("publisher"), "url": item.get("URL"),
                    "official_authors": official_names, "matched_raw_names": matches,
                    "raw_name_count": len(raw_names)})
            candidates.sort(key=lambda row: (-row["title_similarity"], -len(row["matched_raw_names"])))
            best = candidates[0] if candidates else None
            title_match = bool(best and best["title_similarity"] >= 0.90)
            all_names_match = bool(best and len(raw_names) > 0 and len(best["matched_raw_names"]) == len(raw_names))
            return {"status": "queried", "request_url": url, "http_status": 200,
                "response_rate_limit_headers": rate_headers,
                "crossref_response_sha256": hashlib.sha256(payload).hexdigest(),
                "crossref_response": body, "candidates": candidates,
                "high_title_match": title_match, "all_raw_full_names_matched": all_names_match,
                "row_supported": bool(title_match and all_names_match)}
        except HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = (float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else min(2 ** (attempt + 1), 16))
                time.sleep(wait)
                continue
            return {"status": "unavailable", "request_url": url, "http_status": exc.code,
                "response_rate_limit_headers": {name: exc.headers.get(name) if exc.headers else None
                    for name in ("x-rate-limit-limit", "x-rate-limit-interval", "x-concurrency-limit", "x-api-pool")},
                "retry_after": exc.headers.get("Retry-After") if exc.headers else None,
                "error": f"{type(exc).__name__}: {exc}", "high_title_match": False,
                "all_raw_full_names_matched": False, "row_supported": False}
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return {"status": "unavailable", "request_url": url,
                    "error": f"{type(exc).__name__}: {exc}", "high_title_match": False,
                    "all_raw_full_names_matched": False, "row_supported": False}


def load_docs_parts():
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(INPUTS / name) != expected:
            raise RuntimeError(f"Input hash mismatch: {name}")
    docs = {int(row["index"]): row for row in
            (json.loads(line) for line in (INPUTS / "documents.jsonl").read_text().splitlines() if line)}
    parts = json.loads((INPUTS / "partitions.json").read_text())
    if len(docs) != 2494:
        raise RuntimeError("Unexpected document count")
    for name in PARTITIONS:
        fit, pred = parts[name]["fit"], parts[name]["pred"]
        if {docs[i]["group"] for i in fit} & {docs[i]["group"] for i in pred}:
            raise RuntimeError(f"Group leakage in {name}")
    return docs, parts, manifest


def capped_text(tokenizer, doc, cfg):
    chunks = []
    for field, limit in (("title", 80), ("abstract", 560)):
        token_ids = tokenizer.encode(str(doc.get(field, "")), add_special_tokens=False)[:limit]
        chunks.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
    return "\n".join(chunks)


def fit_topics(name, fit_ids, pred_ids, docs, parsed, tokenizer, cfg, output):
    texts = [capped_text(tokenizer, docs[i], cfg) for i in fit_ids]
    vectorizer = TfidfVectorizer(ngram_range=tuple(cfg["topic"]["ngram_range"]),
        min_df=cfg["topic"]["min_df"], max_features=cfg["topic"]["max_features"],
        dtype=np.float32, norm="l2")
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] < cfg["topic"]["svd_components"] + 1:
        raise RuntimeError(f"Not enough TF-IDF features for SVD in {name}: {matrix.shape[1]}")
    svd = TruncatedSVD(n_components=cfg["topic"]["svd_components"],
                       n_iter=cfg["topic"]["svd_n_iter"], random_state=cfg["seed"])
    reduced_fit = svd.fit_transform(matrix)
    kmeans = KMeans(n_clusters=cfg["topic"]["clusters"], random_state=cfg["seed"],
                    n_init=cfg["topic"]["kmeans_n_init"])
    topic_fit = kmeans.fit_predict(reduced_fit)
    pred_texts = [capped_text(tokenizer, docs[i], cfg) for i in pred_ids]
    pred_matrix = vectorizer.transform(pred_texts)
    topic_pred = kmeans.predict(svd.transform(pred_matrix))
    topic_by_index = {int(i): int(t) for i, t in zip(fit_ids + pred_ids, list(topic_fit) + list(topic_pred))}
    history = {}
    author_groups = {}
    for idx in fit_ids:
        group = docs[idx]["group"]
        for author in parsed[idx]["authors"]:
            author_groups.setdefault(author["key"], set()).add(group)
            history.setdefault((author["key"], topic_by_index[idx]), set()).add(group)
    query_rows = []
    for idx in pred_ids:
        doc = docs[idx]
        topic = topic_by_index[idx]
        known = parsed[idx]["authors"]
        if parsed[idx]["status"] != "known":
            support_class = "missing_or_uncertain"
            max_author_support = 0
        else:
            author_supports = []
            topic_supports = []
            for author in known:
                groups_for_author = author_groups.get(author["key"], set()) - {doc["group"]}
                groups_for_topic = history.get((author["key"], topic), set()) - {doc["group"]}
                author_supports.append(len(groups_for_author))
                topic_supports.append(len(groups_for_topic))
            max_author_support = max(author_supports, default=0)
            if max_author_support == 0:
                support_class = "cold"
            elif max_author_support == 1:
                support_class = "support1"
            elif max_author_support <= 4:
                support_class = "support2_4"
            else:
                support_class = "support5_plus"
        qualifying = [a["display"] for a in known
                      if len(history.get((a["key"], topic), set()) - {doc["group"]}) >= cfg["author_topic_min_support_groups"]]
        query_rows.append({"index": idx, "topic": topic, "author_parse_status": parsed[idx]["status"],
            "parsed_authors": [a["display"] for a in known],
            "uncertain_authors": [a["display"] for a in parsed[idx]["uncertain"]],
            "max_author_support_groups": max_author_support,
            "author_support_class": support_class,
            "author_topic_support_by_author": {a["display"]: len(history.get((a["key"], topic), set()) - {doc["group"]})
                                              for a in known},
            "qualifying_author_topic_names": qualifying,
            "qualifying_author_topic_count": len(qualifying),
            "query_group_excluded": doc["group"] not in set().union(*[
                history.get((a["key"], topic), set()) for a in known]) if known else True})
    model_path = output / f"topic_model_{name}.joblib"
    joblib.dump({"vectorizer": vectorizer, "svd": svd, "kmeans": kmeans,
                 "fit_indices": list(map(int, fit_ids)), "fit_topic": list(map(int, topic_fit)),
                 "seed": cfg["seed"], "fit_only": True}, model_path)
    with (output / f"topic_lineage_{name}.jsonl").open("w") as stream:
        for row in query_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(output / f"topics_{name}.npz", indices=np.asarray(fit_ids + pred_ids),
                        topic=np.asarray(list(topic_fit) + list(topic_pred), dtype=np.int8))
    counts = {key: sum(row["author_support_class"] == key for row in query_rows)
              for key in ("support1", "support2_4", "support5_plus", "cold", "missing_or_uncertain")}
    supported_topic = sum(row["qualifying_author_topic_count"] > 0 for row in query_rows)
    return {"partition": name, "fit_n": len(fit_ids), "prediction_n": len(pred_ids),
        "tfidf_shape": list(matrix.shape), "svd_components": int(reduced_fit.shape[1]),
        "topic_counts_fit": np.bincount(topic_fit, minlength=cfg["topic"]["clusters"]).tolist(),
        "prediction_author_support_counts": counts,
        "known_author_fraction": float(sum(counts[k] for k in ("support1", "support2_4", "support5_plus")) / len(pred_ids)),
        "has_parseable_author_name_fraction": float((len(pred_ids) - counts["missing_or_uncertain"]) / len(pred_ids)),
        "prediction_with_author_topic_support_ge2_groups": supported_topic,
        "author_topic_coverage": float(supported_topic / len(pred_ids)),
        "gate_minimum": cfg["author_topic_coverage_gate"],
        "group_exclusion_checked": all(row["query_group_excluded"] for row in query_rows),
        "model_sha256": sha(model_path), "lineage_sha256": sha(output / f"topic_lineage_{name}.jsonl"),
        "topics_sha256": sha(output / f"topics_{name}.npz")}


def run(output):
    if output.exists():
        raise RuntimeError("B0 output namespace exists; preserve it and choose a fresh one")
    cfg = json.loads((HERE / "config.json").read_text())
    if cfg["seed"] != SAMPLE_SEED:
        raise RuntimeError("B0 sample/topic seed changed from locked plan")
    docs, parts, manifest = load_docs_parts()
    output.mkdir(parents=True)
    source = {name: sha(HERE / name) for name in ("config.json", "PROTOCOL.md", "README.md", "run.py")}
    write_json(output / "source_lock.json", {"source": source,
        "input_manifest_sha256": sha(INPUTS / "manifest.json"),
        "target_file_hashed_not_parsed": manifest["files"]["targets.jsonl"],
        "target_values_parsed": False})
    parsed = {idx: parse_authors(doc.get("authors", "")) for idx, doc in docs.items()}
    with (output / "author_parse_audit.jsonl").open("x") as stream:
        for idx, doc in sorted(docs.items()):
            stream.write(json.dumps({"index": idx, "id": doc["id"], "raw_authors": doc.get("authors", ""),
                "author_provenance": doc.get("author_provenance"), **parsed[idx]}, ensure_ascii=False) + "\n")
    rng = np.random.default_rng(cfg["seed"])
    sample_ids = sorted(map(int, rng.choice(np.asarray(sorted(docs)), size=cfg["sample_size"], replace=False)))
    sampled = [docs[idx] for idx in sample_ids]
    evidence_dir = output / "metadata_evidence"
    evidence_dir.mkdir()
    metadata_results = {}
    for position, doc in enumerate(sampled):
        idx = int(doc["index"])
        result = official_metadata_query(doc, cfg["metadata_429_retries"])
        result["request_spacing_seconds"] = cfg["metadata_request_interval_seconds"]
        metadata_results[idx] = result
        write_json(evidence_dir / f"{idx}.json", {"sample_index": idx,
            "raw_title": docs[idx]["title"], "raw_authors": docs[idx].get("authors", ""),
            "raw_doi_field": docs[idx].get("doi"), **result})
        if position + 1 < len(sampled):
            time.sleep(cfg["metadata_request_interval_seconds"])
    sampled_rows = []
    for idx in sample_ids:
        result = metadata_results[idx]
        sampled_rows.append({"index": idx, "id": docs[idx]["id"],
            "title": docs[idx]["title"], "raw_authors": docs[idx].get("authors", ""),
            "author_parse_status": parsed[idx]["status"],
            "nonmissing_authors": bool(parsed[idx]["authors"]),
            "crossref_status": result["status"], "high_title_match": result["high_title_match"],
            "all_raw_full_names_matched": result["all_raw_full_names_matched"],
            "row_supported": result["row_supported"],
            "best_candidate": result.get("candidates", [None])[0]})
    eligible = [row for row in sampled_rows if row["nonmissing_authors"]]
    supported = sum(row["row_supported"] for row in eligible)
    metadata_rate = supported / len(eligible) if eligible else 0.0
    write_json(output / "metadata_sample.json", {"seed": cfg["seed"], "sample_indices": sample_ids,
        "sample_rows": sampled_rows, "eligible_nonmissing_rows": len(eligible),
        "fully_supported_rows": supported, "support_rate": metadata_rate,
        "threshold": cfg["metadata_gate_min_supported_rows"],
        "sample_is_not_whole_bundle_verification": True})
    from transformers import AutoTokenizer
    model_cfg = json.loads((REMOTE / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model"]["id"],
        revision=model_cfg["model"]["revision"], local_files_only=True)
    answer_ids = [tokenizer.encode(s, add_special_tokens=False) for s in model_cfg["input"]["answer_strings"]]
    if answer_ids != [[16], [17], [18], [19], [20]]:
        raise RuntimeError(f"Pinned tokenizer IDs differ: {answer_ids}")
    topic_rows = {}
    for name in PARTITIONS:
        topic_rows[name] = fit_topics(name, list(map(int, parts[name]["fit"])),
            list(map(int, parts[name]["pred"])), docs, parsed, tokenizer, cfg, output)
    identity_gate = metadata_rate >= cfg["metadata_gate_min_supported_rows"]
    support_gate = all(topic_rows[name]["author_topic_coverage"] >= cfg["author_topic_coverage_gate"] and
                       topic_rows[name]["group_exclusion_checked"] for name in PARTITIONS)
    decision = {"status": "B0_COMPLETE", "labels_read": False,
        "author_provenance": "original_dataset_unverified",
        "metadata_sample_only_not_full_bundle_verification": True,
        "sampled_metadata_identity_support_rate": metadata_rate,
        "metadata_gate": {"threshold": cfg["metadata_gate_min_supported_rows"], "passes": identity_gate},
        "topic_support_by_partition": topic_rows,
        "author_topic_support_gate": {"threshold": cfg["author_topic_coverage_gate"], "passes": support_gate},
        "B1_may_run": bool(identity_gate and support_gate),
        "next_step": "run the locked CPU B1 arms" if identity_gate and support_gate else "stop B before quality fitting; retain incumbent and record feasibility failure"}
    write_json(output / "decision.json", decision)
    lines = ["# Author×topic B0 feasibility", "",
        "B0 is label-free. The 30-row metadata audit is a source-quality sample only and does not verify the full author bundle.", "",
        f"Sample seed {cfg['seed']}; nonmissing sampled rows {len(eligible)}, Crossref-supported rows {supported}, rate {metadata_rate:.3f}, gate {identity_gate}.", "",
        "| Partition | Known history | Support 1 | Support 2–4 | Support 5+ | Cold named | Missing/uncertain | Author×topic support ≥2 groups | Coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in PARTITIONS:
        row = topic_rows[name]
        c = row["prediction_author_support_counts"]
        lines.append(f"| {name} | {row['known_author_fraction']:.3f} | {c['support1']} | {c['support2_4']} | {c['support5_plus']} | {c['cold']} | {c['missing_or_uncertain']} | {row['prediction_with_author_topic_support_ge2_groups']} | {row['author_topic_coverage']:.3f} |")
    lines += ["", f"Author×topic support gate: **{'PASS' if support_gate else 'FAIL'}** (≥{cfg['author_topic_coverage_gate']:.0%} in each partition). Identity metadata gate: **{'PASS' if identity_gate else 'FAIL'}** (≥{cfg['metadata_gate_min_supported_rows']:.0%} sampled-row support).", "",
        f"Decision: **{'B1 may run' if decision['B1_may_run'] else 'stop B before quality fitting'}**. The metadata source sample does not establish identity for every record.", ""]
    (output / "REPORT_VI.md").write_text("\n".join(lines))
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["feasibility"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
