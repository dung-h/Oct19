#!/usr/bin/env python3
"""Frozen SPECTER2 embeddings and supervised reading-preference regression.

Run embed, train, or all. Original labels are loaded only from the source ZIP.
The outer holdout is never used to choose a model or a selection threshold.
"""
import argparse
import csv
import hashlib
import html
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import time
import unicodedata
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
BASE = "allenai/specter2_base"
BASE_REV = "3447645e1def9117997203454fa4495937bfbd83"
ADAPTER = "allenai/specter2_regression"
ADAPTER_REV = "a9d105def7a081c8f7c5a88ab4145646901c6003"
SEED = 42
THRESHOLDS = np.arange(1.0, 3.51, 0.1).round(2)


def clean_text(text):
    text = html.unescape(str(text or ""))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def parse_test(raw):
    # Test.csv in the supplied ZIP is CSV wrapped in an RTF document.
    text = raw.decode("latin1")
    marker = text.find("\\cf0 ")
    if marker < 0:
        return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    text = text[marker + len("\\cf0 "):]
    text = re.sub(r"\\\r?\n", "\n", text)
    text = re.sub(r"\\u(-?\d+) ?(?:\?)?", lambda m: chr(int(m[1]) % 65536), text)
    text = text.replace("\\uc0", "")
    text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m[1], 16)]).decode("cp1252"), text)
    text = text.replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
    text = "\n".join(line[:-1] if line.endswith("\\") else line for line in text.splitlines())
    return pd.DataFrame(list(csv.DictReader(io.StringIO(text.rsplit("}", 1)[0]))))


def load_data():
    with zipfile.ZipFile(ROOT / "assignment-data-mining-261.zip") as archive:
        train = pd.read_csv(archive.open("train.csv"), dtype=str, keep_default_na=False)
        test = parse_test(archive.read("Test.csv"))
        template = pd.read_csv(archive.open("Submission.csv"), dtype=str)
    assert len(train) == 2494 and len(test) == 596
    assert train.id.is_unique and test.id.is_unique
    assert set(test.id) == set(template.id) and set(train.id).isdisjoint(test.id)
    train["split"], test["split"] = "train", "test"
    data = pd.concat([train, test], ignore_index=True).fillna("")
    abstracts = pd.read_csv(ROOT / "abstracts_collected.csv", dtype=str, keep_default_na=False)
    assert not abstracts.duplicated(["split", "id"]).any()
    data = data.merge(abstracts[["split", "id", "abstract", "identity_review_note"]],
                      on=["split", "id"], how="left", validate="one_to_one", indicator=True)
    assert (data["_merge"] == "both").all()
    data = data.drop(columns="_merge")
    for col in ["title", "abstract", "authors"]:
        data[col] = data[col].map(clean_text)
    # A conservative first experiment falls back to title for unresolved matches.
    data["abstract_withheld"] = data.identity_review_note.str.strip().ne("")
    data["abstract_originally_missing"] = data.abstract.str.strip().eq("")
    data.loc[data.abstract_withheld, "abstract"] = ""
    data["text"] = data.title + " " + data.abstract
    data["year"] = pd.to_numeric(data.year, errors="raise").astype(float)
    return data, template


def input_digest(data):
    records = data[["split", "id", "title", "abstract"]].to_dict("records")
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def embed(data, batch_size):
    import torch
    from adapters import AutoAdapterModel
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    digest = input_digest(data)
    cache = OUT / "embeddings.npz"
    manifest_path = OUT / "embedding_manifest.json"
    if cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["input_sha256"] == digest and manifest["base_revision"] == BASE_REV
                and manifest["adapter_revision"] == ADAPTER_REV):
            print("Using validated embedding cache", flush=True)
            return
        raise ValueError("Embedding cache differs from current inputs/configuration; move it before rerunning.")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = snapshot_download(BASE, revision=BASE_REV, allow_patterns=["*.json", "*.bin", "vocab.txt"])
    adapter_path = snapshot_download(ADAPTER, revision=ADAPTER_REV, allow_patterns=["*.json", "*.bin"])
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoAdapterModel.from_pretrained(base_path)
    model.load_adapter(adapter_path, load_as="regression", set_active=True)
    model.to(device).eval()
    model.requires_grad_(False)
    texts = (data.title + tokenizer.sep_token + data.abstract).tolist()
    enc = tokenizer(texts, padding=False, truncation=False, return_token_type_ids=False)
    lengths = np.array([len(x) for x in enc["input_ids"]])
    order = np.argsort(lengths, kind="stable")
    embeddings = np.zeros((len(data), model.config.hidden_size), dtype=np.float32)
    started = time.monotonic()
    for start in range(0, len(data), batch_size):
        indices = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in indices], padding=True, truncation=True,
                          max_length=512, return_tensors="pt", return_token_type_ids=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=device == "cuda", dtype=torch.float16):
            vectors = model(**batch).last_hidden_state[:, 0, :]
        embeddings[indices] = vectors.float().cpu().numpy()
        if start % (batch_size * 20) == 0 or start + batch_size >= len(data):
            print(f"Embedded {min(start + batch_size, len(data))}/{len(data)}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    assert embeddings.shape == (3090, 768) and np.isfinite(embeddings).all()
    assert (np.linalg.norm(embeddings, axis=1) > 0).all()
    np.savez_compressed(cache, embeddings=embeddings,
                        keys=(data.split + ":" + data.id).to_numpy(dtype=str))
    data[["split", "id", "title", "abstract_originally_missing", "abstract_withheld"]].to_csv(
        OUT / "embedding_index.csv", index=False)
    manifest = {"base_model": BASE, "base_revision": BASE_REV, "adapter": ADAPTER,
                "adapter_revision": ADAPTER_REV, "input_sha256": digest,
                "shape": list(embeddings.shape), "pooling": "CLS", "max_length": 512,
                "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "mixed_precision": device == "cuda", "elapsed_seconds": time.monotonic() - started,
                "truncated_rows": int((lengths > 512).sum()),
                "originally_missing_abstracts": int(data.abstract_originally_missing.sum()),
                "withheld_abstracts_for_identity_review": int(data.abstract_withheld.sum()),
                "title_only_rows": int(data.abstract.eq("").sum()),
                "versions": {p: importlib.metadata.version(p) for p in
                             ["torch", "transformers", "adapters", "huggingface-hub", "numpy", "scikit-learn"]}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")


def grouping(data):
    """Union exact normalized titles and shared plausible publication identifiers."""
    parent = np.arange(len(data))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(indices):
        indices = list(indices)
        for i in indices[1:]:
            parent[root(i)] = root(indices[0])

    def norm_title(title):
        title = unicodedata.normalize("NFKD", title).casefold()
        title = re.sub(r"\((?:extended abstract|short paper|abstract|invited talk|system description)\)", "", title)
        title = re.sub(r"^extended abstract\s*:\s*", "", title)
        return re.sub(r"[^a-z0-9]+", " ", title).strip()

    titles = data.title.map(norm_title)
    for indices in data.groupby(titles).groups.values():
        union(indices)
    for identifier, indices in data.groupby("doi").groups.items():
        if len(indices) > 1 and (identifier.lower().startswith("10.") or "semanticscholar.org/paper/" in identifier):
            union(indices)
    return np.array([root(i) for i in range(len(data))])


def official_metric(labels, predictions, ids, threshold):
    labels, predictions, ids = np.asarray(labels), np.asarray(predictions), np.asarray(ids, dtype=str)
    # Exclusion threshold is chosen on development OOF predictions only.
    selected = np.flatnonzero((predictions > threshold) & (predictions > 1))
    order = selected[np.lexsort((ids[selected], -predictions[selected]))]
    if not len(order):
        return {"score": 0.0, "relevance": 0.0, "coverage": 0.0, "selected": 0}
    weights = 1 / np.log2(np.arange(1, len(order) + 1) + 1)
    quality = float(np.average((labels[order] - 1) / 4, weights=weights))
    denominator = np.count_nonzero(labels > 1)
    recall = float(np.count_nonzero(labels[order] > 1) / denominator) if denominator else 0.0
    score = 2 * quality * recall / (quality + recall) if quality + recall else 0.0
    return {"score": score, "relevance": quality, "coverage": recall, "selected": len(order)}


def model_pipeline(family, parameter, embedding_columns):
    blocks, weights = [], {}
    if "specter2" in family:
        blocks.append(("embedding", Normalizer(), embedding_columns))
        weights["embedding"] = 1.0
    if "tfidf" in family:
        blocks.append(("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                       max_df=0.98, max_features=50000, sublinear_tf=True), "text"))
        weights["text"] = 1.0
    if "metadata" in family:
        blocks += [("venue", OneHotEncoder(handle_unknown="ignore"), ["venue"]),
                   ("year", StandardScaler(), ["year"]),
                   ("authors", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                    max_features=10000, sublinear_tf=True), "authors")]
        weights.update(venue=1.0, year=0.25, authors=0.5)
    feature_builder = ColumnTransformer(blocks, transformer_weights=weights, sparse_threshold=0.9)
    regressor = (SVR(C=parameter, epsilon=0.25, kernel="rbf", cache_size=512)
                 if family.endswith("svr") else Ridge(alpha=parameter, solver="lsqr"))
    return Pipeline([("features", feature_builder), ("regressor", regressor)])


def train(data, template):
    manifest = json.loads((OUT / "embedding_manifest.json").read_text())
    assert manifest["input_sha256"] == input_digest(data), "Stale embedding inputs"
    cache = np.load(OUT / "embeddings.npz")
    assert np.array_equal(cache["keys"], (data.split + ":" + data.id).to_numpy())
    cols = [f"embedding_{i:03d}" for i in range(cache["embeddings"].shape[1])]
    data = pd.concat([data, pd.DataFrame(cache["embeddings"], columns=cols)], axis=1)
    labeled = data[data.split == "train"].reset_index(drop=True)
    test = data[data.split == "test"].reset_index(drop=True)
    y = labeled.Label.astype(int).to_numpy()
    groups = grouping(labeled)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    development, holdout = next(outer.split(labeled, y, groups))
    assert set(groups[development]).isdisjoint(groups[holdout])
    dev = labeled.iloc[development].reset_index(drop=True)
    dev_y = y[development]
    dev_groups = groups[development]
    inner = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1).split(dev, dev_y, dev_groups))
    fold_ids = np.full(len(dev), -1)
    for i, (fit_idx, valid_idx) in enumerate(inner):
        assert set(dev_groups[fit_idx]).isdisjoint(dev_groups[valid_idx])
        fold_ids[valid_idx] = i
    split_table = labeled[["id", "title", "Label"]].copy()
    split_table["group"] = groups
    split_table["partition"] = "holdout"
    split_table["inner_fold"] = -1
    split_table.loc[development, "partition"] = "development"
    split_table.loc[development, "inner_fold"] = fold_ids
    split_table.to_csv(OUT / "validation_split.csv", index=False)
    print(f"Training: development={len(dev)}, untouched holdout={len(holdout)}, grouped records={len(groups)-len(set(groups))}", flush=True)
    families = {
        "metadata_ridge": [1.0, 10.0, 100.0],
        "tfidf_ridge": [1.0, 10.0, 100.0],
        "tfidf_metadata_ridge": [1.0, 10.0, 100.0],
        "specter2_ridge": [0.1, 1.0, 10.0, 100.0],
        "specter2_metadata_ridge": [0.1, 1.0, 10.0, 100.0],
        "specter2_svr": [1.0, 10.0, 50.0],
    }
    candidates, predictions, curves = [], {}, []
    for family, parameters in families.items():
        for parameter in parameters:
            started = time.monotonic()
            oof = np.full(len(dev), np.nan)
            for fit_idx, valid_idx in inner:
                pipeline = model_pipeline(family, parameter, cols)
                pipeline.fit(dev.iloc[fit_idx], dev_y[fit_idx])
                oof[valid_idx] = pipeline.predict(dev.iloc[valid_idx])
            assert np.isfinite(oof).all()
            best = None
            for threshold in THRESHOLDS:
                metrics = [official_metric(dev_y[v], oof[v], dev.id.iloc[v], threshold) for _, v in inner]
                row = {"family": family, "parameter": parameter, "threshold": float(threshold),
                       **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
                       "score_std": float(np.std([m["score"] for m in metrics]))}
                curves.append(row)
                if best is None or row["score"] > best["score"]:
                    best = row
            candidates.append({**best, "mae": float(mean_absolute_error(dev_y, oof))})
            predictions[(family, parameter)] = oof
            print(f"CV {family} param={parameter:g}: H={best['score']:.5f} Q={best['relevance']:.5f} R={best['coverage']:.5f} threshold={best['threshold']:.2f} ({time.monotonic()-started:.1f}s)", flush=True)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    winner = candidates[0]
    specter_winner = next(c for c in candidates if c["family"].startswith("specter2"))
    pd.DataFrame(candidates).to_csv(OUT / "development_cv.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "development_threshold_curves.csv", index=False)
    # Hyperparameters and thresholds are now frozen, before any holdout evaluation.
    family_winners = [next(c for c in candidates if c["family"] == family) for family in families]
    evaluated = []
    holdout_predictions = labeled.iloc[holdout][["id", "title", "Label"]].copy()
    for chosen in family_winners:
        family, parameter, threshold = chosen["family"], chosen["parameter"], chosen["threshold"]
        pipeline = model_pipeline(family, parameter, cols)
        pipeline.fit(dev, dev_y)
        pred = pipeline.predict(labeled.iloc[holdout])
        metrics = official_metric(y[holdout], pred, labeled.id.iloc[holdout], threshold)
        evaluated.append({"family": family, "parameter": parameter, "threshold": threshold,
                          "development_score": chosen["score"], **metrics,
                          "mae": float(mean_absolute_error(y[holdout], pred))})
        holdout_predictions[family + "_prediction"] = pred
        holdout_predictions[family + "_priority"] = priorities(pred, threshold)
        print(f"HOLDOUT {family}: H={metrics['score']:.5f} Q={metrics['relevance']:.5f} R={metrics['coverage']:.5f}", flush=True)
    constant = np.full(len(holdout), dev_y.mean())
    baseline = official_metric(y[holdout], constant, labeled.id.iloc[holdout], 1.0)
    evaluated.append({"family": "recommend_all_constant", "parameter": None, "threshold": 1.0,
                      "development_score": None, **baseline,
                      "mae": float(mean_absolute_error(y[holdout], constant))})
    pd.DataFrame(evaluated).to_csv(OUT / "holdout_metrics.csv", index=False)
    holdout_predictions.to_csv(OUT / "holdout_predictions.csv", index=False)
    oof_export = dev[["id", "title", "Label"]].copy()
    oof_export["fold"] = fold_ids
    for chosen, name in [(winner, "best_dev"), (specter_winner, "specter2")]:
        pred = predictions[(chosen["family"], chosen["parameter"])]
        oof_export[name + "_prediction"] = pred
    oof_export.to_csv(OUT / "development_oof_predictions.csv", index=False)
    for chosen, name in [(specter_winner, "specter2"), (winner, "best_dev")]:
        pipeline = model_pipeline(chosen["family"], chosen["parameter"], cols)
        pipeline.fit(labeled, y)
        pred = pipeline.predict(test)
        submission = pd.DataFrame({"id": test.id, "priority": priorities(pred, chosen["threshold"])})
        submission = template[["id"]].merge(submission, on="id", validate="one_to_one", how="left")
        assert len(submission) == 596 and submission.id.is_unique
        assert np.isfinite(submission.priority).all() and (submission.priority >= 0).all()
        submission.to_csv(OUT / f"submission_{name}.csv", index=False)
        joblib.dump({"pipeline": pipeline, "chosen_on_development": chosen,
                     "embedding_manifest": manifest, "embedding_columns": cols}, OUT / f"model_{name}.joblib")
    summary = {"seed": SEED, "development_rows": len(dev), "holdout_rows": len(holdout),
               "group_count": len(set(groups)), "grouped_extra_records": len(groups)-len(set(groups)),
               "inner_folds": 4, "selection_rule": "highest mean inner-fold harmonic score; holdout never used for selection",
               "selected_overall": winner, "selected_specter2": specter_winner,
               "holdout_results": evaluated, "embedding_manifest": manifest}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print("Saved results, trained models, and 596-row local submission CSVs.", flush=True)


def priorities(predictions, threshold):
    predictions = np.asarray(predictions)
    # No clipping: values above 10 are permitted, and clipping could change rank ties.
    return np.where((predictions > threshold) & (predictions > 1), (predictions - 1) * 2.5, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["embed", "train", "all"], default="all", nargs="?")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data, template = load_data()
    if args.stage in {"embed", "all"}:
        embed(data, args.batch_size)
    if args.stage in {"train", "all"}:
        with threadpool_limits(limits=4):
            train(data, template)


if __name__ == "__main__":
    main()
