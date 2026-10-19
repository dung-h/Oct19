"""Load the immutable source data, embeddings and incumbent feature caches."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EXPERIMENTS = PROJECT / "experiments"
OUT = HERE / "outputs"
SEMANTIC_DIR = EXPERIMENTS / "semantic_roles"
sys.path.insert(0, str(SEMANTIC_DIR))
import roles as semantic_roles

source = semantic_roles.source
ARMS = ["reference", "retrieval_margin", "title_margin", "abstract_margin"]
NEW_ARMS = ARMS[1:]
PRIMARY = "abstract_margin"
READ_THRESHOLD = 0.55
SEED = 20261018
N_FIT = 2494
N_TOTAL = 3090
ENCODER = semantic_roles.ENCODER
ENCODER_REVISION = semantic_roles.ENC_REV


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def context():
    ctx = source.context()
    data = ctx["data"]
    assert len(data) == N_TOTAL
    assert data.split.iloc[:N_FIT].eq("train").all()
    assert data.split.iloc[N_FIT:].eq("test").all()
    assert np.isfinite(ctx["y"][:N_FIT]).all() and np.isnan(ctx["y"][N_FIT:]).all()
    assert ctx["groups"].shape == (N_TOTAL,)

    keys = (data.split.astype(str) + ":" + data.id.astype(str)).to_numpy(str)
    role_out = EXPERIMENTS / "semantic_roles" / "outputs"
    role_texts = pd.read_csv(role_out / "texts.csv", dtype={"id": str}, keep_default_na=False)
    assert role_texts.id.tolist() == data.id.astype(str).tolist()
    assert role_texts.split.tolist() == data.split.astype(str).tolist()
    assert role_texts.title.tolist() == data.title.fillna("").astype(str).tolist()
    assert role_texts.abstract.tolist() == data.abstract.fillna("").astype(str).tolist()

    mini_path = role_out / "embeddings.npz"
    with np.load(mini_path) as z:
        np.testing.assert_array_equal(z["keys"], keys)
        mini = {name: z[name].astype(np.float32, copy=True) for name in ["title", "abstract"]}
    mini_manifest_path = role_out / "embedding_manifest.json"
    mini_manifest = json.loads(mini_manifest_path.read_text())
    assert mini_manifest["input_sha256"] == digest(role_out / "texts.csv")
    assert mini_manifest["stats"] == [
        {"field": "title", "available": N_TOTAL, "truncated": 0},
        {"field": "abstract", "available": int(data.abstract.fillna("").str.strip().ne("").sum()),
         "truncated": 12},
        {"field": "object", "available": 3038, "truncated": 0},
        {"field": "method", "available": 2962, "truncated": 0},
        {"field": "result", "available": 2936, "truncated": 0},
    ]

    retrieval_out = EXPERIMENTS / "diagnose_bottleneck" / "outputs"
    retrieval_path = retrieval_out / "embeddings_retrieval.npz"
    with np.load(retrieval_path) as z:
        np.testing.assert_array_equal(z["keys"], keys)
        retrieval = z["embeddings"].astype(np.float32, copy=True)
    retrieval_manifest_path = retrieval_out / "embeddings_retrieval.json"
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text())

    # Sentence length is recorded per row so truncation cases can be reported.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tokenizer_path = snapshot_download(ENCODER, revision=ENCODER_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.model_max_length = 10**9
    abstracts = data.abstract.fillna("").astype(str).tolist()
    available = np.flatnonzero(np.asarray([bool(x.strip()) for x in abstracts]))
    lengths = tokenizer([abstracts[i] for i in available], truncation=False)["input_ids"]
    truncated = np.zeros(N_TOTAL, dtype=bool)
    truncated[available] = np.asarray([len(x) > 512 for x in lengths], dtype=bool)
    assert int(truncated.sum()) == 12

    abs_available = np.asarray([bool(x.strip()) for x in abstracts], dtype=bool)
    spaces = {
        "retrieval_margin": normalize(retrieval, axis=1),
        "title_margin": normalize(mini["title"], axis=1),
        "abstract_margin": normalize(mini["abstract"], axis=1),
    }
    for name, matrix in spaces.items():
        assert matrix.shape[0] == N_TOTAL
        assert np.isfinite(matrix).all()
        if name == "abstract_margin":
            assert np.all(np.linalg.norm(matrix[abs_available], axis=1) > 0)
        assert np.all(np.linalg.norm(matrix[~abs_available], axis=1) == 0) if name == "abstract_margin" else True

    ctx.update(dict(
        keys=keys,
        labels=ctx["y"][:N_FIT].astype(np.int8),
        ids=data.id.astype(str).to_numpy(),
        venues=data.venue.fillna("unknown").astype(str).to_numpy(),
        abstract_available=abs_available,
        abstract_truncated=truncated,
        spaces=spaces,
        role_texts=role_texts,
        mini_manifest=mini_manifest,
        retrieval_manifest=retrieval_manifest,
        mini_manifest_path=mini_manifest_path,
        retrieval_manifest_path=retrieval_manifest_path,
        mini_embedding_path=mini_path,
        retrieval_embedding_path=retrieval_path,
        tokenizer_path=Path(tokenizer_path),
        catboost_version=importlib.metadata.version("catboost"),
    ))
    return ctx


def outer_jobs(ctx):
    return source.outer_members(ctx)


def load_source_pack(name, fit, pred):
    return source.load_pack(name, np.asarray(fit, int), np.asarray(pred, int))


def source_probabilities(name, fit, pred, ctx):
    fit, pred = np.asarray(fit, int), np.asarray(pred, int)
    if name.startswith("outer"):
        fold = int(name.removeprefix("outer").split("_")[0])
        return source.load_incumbent_outer(fold, fit, pred)
    return source.load_incumbent_final(ctx, fit, pred)


def split_path():
    return EXPERIMENTS / "specter2_v2" / "outputs" / "nested_splits.csv"


def final_pack_path():
    return source.INC / "final_test_features.joblib"

