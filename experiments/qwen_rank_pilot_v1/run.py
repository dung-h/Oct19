"""Prepare, train, and evaluate the locked two-partition Qwen ranking pilot."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import joblib
import torch
from scipy.special import softmax
from scipy.stats import kendalltau
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = ROOT / "experiments/qwen_remote_v1"
INPUTS = REMOTE / "inputs"
OLD = ROOT / "experiments/qwen_learning_check_v1/outputs/check_v1"
HEAD_RUN = ROOT / "experiments/qwen_head_v1/run.py"
sys.path.insert(0, str(REMOTE))
from common import check_probability, digest, jsonl, official, prompt, sha, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("locked_qwen_head_runner", HEAD_RUN)
head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)

_HEAD_PROB_FROM_HEAD = head.prob_from_head


def fp32_head_probabilities(hidden, state):
    """Keep the affine five-class head FP32 inside backbone BF16 autocast."""
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        float_state = {key: value.float() if torch.is_tensor(value) else value
                       for key, value in state.items()}
        return _HEAD_PROB_FROM_HEAD(hidden.float(), float_state)


# The locked generic trainer scores inside autocast; do not edit its source.
head.prob_from_head = fp32_head_probabilities

PARTITIONS = ("outer1_inner0", "outer1_inner1")
EXPECTED_INCUMBENT_H = {"outer1_inner0": 0.6487764769062809,
                        "outer1_inner1": 0.6463847722551991}
TARGETS_FILE = INPUTS / "targets.jsonl"


def code_hashes():
    files = ("config.json", "PROTOCOL.md", "README.md", "run.py", "verify.py", "run_pilot.sh")
    out = {f"pilot/{name}": sha(HERE / name) for name in files}
    out["locked_head/run.py"] = sha(HEAD_RUN)
    for name in ("config.json", "common.py", "model_core.py", "inputs/manifest.json"):
        out[f"remote/{name}"] = sha(REMOTE / name)
    return out


def runtime_snapshot():
    packages = ("torch", "transformers", "peft", "accelerate", "bitsandbytes",
                "numpy", "scikit-learn", "scipy", "safetensors", "huggingface-hub")
    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True).strip() if torch.cuda.is_available() else None
    return {"code_hashes": code_hashes(),
            "versions": {name: importlib.metadata.version(name) for name in packages},
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "device": device, "driver": driver,
            "allocator": {name: os.environ.get(name) for name in
                          ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}}


def assert_runtime_lock(root):
    lock = json.loads((root / "runtime_lock.json").read_text())
    if lock["runtime"] != runtime_snapshot():
        raise RuntimeError("Source/runtime changed after prepare; preserve artifacts and choose a fresh namespace")
    if lock["config_sha256"] != sha(HERE / "config.json"):
        raise RuntimeError("Pilot config changed after prepare")


def load_config():
    cfg = json.loads((HERE / "config.json").read_text())
    base = json.loads((REMOTE / "config.json").read_text())
    if (cfg["model_id"], cfg["model_revision"]) != (base["model"]["id"], base["model"]["revision"]):
        raise RuntimeError("Pilot model ID/revision differs from the pinned remote config")
    if base["model"]["revision"] != "a09a35458c702b33eeacc393d103063234e8bc28":
        raise RuntimeError("Unexpected Qwen revision")
    if (base["input"]["title_tokens"], base["input"]["abstract_tokens"]) != (80, 560):
        raise RuntimeError("Unexpected prompt token caps")
    return cfg, base


def load_unlabeled_inputs():
    """Load documents, memberships and predictions without parsing target labels."""
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(INPUTS / name) != expected:
            raise RuntimeError(f"Input hash mismatch: {name}")
    docs = {int(row["index"]): row for row in jsonl(INPUTS / "documents.jsonl")}
    if len(docs) != 2494 or len({row["id"] for row in docs.values()}) != 2494:
        raise RuntimeError("Unexpected document bundle")
    parts = json.loads((INPUTS / "partitions.json").read_text())
    baselines = json.loads((INPUTS / "baseline.json").read_text())
    if set(PARTITIONS) - set(parts) or set(PARTITIONS) - set(baselines):
        raise RuntimeError("Locked pilot partitions are absent")
    for name in PARTITIONS:
        fit, pred = parts[name]["fit"], parts[name]["pred"]
        if baselines[name]["fit"] != fit or baselines[name]["pred"] != pred:
            raise RuntimeError(f"Incumbent row order differs for {name}")
        if len(fit) != len(set(fit)) or len(pred) != len(set(pred)):
            raise RuntimeError(f"Duplicate membership for {name}")
        if set(fit) & set(pred) or not set(fit + pred) <= set(docs):
            raise RuntimeError(f"Invalid fit/prediction membership for {name}")
        fit_groups = {docs[i]["group"] for i in fit}
        pred_groups = {docs[i]["group"] for i in pred}
        if fit_groups & pred_groups:
            raise RuntimeError(f"Publication group crosses fit/prediction in {name}")
        check_probability(baselines[name]["probabilities"], len(pred))
    return docs, parts, baselines, manifest


_INDEX_RE = re.compile(r'"index"\s*:\s*(\d+)')


def read_targets_only(indices, expected_docs=None):
    """Parse only requested fit or prediction rows; leave all other ratings untouched."""
    wanted = set(map(int, indices))
    found = {}
    with TARGETS_FILE.open() as stream:
        for line in stream:
            match = _INDEX_RE.search(line)
            if not match:
                raise RuntimeError("Malformed target row without index")
            idx = int(match.group(1))
            if idx in wanted:
                row = json.loads(line)
                found[idx] = int(row["rating"])
                expected_id = (expected_docs[idx]["id"] if expected_docs is not None else None)
                if expected_id is not None and row["id"] != expected_id:
                    raise RuntimeError(f"Target/document ID mismatch for {idx}")
    if set(found) != wanted:
        raise RuntimeError(f"Missing requested target rows: {sorted(wanted - set(found))[:10]}")
    return found


def old_cache_verification():
    verification = json.loads((OLD / "postrun_verification.json").read_text())
    lock = json.loads((OLD / "runtime_lock.json").read_text())
    if verification.get("status") != "ARTIFACTS_VERIFIED" or not verification.get("source_and_code_unchanged"):
        raise RuntimeError("Frozen feature cache is not marked verified")
    old_sources = {name: sha(REMOTE / name) for name in lock["source"]}
    if old_sources != lock["source"]:
        raise RuntimeError("Frozen cache remote source hashes have changed")
    old_package = ROOT / "experiments/qwen_learning_check_v1"
    old_code = {name: sha(old_package / name) for name in lock["code"]}
    if old_code != lock["code"]:
        raise RuntimeError("Frozen cache producing code hashes have changed")
    now = runtime_snapshot()
    for key, expected in lock["versions"].items():
        if now["versions"].get(key) != expected:
            raise RuntimeError(f"Frozen cache package version differs: {key}")
    for key in ("python", "cuda", "device", "driver", "allocator"):
        if now[key] != lock.get(key):
            raise RuntimeError(f"Frozen cache runtime differs: {key}")
    extract = json.loads((OLD / "EXTRACT_COMPLETE.json").read_text())
    if extract.get("status") != "FEATURES_COMPLETE" or sha(OLD / "features.npz") != extract.get("features_sha"):
        raise RuntimeError("Frozen feature artifact hash mismatch")
    old_members = json.loads((OLD / "membership.json").read_text())
    fp = np.load(OLD / "features.npz")
    if fp["hidden"].shape != (1494, 3584) or fp["logits"].shape != (1494, 5):
        raise RuntimeError("Frozen cache feature shape mismatch")
    if list(map(int, fp["indices"])) != list(map(int, old_members["pool"])):
        raise RuntimeError("Frozen cache row order differs from its membership")
    return {"runtime_lock_sha256": sha(OLD / "runtime_lock.json"),
            "postrun_verification_sha256": sha(OLD / "postrun_verification.json"),
            "features_sha256": extract["features_sha"],
            "features_shape": list(fp["hidden"].shape),
            "membership_sha256": sha(OLD / "membership.json"),
            "source": lock["source"], "producing_code": lock["code"]}


def build_members(part, docs, cfg):
    train = list(map(int, part["fit"]))
    validation = list(map(int, part["pred"]))
    if set(train) & set(validation):
        raise RuntimeError("Fit and prediction IDs overlap")
    labels = read_targets_only(train, docs)
    if set(labels.values()) != {1, 2, 3, 4, 5}:
        raise RuntimeError("Fit split must contain all five labels")
    rng = np.random.default_rng(cfg["seed"])
    probes = []
    # A fixed stratified probe set, never used for optimizer updates.
    labels_by_class = {label: [i for i in train if labels[i] == label] for label in range(1, 6)}
    chosen = []
    for label, rows in labels_by_class.items():
        quota = int(cfg["train_probes"]) // 5
        chosen.extend(rng.choice(rows, size=quota, replace=False).tolist())
    remainder = cfg["train_probes"] - len(chosen)
    if remainder:
        left = sorted(set(train) - set(chosen))
        chosen.extend(rng.choice(left, size=remainder, replace=False).tolist())
    probes = sorted(map(int, chosen))
    if len(probes) != cfg["train_probes"] or not set(probes) <= set(train):
        raise RuntimeError("Invalid fixed fit-only train-probe membership")
    groups_a = {docs[i]["group"] for i in train}
    groups_b = {docs[i]["group"] for i in validation}
    if groups_a & groups_b:
        raise RuntimeError("Group leakage in fit/prediction")
    return {"pool": train, "train": train, "validation": validation,
            "train_probes": probes, "fit_labels_sha256": digest(sorted(labels.items()))}, labels


def extract_or_reuse_features(root, docs, cfg, base, parts):
    import torch
    feature_start = time.monotonic()
    cache_info = old_cache_verification()
    cached = np.load(OLD / "features.npz")
    rows = {int(idx): (cached["hidden"][pos].astype(np.float32),
                       cached["logits"][pos].astype(np.float32))
            for pos, idx in enumerate(cached["indices"])}
    required = sorted(set(i for name in PARTITIONS
                          for i in (parts[name]["fit"] + parts[name]["pred"])))
    missing = [i for i in required if i not in rows]
    tokenizer, answer = head.load_tokenizer(base)
    if answer != [16, 17, 18, 19, 20]:
        raise RuntimeError(f"Pinned tokenizer rating IDs changed: {answer}")
    preflight_path = root / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["tokenizer_verified_in_prepare"] = True
    preflight["rating_token_ids"] = answer
    write_json(preflight_path, preflight)
    token_records = {}
    for idx in required:
        ids, stats = prompt(tokenizer, docs[idx], base)
        token_records[idx] = {"index": idx, "token_count": len(ids),
                              "token_sha256": digest(ids), "title": stats["title"],
                              "abstract": stats["abstract"]}

    parity = {"hidden_max_abs_error": 0.0, "logit_max_abs_error": 0.0,
              "anchor_index": None}
    if missing:
        load_start = time.monotonic()
        model, _ = head.load_model(base, trainable=False, seed=cfg["seed"])
        frozen_model_load_seconds = time.monotonic() - load_start
        captured, hook = head.attach_hidden_hook(model)
        model.eval()
        anchors = [int(cached["indices"][0]), int(cached["indices"][len(cached["indices"]) // 2]),
                   int(cached["indices"][-1])]
        parity_anchor = anchors[1]
        ids, _ = prompt(tokenizer, docs[parity_anchor], base)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            h, z = head.hidden_and_digit_logits(
                model, torch.tensor([ids], device="cuda", dtype=torch.long), answer, captured)
        cache_pos = {int(idx): pos for pos, idx in enumerate(cached["indices"])}[parity_anchor]
        parity["hidden_max_abs_error"] = float(np.max(np.abs(h[0].float().cpu().numpy() - cached["hidden"][cache_pos].astype(np.float32))))
        parity["logit_max_abs_error"] = float(np.max(np.abs(z[0].float().cpu().numpy() - cached["logits"][cache_pos].astype(np.float32))))
        parity["anchor_index"] = parity_anchor
        if max(parity["hidden_max_abs_error"], parity["logit_max_abs_error"]) > 1e-4:
            hook.remove(); del model; gc.collect(); torch.cuda.empty_cache()
            raise RuntimeError(f"Frozen cache forward parity failed: {parity}")
        audit_path = root / "token_audit.jsonl"
        with audit_path.open("x") as audit:
            for idx in required:
                audit.write(json.dumps(token_records[idx], ensure_ascii=False) + "\n")
        for ordinal, idx in enumerate(missing, start=1):
            prompt_ids, _ = prompt(tokenizer, docs[idx], base)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                h, z = head.hidden_and_digit_logits(
                    model, torch.tensor([prompt_ids], device="cuda", dtype=torch.long), answer, captured)
            rows[idx] = (h[0].float().cpu().numpy().astype(np.float32),
                         z[0].float().cpu().numpy().astype(np.float32))
            if ordinal % 50 == 0 or ordinal == len(missing):
                print(json.dumps({"stage": "frozen_feature_extraction", "done": ordinal,
                                  "total": len(missing), "index": idx,
                                  "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
        hook.remove(); del model; gc.collect(); torch.cuda.empty_cache()
    else:
        frozen_model_load_seconds = 0.0
        with (root / "token_audit.jsonl").open("x") as audit:
            for idx in required:
                audit.write(json.dumps(token_records[idx], ensure_ascii=False) + "\n")
    if set(rows) < set(required):
        raise RuntimeError("Frozen feature extraction did not cover pilot rows")
    ordered_hidden = np.stack([rows[i][0] for i in required]).astype(np.float32)
    ordered_logits = np.stack([rows[i][1] for i in required]).astype(np.float32)
    feature_file = root / "frozen_features.npz"
    np.savez_compressed(feature_file, indices=np.asarray(required, dtype=np.int64),
                        hidden=ordered_hidden, logits=ordered_logits)
    pos = {idx: k for k, idx in enumerate(required)}
    row_sum_error = 0.0
    for name in PARTITIONS:
        part_root = root / name
        part_root.mkdir()
        members, labels = build_members(parts[name], docs, cfg)
        write_json(part_root / "membership.json", members)
        write_json(part_root / "fit_labels.json", {str(k): v for k, v in labels.items()})
        member_pos = [pos[i] for i in required]
        # The helper trainer expects the shared features file at each partition root.
        shutil.copy2(feature_file, part_root / "features.npz")
        frozen_probe = softmax(ordered_logits[[pos[i] for i in members["train_probes"]]].astype(np.float64), axis=1)
        row_sum_error = max(row_sum_error, float(np.max(np.abs(frozen_probe.sum(1) - 1.0))))
        np.savez_compressed(part_root / "control_predictions.npz",
                            frozen_digit_train_probes=frozen_probe)
        write_json(part_root / "membership_lock.json", {"membership_sha256": sha(part_root / "membership.json"),
            "fit_ids_sha256": digest(members["train"]), "prediction_ids_sha256": digest(members["validation"]),
            "fit_labels_sha256": members["fit_labels_sha256"], "fit_label_count": len(labels),
            "prediction_label_values_loaded": False})
    write_json(root / "source_cache_runtime_lock.json", cache_info)
    write_json(root / "feature_extraction.json", {"status": "FEATURES_COMPLETE",
        "required_rows": len(required), "reused_cache_rows": len(cached["indices"]),
        "newly_extracted_rows": len(missing), "frozen_cache_parity": parity,
        "frozen_model_load_seconds": frozen_model_load_seconds,
        "feature_stage_seconds": time.monotonic() - feature_start,
        "frozen_digit_probability_row_sum_max_error": row_sum_error,
        "features_sha256": sha(feature_file), "token_audit_sha256": sha(root / "token_audit.jsonl"),
        "feature_indices_sha256": digest(required), "label_free": True})


def record_stage(root, stage, status, detail=None):
    path = root / "stage_status.json"
    current = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    current["stages"][stage] = {"status": status, "time_unix": time.time(), "detail": detail or {}}
    write_json(path, current)


def prepare(root):
    if root.exists():
        raise RuntimeError("Output namespace already exists; preserve it and choose a fresh namespace")
    cfg, base = load_config()
    docs, parts, baselines, manifest = load_unlabeled_inputs()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked Qwen pilot")
    if torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The locked recipe requires a BF16-capable GPU")
    gpu = {"name": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0)),
           "bf16_supported": bool(torch.cuda.is_bf16_supported()),
           "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
           "free_memory_bytes": int(torch.cuda.mem_get_info(0)[0])}
    disk = shutil.disk_usage(root.parent if root.parent.exists() else ROOT)
    if disk.free < 30 * 1024**3:
        raise RuntimeError(f"At least 30 GiB free disk required; found {disk.free / 1024**3:.1f} GiB")
    # Establish the immutable source/runtime fingerprint before creating outputs.
    snapshot = runtime_snapshot()
    root.mkdir(parents=True)
    write_json(root / "runtime_lock.json", {"runtime": snapshot,
        "config_sha256": sha(HERE / "config.json"),
        "input_manifest_sha256": sha(INPUTS / "manifest.json"),
        "partition_names": list(PARTITIONS), "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"]})
    write_json(root / "preflight.json", {"gpu": gpu, "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free, "hf_home": os.environ.get("HF_HOME"),
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "tokenizer_verified_in_prepare": False})
    write_json(root / "baseline_replay_contract.json", {
        name: {"fit_ids_match": True, "prediction_ids_match": True,
               "prediction_rows": len(baselines[name]["pred"]),
               "probability_max_row_sum_error": float(np.max(np.abs(np.asarray(baselines[name]["probabilities"]).sum(1) - 1.0))),
               "registered_incumbent_H": EXPECTED_INCUMBENT_H[name],
               "prediction_labels_loaded": False}
        for name in PARTITIONS})
    record_stage(root, "prepare", "RUNNING", {"documents": len(docs), "manifest_sha256": sha(INPUTS / "manifest.json")})
    try:
        extract_or_reuse_features(root, docs, cfg, base, parts)
        record_stage(root, "prepare", "COMPLETE", json.loads((root / "feature_extraction.json").read_text()))
    except Exception as exc:
        write_json(root / "prepare_FAILED.json", {"error": f"{type(exc).__name__}: {exc}", "time_unix": time.time()})
        record_stage(root, "prepare", "FAILED", {"error": f"{type(exc).__name__}: {exc}"})
        raise


def read_partition_context(root, partition):
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition {partition}")
    assert_runtime_lock(root)
    docs, parts, baselines, _ = load_unlabeled_inputs()
    members = json.loads((root / partition / "membership.json").read_text())
    fit_labels = {int(k): int(v) for k, v in json.loads((root / partition / "fit_labels.json").read_text()).items()}
    if set(fit_labels) != set(members["train"]):
        raise RuntimeError("Fit-only label cache does not align with training membership")
    return docs, parts, baselines, members, fit_labels


def run_partition_stage(root, partition, stage):
    cfg, base = load_config()
    docs, _, _, members, fit_labels = read_partition_context(root, partition)
    part_root = root / partition
    if stage == "cpu-head":
        head.cpu_head(part_root, cfg, docs, fit_labels, members)
        record_stage(root, f"{partition}.cpu_head", "COMPLETE", json.loads((part_root / "HEAD_COMPLETE.json").read_text()))
        return
    if stage == "smoke":
        import torch
        tokenizer, answer = head.load_tokenizer(base)
        features = head.load_feature(part_root)
        feature_pos = head.feature_positions(features)
        anchors = [next(i for i in members["train"] if fit_labels[i] == label)
                   for label in range(1, 6)]
        head_data = joblib.load(part_root / "head_B.joblib")
        head_state = head.make_head_state(head_data)
        frozen_model, _ = head.load_model(base, trainable=False, seed=cfg["seed"])
        captured, hook = head.attach_hidden_hook(frozen_model)
        hidden_error = probability_error = 0.0
        with torch.inference_mode():
            for idx in anchors:
                token_ids, _ = prompt(tokenizer, docs[idx], base)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h, _ = head.hidden_and_digit_logits(
                        frozen_model, torch.tensor([token_ids], device="cuda", dtype=torch.long), answer, captured)
                cached = features["hidden"][feature_pos[idx]].astype(np.float32)
                hidden_error = max(hidden_error, float(np.max(np.abs(h[0].float().cpu().numpy() - cached))))
                # Use the locked serialized scaler values directly; no refit occurs here.
                cpu_z = ((cached.astype(np.float64) - head_data["mean"].astype(np.float64)) /
                         head_data["scale"].astype(np.float64)) @ head_data["weight"].astype(np.float64).T + head_data["bias"].astype(np.float64)
                cpu_p = softmax(cpu_z.reshape(1, -1), axis=1)[0]
                gpu_p = head.prob_from_head(h, {k: v.to("cuda") if torch.is_tensor(v) else v
                                                for k, v in head_state.items()})[0].cpu().numpy()
                probability_error = max(probability_error, float(np.max(np.abs(cpu_p - gpu_p))))
        hook.remove(); del frozen_model; gc.collect(); torch.cuda.empty_cache()
        if hidden_error > 1e-4 or probability_error > 1e-4:
            raise RuntimeError(f"Frozen feature/head CPU-GPU parity failed: hidden={hidden_error}, probability={probability_error}")
        write_json(part_root / "head_cpu_gpu_parity.json", {
            "status": "PASS", "anchor_indices": anchors,
            "hidden_max_abs_error": hidden_error,
            "probability_max_abs_error": probability_error,
            "tolerance": 1e-4})
        head.smoke(part_root, cfg, base, docs, fit_labels, members)
        record_stage(root, f"{partition}.smoke", "COMPLETE", {
            "head_cpu_gpu_parity": json.loads((part_root / "head_cpu_gpu_parity.json").read_text()),
            "smoke": json.loads((part_root / "smoke/COMPLETE.json").read_text())})
        return
    if stage == "train":
        if not (part_root / "smoke/COMPLETE.json").exists():
            raise RuntimeError("Fresh successful smoke is required")
        original_jsonl = head.jsonl
        allowed = set(members["train"])

        def training_labels_only(path):
            if Path(path).resolve() == TARGETS_FILE.resolve():
                return [{"index": idx, "rating": label, "id": docs[idx]["id"]}
                        for idx, label in read_targets_only(allowed, docs).items()]
            raise RuntimeError(f"Training helper requested an unexpected JSONL: {path}")

        head.jsonl = training_labels_only
        try:
            head.train(part_root, cfg, base, docs, members)
        finally:
            head.jsonl = original_jsonl
        complete = json.loads((part_root / "training/COMPLETE.json").read_text())
        record_stage(root, f"{partition}.train", "COMPLETE", complete)
        return
    raise ValueError(stage)


def gain(p):
    return np.asarray(p, dtype=float) @ np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])


def selected_priority(selected, ranked_gain):
    out = np.zeros(len(selected), dtype=float)
    out[selected] = 1.0 + 9.0 * np.asarray(ranked_gain, dtype=float)[selected]
    return out


def native_priority(p, cutoff):
    return selected_priority((1.0 - p[:, 0]) > cutoff, gain(p))


def matched_budget_mask(read_score, ids, count):
    order = np.lexsort((np.asarray(ids, dtype=str), -np.asarray(read_score, dtype=float)))
    selected = np.zeros(len(ids), dtype=bool)
    selected[order[:count]] = True
    return selected


def probability_metrics(y, p):
    raw = np.asarray(p, dtype=np.float64)
    row_error = float(np.max(np.abs(raw.sum(1) - 1.0)))
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw.sum(1) <= 0):
        raise RuntimeError("Invalid probability matrix")
    normalized = raw / raw.sum(1, keepdims=True)
    read_y = (np.asarray(y) > 1).astype(int)
    read_p = 1.0 - raw[:, 0]
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for k in range(len(bins) - 1):
        mask = ((read_p >= bins[k]) & (read_p < bins[k + 1]))
        if k == len(bins) - 2:
            mask |= read_p == 1.0
        if mask.any():
            ece += float(mask.mean() * abs(read_p[mask].mean() - read_y[mask].mean()))
    return {"five_class_logloss": float(log_loss(y, normalized, labels=[1, 2, 3, 4, 5])),
            "read_auc": float(roc_auc_score(read_y, read_p)),
            "read_logloss": float(log_loss(read_y, read_p, labels=[0, 1])),
            "read_brier": float(np.mean((read_p - read_y) ** 2)),
            "read_ece_10_bins": float(ece),
            "probability_row_sum_max_error": row_error}


def confusion_and_classes(y, p):
    pred = np.asarray(p).argmax(1) + 1
    matrix = np.zeros((5, 5), dtype=int)
    for actual, guess in zip(y, pred):
        matrix[int(actual) - 1, int(guess) - 1] += 1
    metrics = []
    for label in range(1, 6):
        tp = int(matrix[label - 1, label - 1])
        support = int(np.sum(np.asarray(y) == label))
        predicted = int(np.sum(pred == label))
        metrics.append({"label": label, "support": support, "predicted": predicted,
                        "precision": tp / predicted if predicted else 0.0,
                        "recall": tp / support if support else 0.0})
    return matrix.tolist(), metrics


def metric_for_policy(y, priority, ids):
    return official(y, priority, ids)


def paired_group_bootstrap(y, candidate_priority, anchor_priority, ids, groups, repeats, seed):
    keys = np.asarray(sorted(set(groups)))
    by_group = {g: np.flatnonzero(np.asarray(groups) == g) for g in keys}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        ix = np.concatenate([by_group[g] for g in chosen])
        deltas.append(official(np.asarray(y)[ix], np.asarray(candidate_priority)[ix], np.asarray(ids)[ix])["H"] -
                      official(np.asarray(y)[ix], np.asarray(anchor_priority)[ix], np.asarray(ids)[ix])["H"])
    return {"lower_2_5": float(np.quantile(deltas, .025)),
            "upper_97_5": float(np.quantile(deltas, .975)),
            "mean": float(np.mean(deltas)), "repeats": int(repeats), "groups": int(len(keys))}


def ranking_diagnostics(y, ids, selected, old_gain, new_gain):
    chosen = np.flatnonzero(selected)
    old_order = chosen[np.lexsort((np.asarray(ids)[chosen], -np.asarray(old_gain)[chosen]))]
    new_order = chosen[np.lexsort((np.asarray(ids)[chosen], -np.asarray(new_gain)[chosen]))]
    old_rank = {int(idx): rank + 1 for rank, idx in enumerate(old_order)}
    new_rank = {int(idx): rank + 1 for rank, idx in enumerate(new_order)}
    changes = [{"id": str(ids[idx]), "old_rank": old_rank[int(idx)],
                "trained_rank": new_rank[int(idx)],
                "rank_change": new_rank[int(idx)] - old_rank[int(idx)]}
               for idx in chosen]
    changes.sort(key=lambda row: (-abs(row["rank_change"]), row["id"]))
    weights = 1.0 / np.log2(np.arange(len(chosen)) + 2.0)
    weight_sum = float(weights.sum())
    old_order_rows = {int(idx): rank for rank, idx in enumerate(old_order)}
    new_order_rows = {int(idx): rank for rank, idx in enumerate(new_order)}
    relevance = (np.asarray(y, dtype=float) - 1.0) / 4.0
    by_label = []
    for label in range(1, 6):
        subset = [int(idx) for idx in chosen if y[idx] == label]
        old_part = sum(weights[old_order_rows[idx]] * relevance[idx] for idx in subset) / weight_sum if weight_sum else 0.0
        new_part = sum(weights[new_order_rows[idx]] * relevance[idx] for idx in subset) / weight_sum if weight_sum else 0.0
        by_label.append({"true_label": label, "count_selected": len(subset),
                         "incumbent_Q_contribution": float(old_part),
                         "trained_Q_contribution": float(new_part),
                         "delta_Q_contribution": float(new_part - old_part)})
    tau = kendalltau([old_rank[int(i)] for i in chosen], [new_rank[int(i)] for i in chosen]).statistic if len(chosen) > 1 else 1.0
    return {"selected_count": len(chosen), "moved_count": int(sum(r["rank_change"] != 0 for r in changes)),
            "mean_absolute_rank_change": float(np.mean([abs(r["rank_change"]) for r in changes])) if changes else 0.0,
            "max_absolute_rank_change": max([abs(r["rank_change"]) for r in changes], default=0),
            "kendall_tau": float(tau) if np.isfinite(tau) else None,
            "largest_rank_moves": changes[:20], "classwise_Q_contribution": by_label}


def evaluate(root):
    if (root / "decision.json").exists():
        raise RuntimeError("Quality evaluation already exists; preserve it")
    assert_runtime_lock(root)
    cfg, _ = load_config()
    # Gate before touching target-label rows: both complete adapters and scores are mandatory.
    completions = {}
    for name in PARTITIONS:
        part_root = root / name
        complete_path = part_root / "training/COMPLETE.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"Both partitions must finish training/scoring before evaluation: {name}")
        complete = json.loads(complete_path.read_text())
        if complete.get("status") != "TRAINING_COMPLETE" or complete.get("epochs") != cfg["epochs"]:
            raise RuntimeError(f"Incomplete training for {name}")
        final_pred = part_root / f"training/epoch{cfg['epochs']}_predictions.npz"
        if not final_pred.is_file() or not (part_root / "HEAD_COMPLETE.json").is_file() or not (part_root / "smoke/COMPLETE.json").is_file():
            raise FileNotFoundError(f"Required final predictions/head/smoke missing for {name}")
        completions[name] = complete
    docs, parts, baselines, _ = load_unlabeled_inputs()
    all_prediction_ids = [idx for name in PARTITIONS for idx in parts[name]["pred"]]
    pred_targets = read_targets_only(all_prediction_ids, docs)
    reports, policy_rows, confusions, class_rows = {}, {}, {}, {}
    primary_boots = {}
    for part_no, name in enumerate(PARTITIONS):
        part_root = root / name
        ids_index = list(map(int, parts[name]["pred"]))
        ids = np.asarray([docs[i]["id"] for i in ids_index], dtype=str)
        y = np.asarray([pred_targets[i] for i in ids_index], dtype=int)
        baseline = np.asarray(baselines[name]["probabilities"], dtype=np.float64)
        if len(ids) != len(y) or not set(y) <= {1, 2, 3, 4, 5}:
            raise RuntimeError(f"Prediction truth alignment failure in {name}")
        trained = np.load(part_root / f"training/epoch{cfg['epochs']}_predictions.npz")["validation"].astype(np.float64)
        frozen = np.load(part_root / "head_B_predictions.npz")["validation"].astype(np.float64)
        if any(p.shape != (len(y), 5) for p in (baseline, trained, frozen)):
            raise RuntimeError(f"Five-class prediction shape/order failure in {name}")
        pmetrics = {"incumbent": probability_metrics(y, baseline),
                    "frozen_B": probability_metrics(y, frozen),
                    "trained_C": probability_metrics(y, trained)}
        baseline_read = 1.0 - baseline[:, 0]
        trained_read = 1.0 - trained[:, 0]
        incumbent_selected = baseline_read > cfg["read_cutoff"]
        trained_selected = trained_read > cfg["read_cutoff"]
        incumbent_gain, trained_gain, frozen_gain = gain(baseline), gain(trained), gain(frozen)
        priorities = {
            "incumbent_native": selected_priority(incumbent_selected, incumbent_gain),
            "trained_primary_incumbent_selection": selected_priority(incumbent_selected, trained_gain),
            "trained_native": selected_priority(trained_selected, trained_gain),
            "frozen_B_incumbent_selection": selected_priority(incumbent_selected, frozen_gain),
            "selection_only": selected_priority(trained_selected, incumbent_gain),
            "ranking_only": selected_priority(incumbent_selected, trained_gain),
            "matched_budget": selected_priority(matched_budget_mask(trained_read, ids,
                                              int(incumbent_selected.sum())), trained_gain),
        }
        arm_metrics = {arm: metric_for_policy(y, priority, ids)
                       for arm, priority in priorities.items()}
        primary = arm_metrics["trained_primary_incumbent_selection"]
        incumbent = arm_metrics["incumbent_native"]
        selected_ids = [ids[i] for i in np.flatnonzero(incumbent_selected)]
        primary_ids = [ids[i] for i in np.flatnonzero(priorities["trained_primary_incumbent_selection"] > 0)]
        selection_invariant = selected_ids == primary_ids
        invariant_fields = ("selected", "TP", "FP", "FN", "TN", "missed_label4", "missed_label5")
        count_invariant = all(primary[key] == incumbent[key] for key in invariant_fields)
        if not selection_invariant or not count_invariant or primary["R"] != incumbent["R"]:
            raise RuntimeError(f"Primary selection invariant failed for {name}")
        matrices = {}
        per_class = {}
        for arm_name, p in (("incumbent_native", baseline), ("frozen_B", frozen),
                            ("trained_C_native", trained), ("trained_C_primary", trained)):
            matrices[arm_name], per_class[arm_name] = confusion_and_classes(y, p)
        selection_transitions = []
        for label in range(1, 6):
            label_mask = y == label
            selection_transitions.append({"true_label": label,
                "added_by_trained_native": int((trained_selected & ~incumbent_selected & label_mask).sum()),
                "dropped_by_trained_native": int((~trained_selected & incumbent_selected & label_mask).sum()),
                "incumbent_selected": int((incumbent_selected & label_mask).sum()),
                "trained_selected": int((trained_selected & label_mask).sum()),
                "total": int(label_mask.sum())})
        rank_diag = ranking_diagnostics(y, ids, incumbent_selected, incumbent_gain, trained_gain)
        reports[name] = {"n_fit": len(parts[name]["fit"]), "n_prediction": len(y),
            "incumbent_expected_H": EXPECTED_INCUMBENT_H[name],
            "incumbent_replayed_H": incumbent["H"],
            "incumbent_H_replay_error": incumbent["H"] - EXPECTED_INCUMBENT_H[name],
            "probability_metrics": pmetrics, "policies": arm_metrics,
            "primary_delta_H": primary["H"] - incumbent["H"],
            "selection_invariant": selection_invariant, "count_invariants": count_invariant,
            "selected_ids_sha256": digest(selected_ids), "confusion_matrices_argmax_diagnostic": matrices,
            "class_precision_recall_argmax_diagnostic": per_class,
            "selection_transitions_by_true_label": selection_transitions,
            "ranking_only_diagnostics": rank_diag,
            "lambda": json.loads((part_root / "lambda_cv_details.json").read_text())["selected_lambda"],
            "training": completions[name]}
        policy_rows[name] = arm_metrics
        confusions[name] = matrices
        class_rows[name] = per_class
        primary_boots[name] = paired_group_bootstrap(
            y, priorities["trained_primary_incumbent_selection"], priorities["incumbent_native"],
            ids, [docs[i]["group"] for i in ids_index], cfg["bootstrap_repeats"], cfg["seed"] + part_no)
        # Store exact validation probabilities only in this post-training evaluator output.
        np.savez_compressed(part_root / "quality_predictions.npz", indices=np.asarray(ids_index),
                            incumbent=baseline, frozen_B=frozen, trained_C=trained,
                            labels=y)
    primary_delta = [reports[name]["primary_delta_H"] for name in PARTITIONS]
    mean_incumbent = float(np.mean([reports[name]["incumbent_replayed_H"] for name in PARTITIONS]))
    mean_primary = float(np.mean([reports[name]["policies"]["trained_primary_incumbent_selection"]["H"] for name in PARTITIONS]))
    gate = {"technical_pass": True,
            "selection_and_count_invariants_pass": all(reports[name]["selection_invariant"] and reports[name]["count_invariants"] for name in PARTITIONS),
            "mean_primary_delta_H": mean_primary - mean_incumbent,
            "mean_delta_gate": mean_primary - mean_incumbent >= cfg["gate"]["mean_H_improvement_min"],
            "each_partition_nonnegative": all(value >= cfg["gate"]["each_partition_H_change_min"] for value in primary_delta),
            "partition_delta_H": dict(zip(PARTITIONS, primary_delta))}
    gate["passes"] = gate["technical_pass"] and gate["selection_and_count_invariants_pass"] and gate["mean_delta_gate"] and gate["each_partition_nonnegative"]
    policy_means = {arm: {metric: float(np.mean([reports[name]["policies"][arm][metric] for name in PARTITIONS]))
                          for metric in ("H", "Q", "R")}
                    for arm in next(iter(policy_rows.values()))}
    frozen_to_trained = {}
    for metric in ("five_class_logloss", "read_auc", "read_logloss", "read_brier", "read_ece_10_bins"):
        deltas = {name: reports[name]["probability_metrics"]["trained_C"][metric] -
                  reports[name]["probability_metrics"]["frozen_B"][metric] for name in PARTITIONS}
        frozen_to_trained[metric] = {"partition_delta": deltas,
                                    "mean_delta": float(np.mean(list(deltas.values())))}
    frozen_to_trained["useful_preference_learning_evidence"] = bool(
        frozen_to_trained["five_class_logloss"]["mean_delta"] <= -0.01 and
        frozen_to_trained["read_auc"]["mean_delta"] >= 0.0)
    decision = {"status": "QUALITY_EVALUATED", "exploratory_seen_partitions": True,
        "prediction_labels_read_after_both_partitions_complete": True,
        "output_path": str(root), "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "partition_metrics": reports, "mean_incumbent_H": mean_incumbent,
        "mean_primary_H": mean_primary, "mean_policy_H_Q_R": policy_means,
        "frozen_to_trained": frozen_to_trained,
        "mean_primary_Q": policy_means["trained_primary_incumbent_selection"]["Q"],
        "mean_primary_R": policy_means["trained_primary_incumbent_selection"]["R"],
        "summed_primary_counts": {key: sum(reports[n]["policies"]["trained_primary_incumbent_selection"][key] for n in PARTITIONS)
                                  for key in ("selected", "TP", "FP", "FN", "TN", "missed_label4", "missed_label5")},
        "paired_group_bootstrap_primary_delta_H": primary_boots, "gate": gate,
        "next_step": "separately review budget for the locked confirmation protocol" if gate["passes"] else "stop the current Qwen ranking branch and retain the incumbent"}
    write_json(root / "decision.json", decision)
    write_json(root / "confusion_matrices.json", confusions)
    write_json(root / "class_precision_recall.json", class_rows)
    write_json(root / "policy_metrics.json", policy_rows)
    write_json(root / "VALIDATION_LABELS_EVALUATED.json", {"status": "COMPLETE", "prediction_rows": len(all_prediction_ids),
        "prediction_ids_sha256": digest(all_prediction_ids), "labels_sha256": digest(sorted(pred_targets.items()))})
    write_report(root / "REPORT_VI.md", decision, cfg, root)
    record_stage(root, "evaluate", "COMPLETE", {"gate_pass": gate["passes"], "mean_delta_H": gate["mean_primary_delta_H"]})


def write_report(path, decision, cfg, root):
    parts = decision["partition_metrics"]
    preflight = json.loads((root / "preflight.json").read_text())
    runtime = json.loads((root / "runtime_lock.json").read_text())["runtime"]
    extraction = json.loads((root / "feature_extraction.json").read_text())
    lines = ["# Báo cáo pilot Qwen ranking", "",
        "Đây là pilot exploratory trên hai prediction partitions đã được xem trước. Không gộp hai ranking; H/Q/R được tính từng fold rồi lấy mean, còn confusion và counts được cộng.", "",
        f"GPU: {preflight['gpu']['name']}, VRAM {preflight['gpu']['total_memory_bytes']/1024**3:.1f} GiB, driver {runtime['driver']}, CUDA torch {runtime['cuda']}; Python {runtime['python']}, torch {runtime['versions']['torch']}, Transformers {runtime['versions']['transformers']}, PEFT {runtime['versions']['peft']}, bitsandbytes {runtime['versions']['bitsandbytes']}. Model `{decision['model_id']}` revision `{decision['model_revision']}`. Output: `{root}`.",
        f"Frozen feature stage {extraction['feature_stage_seconds']:.1f}s (model load {extraction['frozen_model_load_seconds']:.1f}s; reused {extraction['reused_cache_rows']} verified rows and extracted {extraction['newly_extracted_rows']}).",
        "## Frozen head → trained Qwen", "",
        "| Partition | Model | 5-class CE | Read AUC | Read CE | Read Brier | Read ECE | H | Q | R | TP/FP/FN/TN | Missed 4/5 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for name in PARTITIONS:
        r = parts[name]
        for arm, label, policy in (("incumbent", "Incumbent", "incumbent_native"),
                                   ("frozen_B", "Frozen B", None),
                                   ("trained_C", "Trained C", "trained_native"),
                                   ("trained_C", "Trained C, incumbent selection", "trained_primary_incumbent_selection")):
            m = r["probability_metrics"][arm]
            p = r["policies"][policy] if policy else r["policies"]["frozen_B_incumbent_selection"]
            lines.append(f"| {name} | {label} | {m['five_class_logloss']:.4f} | {m['read_auc']:.4f} | {m['read_logloss']:.4f} | {m['read_brier']:.4f} | {m['read_ece_10_bins']:.4f} | {p['H']:.4f} | {p['Q']:.4f} | {p['R']:.4f} | {p['TP']}/{p['FP']}/{p['FN']}/{p['TN']} | {p['missed_label4']}/{p['missed_label5']} |")
    lines += ["", "## Incumbent → trained primary", "",
        "| Partition | Incumbent H/Q/R | Trained primary H/Q/R | ΔH | TP/FP/FN/TN | Missed 4/5 | CE 5 nhãn C | Read AUC C | Brier C | Invariant |", "|---|---|---|---:|---|---|---:|---:|---:|---|"]
    for name in PARTITIONS:
        r = parts[name]
        cm = r["probability_metrics"]["trained_C"]
        incumbent = r["policies"]["incumbent_native"]
        primary = r["policies"]["trained_primary_incumbent_selection"]
        lines.append(f"| {name} | {incumbent['H']:.4f}/{incumbent['Q']:.4f}/{incumbent['R']:.4f} | {primary['H']:.4f}/{primary['Q']:.4f}/{primary['R']:.4f} | {r['primary_delta_H']:+.4f} | {primary['TP']}/{primary['FP']}/{primary['FN']}/{primary['TN']} | {primary['missed_label4']}/{primary['missed_label5']} | {cm['five_class_logloss']:.4f} | {cm['read_auc']:.4f} | {cm['read_brier']:.4f} | {r['selection_invariant'] and r['count_invariants']} |")
    lines += ["", "### Bốn policy và bottleneck", "",
        "`selection-only` dùng selection C và ranking incumbent; `ranking-only` giữ nguyên selection incumbent và dùng ranking C; `matched-budget` chọn đúng số lượng incumbent cao nhất theo read score C rồi xếp expected gain C. Các bảng H/Q/R cùng fold nằm trong `policy_metrics.json`. CE/Brier mô tả confidence; read AUC mô tả discrimination; khác biệt selection-only/ranking-only/matched-budget cho thấy threshold/selection hay ranking đang chi phối. Đây là chẩn đoán, không phải bằng chứng calibration đã được sửa.", "",
        "| Partition | Policy | H | Q | R | TP/FP/FN/TN | Missed 4/5 |", "|---|---|---:|---:|---:|---|---|"]
    for name in PARTITIONS:
        for arm in ("incumbent_native", "trained_native", "selection_only", "ranking_only", "matched_budget", "frozen_B_incumbent_selection"):
            m = parts[name]["policies"][arm]
            lines.append(f"| {name} | {arm} | {m['H']:.4f} | {m['Q']:.4f} | {m['R']:.4f} | {m['TP']}/{m['FP']}/{m['FN']}/{m['TN']} | {m['missed_label4']}/{m['missed_label5']} |")
    lines += ["", "### Ranking shifts khi giữ selection incumbent", "",
        "Rank-change âm nghĩa là C xếp paper cao hơn. Classwise Q contribution phân rã thay đổi Q theo true label; tổng delta theo năm nhãn bằng ΔQ ranking-only (sai số làm tròn có thể xuất hiện).", ""]
    for name in PARTITIONS:
        diag = parts[name]["ranking_only_diagnostics"]
        lines += [f"**{name}:** selected {diag['selected_count']}, moved {diag['moved_count']}, mean |rank shift| {diag['mean_absolute_rank_change']:.2f}, Kendall τ {diag['kendall_tau']}", "",
            "Largest 20 rank changes:", "", "| ID | Incumbent rank | Trained rank | Δrank |", "|---|---:|---:|---:|"]
        for row in diag["largest_rank_moves"]:
            lines.append(f"| {row['id']} | {row['old_rank']} | {row['trained_rank']} | {row['rank_change']:+d} |")
        lines += ["", "Classwise Q contribution:", "", "| True label | N selected | Incumbent Q contribution | Trained Q contribution | ΔQ contribution |", "|---:|---:|---:|---:|---:|"]
        for row in diag["classwise_Q_contribution"]:
            lines.append(f"| {row['true_label']} | {row['count_selected']} | {row['incumbent_Q_contribution']:.5f} | {row['trained_Q_contribution']:.5f} | {row['delta_Q_contribution']:+.5f} |")
        lines.append("")
    lines += ["## Bằng chứng học preference", "",
        f"Trained C so frozen B: mean Δ five-class CE {decision['frozen_to_trained']['five_class_logloss']['mean_delta']:+.4f}, read AUC {decision['frozen_to_trained']['read_auc']['mean_delta']:+.4f}, read CE {decision['frozen_to_trained']['read_logloss']['mean_delta']:+.4f}, Brier {decision['frozen_to_trained']['read_brier']['mean_delta']:+.4f}, ECE10 {decision['frozen_to_trained']['read_ece_10_bins']['mean_delta']:+.4f}. Theo tiêu chí mô tả (5-class CE giảm ít nhất .01 và read AUC không giảm), evidence hữu ích: **{decision['frozen_to_trained']['useful_preference_learning_evidence']}**. Read CE/Brier/ECE phản ánh probability quality; AUC phản ánh ranking sở thích đọc. Hai split exploratory không chứng minh khái quát ra dữ liệu khác.", "",
        "## Argmax audit", "",
        "Confusion matrix và precision/recall mỗi nhãn là audit argmax năm lớp; argmax không tạo priority cuộc thi. Chi tiết per partition nằm trong `confusion_matrices.json` và `class_precision_recall.json`.", "",
        "### Confusion matrix trained C (true label × predicted label)", ""]
    for name in PARTITIONS:
        matrix = parts[name]["confusion_matrices_argmax_diagnostic"]["trained_C_native"]
        lines += [f"**{name}**", "", "| True \\ Pred | 1 | 2 | 3 | 4 | 5 |", "|---|---:|---:|---:|---:|---:|"]
        for label, row in enumerate(matrix, start=1):
            lines.append(f"| {label} | " + " | ".join(map(str, row)) + " |")
        lines += ["", "Precision/recall theo nhãn:", "", "| Nhãn | Precision | Recall | Support |", "|---:|---:|---:|---:|"]
        for row in parts[name]["class_precision_recall_argmax_diagnostic"]["trained_C_native"]:
            lines.append(f"| {row['label']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['support']} |")
        lines.append("")
    lines += ["### Nhãn được thêm/bỏ theo selection trained-native", "",
        "Các mục đếm paper mỗi true label mà C thêm vào hoặc bỏ khỏi selection so với incumbent; primary ranking-only giữ selection incumbent nên các chuyển đổi của primary bằng 0 theo thiết kế.", ""]
    for name in PARTITIONS:
        lines += [f"**{name}**", "", "| True label | Thêm | Bỏ | Incumbent selected | Trained selected | Tổng |", "|---:|---:|---:|---:|---:|---:|"]
        for row in parts[name]["selection_transitions_by_true_label"]:
            lines.append(f"| {row['true_label']} | {row['added_by_trained_native']} | {row['dropped_by_trained_native']} | {row['incumbent_selected']} | {row['trained_selected']} | {row['total']} |")
        lines.append("")
    lines += [
        "## Gate và quyết định", "",
        f"Mean incumbent H {decision['mean_incumbent_H']:.4f}; mean trained-primary H {decision['mean_primary_H']:.4f}; ΔH {decision['gate']['mean_primary_delta_H']:+.4f}. Gate: **{'PASS' if decision['gate']['passes'] else 'FAIL'}** (mean ΔH ≥ {cfg['gate']['mean_H_improvement_min']:.3f}, mỗi partition ΔH ≥ 0, selection/count invariants và technical checks).", "",
        f"Primary mean Q {decision['mean_primary_Q']:.4f}, mean R {decision['mean_primary_R']:.4f}; summed TP/FP/FN/TN {decision['summed_primary_counts']['TP']}/{decision['summed_primary_counts']['FP']}/{decision['summed_primary_counts']['FN']}/{decision['summed_primary_counts']['TN']}; missed 4/5 {decision['summed_primary_counts']['missed_label4']}/{decision['summed_primary_counts']['missed_label5']}.", "",
        f"Bước được hỗ trợ: {decision['next_step']}. Không train final 2,494, không score test 596, không upload hoặc thay incumbent.", "",
        "Group bootstrap theo partition cho primary−incumbent chỉ mô tả độ bất định của các split đã xem; không biến pilot thành confirmation.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "cpu-head", "smoke", "train", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", choices=PARTITIONS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == "prepare":
        prepare(root)
        return
    if args.stage == "evaluate":
        try:
            evaluate(root)
        except Exception as exc:
            if root.exists():
                write_json(root / "evaluate_FAILED.json", {"error": f"{type(exc).__name__}: {exc}", "time_unix": time.time()})
            raise
        return
    if not args.partition:
        parser.error(f"--partition is required for {args.stage}")
    assert_runtime_lock(root)
    try:
        run_partition_stage(root, args.partition, args.stage)
    except Exception as exc:
        safe = args.stage.replace("-", "_")
        write_json(root / args.partition / f"{safe}_FAILED.json", {"error": f"{type(exc).__name__}: {exc}", "time_unix": time.time()})
        record_stage(root, f"{args.partition}.{safe}", "FAILED", {"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
