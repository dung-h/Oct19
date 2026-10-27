"""Fit, smoke, train, and evaluate the regularized Qwen five-class head."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = ROOT / "experiments/qwen_remote_v1"
OLD = ROOT / "experiments/qwen_learning_check_v1/outputs/check_v1"
sys.path.insert(0, str(REMOTE))
from common import check_probability, digest, jsonl, load_inputs, official, prompt, sha, write_json  # noqa: E402
from model_core import load_model, load_tokenizer, save_adapter  # noqa: E402


def code_hashes():
    files = ["config.json", "PLAN.md", "README.md", "run.py", "run_head.sh", "verify.py"]
    result = {f"head/{f}": sha(HERE / f) for f in files}
    for f in ("config.json", "common.py", "model_core.py", "inputs/manifest.json"):
        result[f"remote/{f}"] = sha(REMOTE / f)
    return result


def current_runtime():
    packages = ("torch", "transformers", "peft", "accelerate", "bitsandbytes",
                "numpy", "scikit-learn", "scipy", "safetensors", "huggingface-hub")
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True).strip() if torch.cuda.is_available() else None
    return {
        "code_hashes": code_hashes(),
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "python": platform.python_version(), "cuda": torch.version.cuda,
        "device": device, "driver": driver,
        "allocator": {name: os.environ.get(name) for name in
                      ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
    }


def assert_lock(root):
    lock = json.loads((root / "runtime_lock.json").read_text())
    now = current_runtime()
    if lock["runtime"] != now:
        raise RuntimeError("Source/runtime changed after prepare; retain artifacts and use a fresh namespace")


def load_feature(root):
    fp = np.load(root / "features.npz")
    return {name: fp[name] for name in fp.files}


def feature_positions(features):
    return {int(idx): pos for pos, idx in enumerate(features["indices"])}


def fit_affine_head(x, y, lam, max_iter):
    """Fit mean multiclass CE + lam/2 ||W||^2 with a value/gradient pair."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64) - 1
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid standardized training features")
    if set(y.tolist()) != set(range(5)):
        raise ValueError("Each fitted head must see all five classes")
    counts = np.bincount(y, minlength=5).astype(float)
    bias = np.log((counts + 1.0) / (len(y) + 5.0))
    d = x.shape[1]
    initial = np.concatenate([np.zeros(5 * d), bias])

    def objective(theta):
        w = theta[:5*d].reshape(5, d)
        b = theta[5*d:]
        logits = x @ w.T + b
        probs = softmax(logits, axis=1)
        loss = (logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]).mean()
        loss += 0.5 * lam * np.square(w).sum()
        probs[np.arange(len(y)), y] -= 1.0
        probs /= len(y)
        gw = probs.T @ x + lam * w
        gb = probs.sum(axis=0)
        grad = np.concatenate([gw.ravel(), gb])
        return float(loss), grad

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40})
    value, grad = objective(result.x)
    if not result.success or not np.isfinite(value) or not np.isfinite(grad).all():
        raise RuntimeError(f"L-BFGS failed at lambda={lam}: {result.message}; loss={value}")
    return result.x[:5*d].reshape(5, d).astype(np.float32), result.x[5*d:].astype(np.float32), {
        "success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "objective": value,
        "gradient_inf_norm": float(np.abs(grad).max()),
    }


def probabilities(x, scaler, weight, bias):
    scaled = scaler.transform(np.asarray(x, dtype=np.float64)).astype(np.float64)
    z = scaled @ weight.astype(np.float64).T + bias.astype(np.float64)
    p = softmax(z, axis=1)
    return check_probability(p, len(p))


def cv_select(features, targets, docs, train_ids, cfg):
    pos = feature_positions(features)
    ids = np.asarray(train_ids, dtype=int)
    groups = np.asarray([docs[int(i)]["group"] for i in ids])
    y = np.asarray([targets[int(i)] for i in ids], dtype=int)
    if set(y) != {1, 2, 3, 4, 5}:
        raise ValueError("Train split must contain all five ratings")
    folds = list(GroupKFold(n_splits=cfg["cv_folds"]).split(ids, y, groups))
    oof = {float(lam): np.zeros((len(ids), 5), dtype=np.float64) for lam in cfg["lambda_grid"]}
    fold_records = []
    for fold_no, (tr, va) in enumerate(folds, start=1):
        tr_ids, va_ids = ids[tr], ids[va]
        if {docs[int(i)]["group"] for i in tr_ids} & {docs[int(i)]["group"] for i in va_ids}:
            raise AssertionError("Group leakage in penalty CV")
        ytr = np.asarray([targets[int(i)] for i in tr_ids])
        if set(ytr) != {1, 2, 3, 4, 5}:
            raise ValueError(f"CV training fold {fold_no} lacks a rating class")
        xtr = features["hidden"][[pos[int(i)] for i in tr_ids]]
        xva = features["hidden"][[pos[int(i)] for i in va_ids]]
        scaler = StandardScaler().fit(xtr)
        xtr_s = scaler.transform(xtr).astype(np.float64)
        xva_s = scaler.transform(xva).astype(np.float64)
        for lam in oof:
            w, b, opt = fit_affine_head(xtr_s, ytr, lam, cfg["max_iterations"])
            oof[lam][va] = softmax(xva_s @ w.T + b, axis=1)
            fold_records.append({"fold": fold_no, "lambda": lam, "train_n": len(tr),
                                 "validation_n": len(va), "optimizer": opt})
    rows = []
    for lam, p in oof.items():
        ce = float(log_loss(y, p, labels=[1, 2, 3, 4, 5]))
        rows.append({"lambda": lam, "oof_logloss": ce})
    rows.sort(key=lambda r: (r["oof_logloss"], -r["lambda"]))
    best = min(r["oof_logloss"] for r in rows)
    selected = max(r["lambda"] for r in rows if r["oof_logloss"] <= best + 1e-6)
    return selected, oof[selected], rows, fold_records


def save_head(path, scaler, weight, bias, lam, input_dim):
    joblib.dump({"mean": np.asarray(scaler.mean_, dtype=np.float32),
                 "scale": np.asarray(scaler.scale_, dtype=np.float32),
                 "weight": np.asarray(weight, dtype=np.float32),
                 "bias": np.asarray(bias, dtype=np.float32),
                 "lambda": float(lam), "class_order": [1, 2, 3, 4, 5],
                 "input_dim": int(input_dim)}, path)


def prepare(root, cfg, docs, targets, parts):
    if root.exists():
        raise RuntimeError("Output namespace already exists; keep it and choose a fresh output")
    verification = json.loads((OLD / "postrun_verification.json").read_text())
    prior_runtime = json.loads((OLD / "runtime_lock.json").read_text())
    if verification.get("status") != "ARTIFACTS_VERIFIED" or not verification.get("source_and_code_unchanged"):
        raise RuntimeError("Prior learning-check artifacts lack successful source verification")
    old_source = {k: sha(REMOTE / k) for k in prior_runtime["source"]}
    if old_source != prior_runtime["source"]:
        raise RuntimeError("Qwen source used by old cached predictions has changed")
    old_code = {k: sha(ROOT / "experiments/qwen_learning_check_v1" / k) for k in prior_runtime["code"]}
    if old_code != prior_runtime["code"]:
        raise RuntimeError("Learning-check source/config changed since its predictions were produced")
    now = current_runtime()
    if any(now["versions"].get(key) != value for key, value in prior_runtime["versions"].items()):
        raise RuntimeError("Runtime stack differs from the verified cached experiment")
    for key in ("python", "cuda", "device", "driver", "allocator"):
        if now[key] != prior_runtime.get(key):
            raise RuntimeError(f"Runtime field {key} differs from the verified cached experiment")
    if sha(OLD / "features.npz") != json.loads((OLD / "EXTRACT_COMPLETE.json").read_text())["features_sha"]:
        raise RuntimeError("Frozen-feature cache hash mismatch")
    old_complete = json.loads((OLD / "training/COMPLETE.json").read_text())
    if old_complete.get("status") != "TRAINING_COMPLETE" or old_complete.get("steps") != 225:
        raise RuntimeError("Old digit-LoRA epoch-3 cache is incomplete")
    old_adapter = OLD / "training/epoch3_adapter"
    for filename, expected in old_complete["checkpoint_hashes"]["3"].items():
        if sha(old_adapter / filename) != expected:
            raise RuntimeError(f"Cached digit-LoRA checkpoint hash mismatch: {filename}")
    members = json.loads((OLD / "membership.json").read_text())
    pool = parts[cfg["pool_partition"]]["fit"]
    if members["pool"] != pool or len(members["train"]) != 1196 or len(members["validation"]) != 298:
        raise RuntimeError("Prior split membership differs from locked plan")
    train, val = set(members["train"]), set(members["validation"])
    if train & val or train | val != set(pool):
        raise RuntimeError("Bad train/validation membership")
    if {docs[i]["group"] for i in train} & {docs[i]["group"] for i in val}:
        raise RuntimeError("Publication group crosses train and validation")
    if train & set(parts[cfg["pool_partition"]]["pred"]):
        raise RuntimeError("Old held-out documents entered diagnostic split")
    if {targets[i] for i in train} != {1, 2, 3, 4, 5}:
        raise RuntimeError("Train membership lacks a rating class")
    cached_features = load_feature(OLD)
    if cached_features["hidden"].shape != (1494, 3584) or cached_features["logits"].shape != (1494, 5):
        raise RuntimeError("Frozen feature cache has an unexpected shape")
    if len(set(int(i) for i in cached_features["indices"])) != len(cached_features["indices"]):
        raise RuntimeError("Frozen feature cache has duplicate row indices")
    if [int(i) for i in members["pool"]] != [int(i) for i in cached_features["indices"]]:
        raise RuntimeError("Feature cache row order differs from membership")
    for name in ("control_predictions.npz", "training/epoch3_predictions.npz"):
        if not (OLD / name).is_file():
            raise FileNotFoundError(OLD / name)
    checks = {
        "membership.json": sha(OLD / "membership.json"),
        "features.npz": sha(OLD / "features.npz"),
        "control_predictions.npz": sha(OLD / "control_predictions.npz"),
        "A_epoch3_predictions.npz": sha(OLD / "training/epoch3_predictions.npz"),
        "runtime_lock.json": sha(OLD / "runtime_lock.json"),
        "postrun_verification.json": sha(OLD / "postrun_verification.json"),
    }
    source_names = {"A_epoch3_predictions.npz": "training/epoch3_predictions.npz"}
    root.mkdir(parents=True)
    for name in checks:
        source = OLD / source_names.get(name, name)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    old_a = np.load(root / "A_epoch3_predictions.npz")
    controls = np.load(root / "control_predictions.npz")
    check_probability(old_a["validation"], len(members["validation"]))
    check_probability(old_a["train_probes"], len(members["train_probes"]))
    check_probability(controls["tfidf_validation"], len(members["validation"]))
    write_json(root / "source_cache_lock.json", {"old_cache_path": str(OLD), "files": checks,
        "old_code": prior_runtime["code"], "old_source": prior_runtime["source"],
        "digit_epoch3_reused": True, "digit_epoch3_checkpoint_hashes": old_complete["checkpoint_hashes"]["3"]})
    write_json(root / "membership.json", members)
    write_json(root / "runtime_lock.json", {"runtime": current_runtime(), "config_sha256": sha(HERE / "config.json"),
                                              "membership_sha256": sha(root / "membership.json")})
    print(json.dumps({"status": "PREPARED", "train": len(train), "validation": len(val),
                      "features": load_feature(root)["hidden"].shape, "digit_epoch3_reused": True}), flush=True)


def cpu_head(root, cfg, docs, targets, members):
    if (root / "HEAD_COMPLETE.json").exists():
        raise RuntimeError("CPU head stage already complete")
    features = load_feature(root)
    pos = feature_positions(features)
    selected, oof_selected, grid, folds = cv_select(features, targets, docs, members["train"], cfg)
    train_ids = members["train"]
    y_train = np.asarray([targets[i] for i in train_ids])
    x_train = features["hidden"][[pos[i] for i in train_ids]]
    scaler = StandardScaler().fit(x_train)
    weight, bias, opt = fit_affine_head(scaler.transform(x_train), y_train, selected, cfg["max_iterations"])
    save_head(root / "head_B.joblib", scaler, weight, bias, selected, x_train.shape[1])
    y_oof = np.zeros((len(train_ids), 5), dtype=np.float64)
    y_oof[:] = oof_selected
    np.savez_compressed(root / "head_B_oof.npz", train_indices=np.asarray(train_ids), probabilities=y_oof)
    np.savetxt(root / "lambda_cv.csv", np.asarray([[r["lambda"], r["oof_logloss"]] for r in grid]),
               delimiter=",", header="lambda,oof_logloss", comments="")
    write_json(root / "lambda_cv_details.json", {"selected_lambda": selected, "selection_rule": "minimum pooled OOF CE; within 1e-6 choose stronger penalty",
        "grid": grid, "folds": folds, "refit": opt, "train_class_counts": np.bincount(y_train, minlength=6)[1:].tolist()})
    pred_indices = members["validation"] + members["train_probes"]
    x_pred = features["hidden"][[pos[i] for i in pred_indices]]
    pred = probabilities(x_pred, scaler, weight, bias)
    nval = len(members["validation"])
    np.savez_compressed(root / "head_B_predictions.npz", validation=pred[:nval], train_probes=pred[nval:])
    write_json(root / "HEAD_COMPLETE.json", {"status": "HEAD_COMPLETE", "selected_lambda": selected,
        "train_only_fit": True, "validation_labels_used": False, "class_order": [1, 2, 3, 4, 5],
        "head_parameters": int(weight.size + bias.size), "oof_rows": len(y_train)})
    print(json.dumps({"status": "HEAD_COMPLETE", "selected_lambda": selected,
                      "cv": grid, "head_parameters": int(weight.size + bias.size)}, allow_nan=False), flush=True)


def attach_hidden_hook(model):
    captured = {}
    lm_head = model.get_output_embeddings()
    def capture(module, args):
        captured["hidden"] = args[0][:, -1:, :]
    handle = lm_head.register_forward_pre_hook(capture)
    return captured, handle


def hidden_and_digit_logits(model, ids, answer, captured):
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    h = captured.pop("hidden")[:, -1, :]
    if h.ndim != 2:
        raise RuntimeError(f"Expected final hidden shape [batch, dim], got {tuple(h.shape)}")
    z = out.logits[:, -1, answer].float()
    return h, z


def make_head_state(head_data):
    return {"weight": torch.as_tensor(head_data["weight"], dtype=torch.float32).clone(),
            "bias": torch.as_tensor(head_data["bias"], dtype=torch.float32).clone(),
            "mean": torch.as_tensor(head_data["mean"], dtype=torch.float32).clone(),
            "scale": torch.as_tensor(head_data["scale"], dtype=torch.float32).clone(),
            "lambda": float(head_data["lambda"]), "class_order": [1, 2, 3, 4, 5]}


def head_logits(hidden, state):
    x = (hidden.float() - state["mean"]) / state["scale"]
    return F.linear(x, state["weight"], state["bias"])


def prob_from_head(hidden, state):
    return torch.softmax(head_logits(hidden, state), dim=-1)


def tensor_state_to_cpu(state):
    return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in state.items()}


def smoke(root, cfg, base, docs, targets, members):
    if (root / "smoke").exists():
        raise RuntimeError("Smoke namespace already exists")
    if not (root / "HEAD_COMPLETE.json").exists():
        raise RuntimeError("CPU head must finish before GPU smoke")
    dest = root / "smoke"
    dest.mkdir()
    tokenizer, answer = load_tokenizer(base)
    if len(answer) != 5:
        raise RuntimeError("Five rating token verification failed")
    features = load_feature(root); pos = feature_positions(features)
    anchor_ids = []
    for label in range(1, 6):
        anchor_ids.append(next(i for i in members["train"] if targets[i] == label))
    head_data = joblib.load(root / "head_B.joblib")
    state = make_head_state(head_data)
    tokens = {i: prompt(tokenizer, docs[i], base)[0] for i in anchor_ids}
    model, lora_names = load_model(base, trainable=True, seed=cfg["seed"])
    capture, hook = attach_hidden_hook(model)
    hidden_err = 0.0; frozen_prob_err = 0.0
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in anchor_ids:
            ids = torch.tensor([tokens[i]], device="cuda", dtype=torch.long)
            h, z = hidden_and_digit_logits(model, ids, answer, capture)
            cached_h = torch.as_tensor(features["hidden"][pos[i]], device="cuda")
            hidden_err = max(hidden_err, float((h[0].float() - cached_h.float()).abs().max()))
            old_controls = np.load(root / "control_predictions.npz")
            oldp = old_controls["frozen_digit_train_probes"]
            if i in members["train_probes"]:
                old_row = members["train_probes"].index(i)
                expected_digit = oldp[old_row]
                frozen_prob_err = max(frozen_prob_err, float(np.abs(z.softmax(-1).cpu().numpy()[0] - expected_digit).max()))
    if hidden_err > 1e-4:
        hook.remove(); raise RuntimeError(f"Frozen hidden feature parity failed: {hidden_err}")
    # Digit parity is checked for anchors that are in the fixed train-probe cache.
    model.train()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not lora_names or any("lora_" not in n for n, _ in trainable):
        hook.remove(); raise RuntimeError("Unexpected trainable base parameters")
    before_lora = {n: p.detach().clone() for n, p in trainable}
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(cfg["seed"] + 900)
        head = torch.nn.Linear(state["weight"].shape[1], 5, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(state["weight"].to("cuda")); head.bias.copy_(state["bias"].to("cuda"))
    before_head = {k: v.detach().clone() for k, v in state.items() if torch.is_tensor(v)}
    opt = torch.optim.AdamW([{"params": [p for _, p in trainable], "lr": cfg["training"]["learning_rate"], "weight_decay": cfg["training"]["weight_decay"]},
                             {"params": list(head.parameters()), "lr": cfg["training"]["learning_rate"], "weight_decay": 0.0}])
    logs = []
    model.train(); head.train()
    for step in range(2):
        batch = anchor_ids
        opt.zero_grad(set_to_none=True)
        ce_values = []
        for i in batch:
            ids = torch.tensor([tokens[i]], device="cuda", dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h, _ = hidden_and_digit_logits(model, ids, answer, capture)
            z = head((h.float() - state["mean"].to("cuda")) / state["scale"].to("cuda"))
            ce = F.cross_entropy(z, torch.tensor([targets[i] - 1], device="cuda"))
            penalty = 0.5 * state["lambda"] * head.weight.square().sum()
            loss = (ce + penalty) / len(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Smoke loss is not finite")
            loss.backward(); ce_values.append(float(ce.detach()))
        if any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n):
            raise RuntimeError("Frozen base received gradients")
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable):
            raise RuntimeError("LoRA gradients missing/non-finite")
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
            raise RuntimeError("Head gradients missing/non-finite")
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable] + list(head.parameters()), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Smoke grad norm not finite")
        opt.step()
        logs.append({"step": step + 1, "ce": float(np.mean(ce_values)), "grad_norm": float(norm),
                     "head_l2": float(head.weight.norm()), "peak_allocated_GiB": torch.cuda.max_memory_allocated()/2**30})
        torch.cuda.empty_cache()
    if not any(not torch.equal(before_lora[n], p.detach()) for n, p in trainable):
        raise RuntimeError("LoRA did not update during smoke")
    if torch.equal(before_head["weight"].to("cuda"), head.weight.detach()) and torch.equal(before_head["bias"].to("cuda"), head.bias.detach()):
        raise RuntimeError("Output head did not update during smoke")
    save_adapter(model, dest / "adapter")
    smoke_state = {"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(),
                   "mean": state["mean"].cpu(), "scale": state["scale"].cpu(),
                   "lambda": state["lambda"], "class_order": [1, 2, 3, 4, 5]}
    torch.save(smoke_state, dest / "head.pt")
    model.eval(); head.eval()
    pre = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in anchor_ids:
            h, _ = hidden_and_digit_logits(model, torch.tensor([tokens[i]], device="cuda"), answer, capture)
            pre.append(prob_from_head(h, {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in smoke_state.items()}).cpu().numpy()[0])
    del model, head, opt; gc.collect(); torch.cuda.empty_cache()
    model2, _ = load_model(base, trainable=True, seed=cfg["seed"], adapter_path=dest / "adapter")
    cap2, hook2 = attach_hidden_hook(model2); model2.eval()
    state2 = torch.load(dest / "head.pt", map_location="cpu", weights_only=False)
    post = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in anchor_ids:
            h, _ = hidden_and_digit_logits(model2, torch.tensor([tokens[i]], device="cuda"), answer, cap2)
            local = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in state2.items()}
            post.append(prob_from_head(h, local).cpu().numpy()[0])
    reload_error = float(np.max(np.abs(np.asarray(pre) - np.asarray(post))))
    hook.remove(); hook2.remove()
    if reload_error > 1e-4:
        raise RuntimeError(f"Adapter+head reload parity failed: {reload_error}")
    write_json(dest / "COMPLETE.json", {"status": "TECHNICAL_PASS", "updates": 2,
        "lora_changed": True, "head_changed": True, "base_gradients_absent": True,
        "hidden_feature_max_error": hidden_err, "reload_probability_max_error": reload_error,
        "smoke_updates": logs, "frozen_digit_probe_max_error": frozen_prob_err,
        "frozen_digit_probe_count": sum(i in members["train_probes"] for i in anchor_ids),
        "token_ids": answer, "lora_parameter_count": sum(p.numel() for _, p in trainable),
        "peak_allocated_GiB": max(r["peak_allocated_GiB"] for r in logs)})
    (dest / "training.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in logs))
    print(json.dumps(json.loads((dest / "COMPLETE.json").read_text()), allow_nan=False), flush=True)


def score_indices(model, captured, tokens, answer, indices, state, batch_context="cuda"):
    values = []
    model.eval()
    with torch.inference_mode(), torch.autocast(batch_context, dtype=torch.bfloat16):
        for idx in indices:
            ids = torch.tensor([tokens[idx]], device="cuda", dtype=torch.long)
            h, _ = hidden_and_digit_logits(model, ids, answer, captured)
            local = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in state.items()}
            values.append(prob_from_head(h, local).squeeze(0).cpu().numpy().astype(np.float64))
            torch.cuda.empty_cache()
    return check_probability(values, len(indices))


def train(root, cfg, base, docs, members):
    dest = root / "training"
    if dest.exists():
        raise RuntimeError("Training namespace exists; optimizer resume is unsupported")
    if json.loads((root / "smoke/COMPLETE.json").read_text()).get("status") != "TECHNICAL_PASS":
        raise RuntimeError("A passing fresh smoke is required")
    if json.loads((root / "HEAD_COMPLETE.json").read_text()).get("status") != "HEAD_COMPLETE":
        raise RuntimeError("CPU head stage is incomplete")
    dest.mkdir()
    tokenizer, answer = load_tokenizer(base)
    indices = list(dict.fromkeys(members["train"] + members["validation"] + members["train_probes"]))
    tokens = {i: prompt(tokenizer, docs[i], base)[0] for i in indices}
    write_json(dest / "token_audit.json", {str(i): digest(tokens[i]) for i in indices})
    head_data = joblib.load(root / "head_B.joblib")
    state = make_head_state(head_data)
    # load_model sets the exact seed before initializing the LoRA matrices.
    load_start = time.monotonic()
    model, lora_names = load_model(base, trainable=True, seed=cfg["seed"])
    model_load_seconds = time.monotonic() - load_start
    captured, hook = attach_hidden_hook(model)
    features = load_feature(root); pos = feature_positions(features)
    parity_ids = [next(i for i in members["train"] if i % 2 == 0)]
    feature_error = 0.0
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx in parity_ids:
            h, _ = hidden_and_digit_logits(model, torch.tensor([tokens[idx]], device="cuda"), answer, captured)
            cached = torch.as_tensor(features["hidden"][pos[idx]], device="cuda")
            feature_error = max(feature_error, float((h[0].float() - cached.float()).abs().max()))
    if feature_error > 1e-4:
        hook.remove(); raise RuntimeError(f"Pre-fit hidden parity failed: {feature_error}")
    model.train()
    lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not lora_names or any("lora_" not in n for n, _ in lora):
        hook.remove(); raise RuntimeError("Unexpected trainable parameters")
    # Avoid consuming the LoRA/dropout RNG stream while making the head module.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(cfg["seed"] + 900)
        head = torch.nn.Linear(state["weight"].shape[1], 5, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(state["weight"].to("cuda")); head.bias.copy_(state["bias"].to("cuda"))
    scaler_mean = state["mean"].to("cuda"); scaler_scale = state["scale"].to("cuda")
    opt = torch.optim.AdamW([
        {"params": [p for _, p in lora], "lr": cfg["training"]["learning_rate"], "weight_decay": cfg["training"]["weight_decay"]},
        {"params": list(head.parameters()), "lr": cfg["training"]["learning_rate"], "weight_decay": 0.0},
    ])
    accumulation = cfg["training"]["gradient_accumulation_steps"]
    order = np.random.default_rng(cfg["seed"] + 1).permutation(members["train"]).tolist()
    steps_per_epoch = math.ceil(len(order) / accumulation)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(opt, math.ceil(total_steps * cfg["training"]["warmup_ratio"]), total_steps)
    training_jsonl = (dest / "training.jsonl").open("x")
    before = {n: p.detach().clone() for n, p in lora}
    started = time.monotonic(); train_epoch_seconds = []; score_seconds = 0.0; checkpoint_hashes = {}
    all_labels = {r["index"]: r["rating"] for r in jsonl(REMOTE / "inputs/targets.jsonl")}
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            order = np.random.default_rng(cfg["seed"] + epoch).permutation(members["train"]).tolist()
            epoch_start = time.monotonic(); model.train(); head.train()
            for batch_start in range(0, len(order), accumulation):
                batch = order[batch_start:batch_start + accumulation]
                update_start = time.monotonic(); opt.zero_grad(set_to_none=True); ces = []
                for idx in batch:
                    ids = torch.tensor([tokens[idx]], device="cuda", dtype=torch.long)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h, _ = hidden_and_digit_logits(model, ids, answer, captured)
                    x = (h.float() - scaler_mean) / scaler_scale
                    z = head(x)
                    ce = F.cross_entropy(z, torch.tensor([all_labels[idx] - 1], device="cuda"))
                    penalty = 0.5 * state["lambda"] * head.weight.square().sum()
                    loss = (ce + penalty) / len(batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Joint-head CE/penalty is not finite")
                    loss.backward(); ces.append(float(ce.detach()))
                if any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n):
                    raise RuntimeError("Frozen base received gradients")
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in lora):
                    raise RuntimeError("Missing/non-finite LoRA gradient")
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise RuntimeError("Missing/non-finite head gradient")
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in lora] + list(head.parameters()), cfg["training"]["max_grad_norm"])
                if not torch.isfinite(norm):
                    raise RuntimeError("Joint-head gradient norm is not finite")
                opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
                step = (epoch - 1) * steps_per_epoch + batch_start // accumulation + 1
                row = {"epoch": epoch, "step": step, "total_steps": total_steps,
                    "cross_entropy": float(np.mean(ces)), "penalty": float(0.5 * state["lambda"] * head.weight.square().sum().detach()),
                    "objective": float(np.mean(ces) + 0.5 * state["lambda"] * head.weight.square().sum().detach()),
                    "grad_norm": float(norm), "lr_lora": scheduler.get_last_lr()[0],
                    "lr_head": scheduler.get_last_lr()[1], "seconds": time.monotonic()-update_start,
                    "peak_allocated_GiB": torch.cuda.max_memory_allocated()/2**30,
                    "head_weight_norm": float(head.weight.norm().detach())}
                training_jsonl.write(json.dumps(row, allow_nan=False) + "\n"); training_jsonl.flush()
                if step % 10 == 0 or step == total_steps:
                    print(json.dumps(row, allow_nan=False), flush=True)
                torch.cuda.empty_cache()
            train_epoch_seconds.append(time.monotonic() - epoch_start)
            adapter_dir = dest / f"epoch{epoch}_adapter"; save_adapter(model, adapter_dir)
            head_state = {"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(),
                "mean": state["mean"].cpu(), "scale": state["scale"].cpu(), "lambda": state["lambda"],
                "class_order": [1, 2, 3, 4, 5]}
            head_file = dest / f"epoch{epoch}_head.pt"; torch.save(head_state, head_file)
            checkpoint_hashes[str(epoch)] = {p.name: sha(p) for p in adapter_dir.iterdir() if p.is_file()}
            checkpoint_hashes[str(epoch)][head_file.name] = sha(head_file)
            t_score = time.monotonic()
            val_p = score_indices(model, captured, tokens, answer, members["validation"], head_state)
            probe_p = score_indices(model, captured, tokens, answer, members["train_probes"], head_state)
            score_seconds += time.monotonic() - t_score
            np.savez_compressed(dest / f"epoch{epoch}_predictions.npz", validation=val_p, train_probes=probe_p)
            print(f"epoch {epoch}/{cfg['epochs']} scored; validation metrics remain locked", flush=True)
        if not any(not torch.equal(before[n], p.detach()) for n, p in lora):
            raise RuntimeError("LoRA did not change")
        pre = score_indices(model, captured, tokens, answer, parity_ids, head_state)
        del model, head, opt, scheduler; gc.collect(); torch.cuda.empty_cache()
        reload_start = time.monotonic()
        reloaded, _ = load_model(base, trainable=True, seed=cfg["seed"], adapter_path=dest / f"epoch{cfg['epochs']}_adapter")
        reload_seconds = time.monotonic() - reload_start
        cap2, hook2 = attach_hidden_hook(reloaded); reloaded.eval()
        reload_head = torch.load(dest / f"epoch{cfg['epochs']}_head.pt", map_location="cpu", weights_only=False)
        post = score_indices(reloaded, cap2, tokens, answer, parity_ids, reload_head)
        reload_error = float(np.max(np.abs(pre - post)))
        hook2.remove()
        if reload_error > 1e-4:
            raise RuntimeError(f"Joint adapter/head reload parity failed: {reload_error}")
        total_seconds = time.monotonic() - started
        write_json(dest / "COMPLETE.json", {"status": "TRAINING_COMPLETE", "epochs": cfg["epochs"],
            "steps": total_steps, "selected_lambda": state["lambda"], "feature_parity_max_error": feature_error,
            "reload_probability_max_error": reload_error, "adapter_changed": True,
            "base_gradients_absent": True, "model_loading_seconds": model_load_seconds,
            "training_seconds_by_epoch": train_epoch_seconds, "scoring_seconds": score_seconds,
            "reloading_seconds": reload_seconds, "total_seconds": total_seconds,
            "checkpoint_hashes": checkpoint_hashes, "max_peak_allocated_GiB": float(torch.cuda.max_memory_allocated()/2**30)})
        print(json.dumps(json.loads((dest / "COMPLETE.json").read_text()), allow_nan=False), flush=True)
    finally:
        training_jsonl.close()
        hook.remove()


def policy_priority(selected, rank_gain):
    priority = np.zeros(len(selected), dtype=float)
    priority[selected] = 1.0 + 9.0 * np.asarray(rank_gain)[selected]
    return priority


def native_priority(p, cutoff):
    read = 1.0 - p[:, 0]
    gain = p @ np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    return policy_priority(read > cutoff, gain)


def policy_table(name, p, anchor, y, ids, cutoff):
    read = 1.0 - p[:, 0]
    gain = p @ np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    anchor_read = 1.0 - anchor[:, 0]
    anchor_gain = anchor @ np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    selected = read > cutoff
    anchor_selected = anchor_read > cutoff
    same_budget = np.zeros(len(y), dtype=bool)
    order = np.lexsort((np.asarray(ids, dtype=str), -read))
    same_budget[order[:int(anchor_selected.sum())]] = True
    return {
        "native": native_priority(p, cutoff),
        "selection_only": policy_priority(selected, anchor_gain),
        "ranking_only": policy_priority(anchor_selected, gain),
        "matched_budget": policy_priority(same_budget, anchor_gain),
    }


def metric_row(arm, policy, y, p, priority, ids):
    m = official(y, priority, ids)
    read_true = (np.asarray(y) > 1).astype(int)
    read_p = 1.0 - np.asarray(p)[:, 0]
    ce5 = float(log_loss(y, p, labels=[1, 2, 3, 4, 5]))
    ce_read = float(log_loss(read_true, read_p, labels=[0, 1]))
    pos = read_true == 1
    cond = np.asarray(p)[pos, 1:] / read_p[pos, None]
    ce_cond = float(log_loss(np.asarray(y)[pos], cond, labels=[2, 3, 4, 5]))
    return {"arm": arm, "policy": policy, **m,
        "rating_logloss": ce5, "read_logloss": ce_read,
        "rating_given_read_logloss": ce_cond, "read_AUC": float(roc_auc_score(read_true, read_p)),
        "read_Brier": float(np.mean((read_p - read_true) ** 2))}


def bootstrap_h(y, priority, ids, groups, repeats, seed):
    keys = sorted(set(groups))
    members = {g: np.flatnonzero(np.asarray(groups) == g) for g in keys}
    rng = np.random.default_rng(seed); values = []
    y = np.asarray(y); priority = np.asarray(priority); ids = np.asarray(ids)
    for _ in range(repeats):
        draw = rng.choice(keys, size=len(keys), replace=True)
        ix = np.concatenate([members[g] for g in draw])
        values.append(official(y[ix], priority[ix], ids[ix])["H"])
    return {"lower_2_5": float(np.quantile(values, .025)), "upper_97_5": float(np.quantile(values, .975)),
            "mean": float(np.mean(values)), "repeats": repeats}


def paired_group_bootstrap(y, challenger_p, anchor_p, groups, ids, cutoff, repeats, seed):
    keys = sorted(set(groups))
    group_members = {g: np.flatnonzero(np.asarray(groups) == g) for g in keys}
    rng = np.random.default_rng(seed)
    deltas = {"H": [], "rating_logloss": [], "read_AUC": []}
    y = np.asarray(y); ids = np.asarray(ids)
    candidate_priority = native_priority(challenger_p, cutoff)
    anchor_priority = native_priority(anchor_p, cutoff)
    for _ in range(repeats):
        draw = rng.choice(keys, size=len(keys), replace=True)
        ix = np.concatenate([group_members[group] for group in draw])
        ys, candidate, anchor = y[ix], challenger_p[ix], anchor_p[ix]
        deltas["H"].append(official(ys, candidate_priority[ix], ids[ix])["H"] -
                           official(ys, anchor_priority[ix], ids[ix])["H"])
        deltas["rating_logloss"].append(
            log_loss(ys, candidate, labels=[1, 2, 3, 4, 5]) -
            log_loss(ys, anchor, labels=[1, 2, 3, 4, 5]))
        read_true = (ys > 1).astype(int)
        deltas["read_AUC"].append(
            roc_auc_score(read_true, 1.0 - candidate[:, 0]) -
            roc_auc_score(read_true, 1.0 - anchor[:, 0]))
    result = {name: {"lower_2_5": float(np.quantile(values, .025)),
                     "upper_97_5": float(np.quantile(values, .975)),
                     "mean": float(np.mean(values))} for name, values in deltas.items()}
    result["repeats"] = repeats
    return result


def evaluate(root, cfg, docs, targets, members):
    if (root / "decision.json").exists():
        raise RuntimeError("Evaluation already completed; preserve output and do not overwrite")
    training = root / "training"
    complete = json.loads((training / "COMPLETE.json").read_text())
    if complete.get("status") != "TRAINING_COMPLETE" or complete.get("epochs") != cfg["epochs"]:
        raise RuntimeError("All joint-head epochs and final reload check are required before evaluation")
    if not (root / "HEAD_COMPLETE.json").exists() or not (root / "smoke/COMPLETE.json").exists():
        raise RuntimeError("CPU head and GPU smoke evidence are required")
    for epoch in range(1, cfg["epochs"] + 1):
        if not (training / f"epoch{epoch}_predictions.npz").exists():
            raise FileNotFoundError(f"Missing predictions for epoch {epoch}")
    ids = members["validation"]
    y = np.asarray([targets[i] for i in ids], dtype=int)
    row_ids = np.asarray([docs[i]["id"] for i in ids], dtype=str)
    groups = [docs[i]["group"] for i in ids]
    oldp = np.load(root / "control_predictions.npz")
    olda = np.load(root / "A_epoch3_predictions.npz")
    bp = np.load(root / "head_B_predictions.npz")
    cp_by_epoch = {e: np.load(training / f"epoch{e}_predictions.npz")["validation"] for e in range(1, cfg["epochs"] + 1)}
    cprobe_by_epoch = {e: np.load(training / f"epoch{e}_predictions.npz")["train_probes"] for e in range(1, cfg["epochs"] + 1)}
    arms = {"prior": oldp["fit_prior_validation"], "tfidf": oldp["tfidf_validation"],
            "A_digit_lora_epoch3": olda["validation"], "B_frozen_head": bp["validation"],
            "C_joint_head_epoch3": cp_by_epoch[cfg["epochs"]]}
    curves = {f"C_joint_head_epoch{e}": cp for e, cp in cp_by_epoch.items()}
    arms.update(curves)
    report_rows = []
    metric_by_key = {}
    for arm_name, p in arms.items():
        check_probability(p, len(y))
        policies = policy_table(arm_name, p, olda["validation"], y, row_ids, cfg["read_cutoff"])
        for policy_name, priority in policies.items():
            row = metric_row(arm_name, policy_name, y, p, priority, row_ids)
            report_rows.append(row); metric_by_key[(arm_name, policy_name)] = (row, priority, p)
    primary = [("B_frozen_head", "native"), ("C_joint_head_epoch3", "native")]
    gates = {}
    a_row = metric_by_key[("A_digit_lora_epoch3", "native")][0]
    for name, pol in primary:
        row, priority, p = metric_by_key[(name, pol)]
        gates[name] = {"CE_delta_vs_A": row["rating_logloss"] - a_row["rating_logloss"],
            "read_AUC_delta_vs_A": row["read_AUC"] - a_row["read_AUC"],
            "H_delta_vs_A": row["H"] - a_row["H"],
            "missed4_delta_vs_A": row["missed_label4"] - a_row["missed_label4"],
            "missed5_delta_vs_A": row["missed_label5"] - a_row["missed_label5"]}
        gates[name]["passes_practical_gate"] = bool(
            gates[name]["CE_delta_vs_A"] <= -.03 and gates[name]["read_AUC_delta_vs_A"] >= .02 and
            gates[name]["H_delta_vs_A"] >= 0 and gates[name]["missed4_delta_vs_A"] <= 0 and
            gates[name]["missed5_delta_vs_A"] <= 0)
    b_row = metric_by_key[("B_frozen_head", "native")][0]
    c_row = metric_by_key[("C_joint_head_epoch3", "native")][0]
    joint_value = {"CE_delta_C_vs_B": c_row["rating_logloss"] - b_row["rating_logloss"],
                   "AUC_delta_C_vs_B": c_row["read_AUC"] - b_row["read_AUC"]}
    joint_value["joint_adds_practical_value"] = bool(joint_value["CE_delta_C_vs_B"] <= -.01 and joint_value["AUC_delta_C_vs_B"] >= 0)
    probe_ids = members["train_probes"]
    probe_y = np.asarray([targets[i] for i in probe_ids], dtype=int)
    probe_row_ids = np.asarray([docs[i]["id"] for i in probe_ids], dtype=str)
    probe_predictions = {
        "A_digit_lora_epoch3": olda["train_probes"],
        "B_frozen_head": bp["train_probes"],
        "C_joint_head_epoch1": cprobe_by_epoch[1],
        "C_joint_head_epoch2": cprobe_by_epoch[2],
        "C_joint_head_epoch3": cprobe_by_epoch[3],
    }
    probe_summary = {name: metric_row(name, "native", probe_y, p,
        native_priority(p, cfg["read_cutoff"]), probe_row_ids)
        for name, p in probe_predictions.items()}
    boot = {}
    paired_boot = {}
    for name in ("A_digit_lora_epoch3", "B_frozen_head", "C_joint_head_epoch3"):
        row, priority, p = metric_by_key[(name, "native")]
        boot[name] = bootstrap_h(y, priority, row_ids, groups, cfg["bootstrap_repeats"], cfg["seed"] + len(boot))
        if name != "A_digit_lora_epoch3":
            paired_boot[name + "_minus_A"] = paired_group_bootstrap(
                y, p, olda["validation"], groups, row_ids, cfg["read_cutoff"],
                cfg["bootstrap_repeats"], cfg["seed"] + 100 + len(paired_boot))
    confusion = {}
    class_rows = []
    for arm in ("A_digit_lora_epoch3", "B_frozen_head", "C_joint_head_epoch3"):
        p = arms[arm]; pred = p.argmax(axis=1) + 1
        cm = np.zeros((5, 5), dtype=int)
        for truth, guess in zip(y, pred): cm[truth-1, guess-1] += 1
        confusion[arm] = cm.tolist()
        for label in range(1, 6):
            tp = int(cm[label-1, label-1]); support = int((y == label).sum()); predicted = int((pred == label).sum())
            class_rows.append({"arm": arm, "label": label, "support": support, "predicted": predicted,
                               "precision": tp / predicted if predicted else 0.0,
                               "recall": tp / support if support else 0.0})
    transitions = []
    a_sel = (1.0 - olda["validation"][:, 0]) > cfg["read_cutoff"]
    for arm in ("B_frozen_head", "C_joint_head_epoch3"):
        sel = (1.0 - arms[arm][:, 0]) > cfg["read_cutoff"]
        for label in range(1, 6):
            mask = y == label
            transitions.append({"arm": arm, "true_label": label,
                "added": int((sel & ~a_sel & mask).sum()), "dropped": int((~sel & a_sel & mask).sum()),
                "selected": int((sel & mask).sum()), "total": int(mask.sum())})
    write_csv(root / "metrics.csv", report_rows)
    write_csv(root / "class_metrics.csv", class_rows)
    write_csv(root / "label_transitions.csv", transitions)
    with (root / "confusion.csv").open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["arm", "true_label", "pred1", "pred2", "pred3", "pred4", "pred5"])
        for name, matrix in confusion.items():
            for idx, row in enumerate(matrix): wr.writerow([name, idx + 1, *row])
    write_json(root / "confusion.json", confusion)
    write_json(root / "group_bootstrap.json", boot)
    decision = {"status": "EVALUATION_COMPLETE", "validation_labels_used_only_after_training": True,
        "n_validation": len(y), "validation_exploratory_previously_seen": True,
        "primary_gates_vs_A": gates, "joint_increment_vs_B": joint_value,
        "group_bootstrap_H": boot, "paired_group_bootstrap_delta_vs_A": paired_boot,
        "train_probe_fit_diagnostic": probe_summary,
        "frozen_feature_lambda": float(joblib.load(root / "head_B.joblib")["lambda"]),
        "native_A_B_C": {name: metric_by_key[(name, "native")][0] for name in
            ("A_digit_lora_epoch3", "B_frozen_head", "C_joint_head_epoch3")},
        "diagnostic_policy_anchor": "Arm A digit-head LoRA epoch3 on the exact same 298 rows; not incumbent",
        "cutoff": cfg["read_cutoff"], "class_confusion_argmax_is_diagnostic_only": True}
    write_json(root / "decision.json", decision)
    write_report(root / "REPORT.md", report_rows, gates, joint_value, boot, paired_boot,
                 probe_summary, cfg, complete)
    (root / "VALIDATION_LABELS_EVALUATED.json").write_text(json.dumps({"status": "COMPLETE", "n": len(y)}, indent=2) + "\n")
    print(json.dumps(decision, allow_nan=False), flush=True)


def write_csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_report(path, rows, gates, joint, boot, paired_boot, probe_summary, cfg, complete):
    lookup = {(r["arm"], r["policy"]): r for r in rows}
    names = ["prior", "tfidf", "A_digit_lora_epoch3", "B_frozen_head", "C_joint_head_epoch3"]
    lines = ["# Kết quả learned head Qwen", "", "Quality diagnostic exploratory trên đúng 298 bài validation đã xem trước.",
             "Không so trực tiếp với các fold pilot incumbent hoặc public leaderboard.", "",
             f"Lambda được chọn trong train-only group CV: {complete['selected_lambda']}; tổng joint updates: {complete['steps']}.", "",
             "| Arm | 5-class CE | Read AUC | Read Brier | H | Q | R | TP/FP/FN/TN | missed4/5 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for name in names:
        r = lookup[(name, "native")]
        lines.append(f"| {name} | {r['rating_logloss']:.4f} | {r['read_AUC']:.4f} | {r['read_Brier']:.4f} | {r['H']:.4f} | {r['Q']:.4f} | {r['R']:.4f} | {r['TP']}/{r['FP']}/{r['FN']}/{r['TN']} | {r['missed_label4']}/{r['missed_label5']} |")
    lines += ["", "`selection_only`, `ranking_only`, `matched_budget` lấy A làm anchor trên cùng split; xem metrics.csv. Argmax confusion chỉ audit nhãn, không tạo priority.", "",
              "| Arm | Policy | H | Q | R | FP | missed4/5 |", "|---|---|---:|---:|---:|---:|---|"]
    for name in ("A_digit_lora_epoch3", "B_frozen_head", "C_joint_head_epoch3"):
        for pol in ("native", "selection_only", "ranking_only", "matched_budget"):
            r = lookup[(name, pol)]
            lines.append(f"| {name} | {pol} | {r['H']:.4f} | {r['Q']:.4f} | {r['R']:.4f} | {r['FP']} | {r['missed_label4']}/{r['missed_label5']} |")
    lines += ["", "## Gate đã khóa", "", "| Candidate | ΔCE vs A | ΔAUC vs A | ΔH vs A | Δmissed4 | Δmissed5 | Gate |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, g in gates.items():
        lines.append(f"| {name} | {g['CE_delta_vs_A']:+.4f} | {g['read_AUC_delta_vs_A']:+.4f} | {g['H_delta_vs_A']:+.4f} | {g['missed4_delta_vs_A']:+d} | {g['missed5_delta_vs_A']:+d} | {'PASS' if g['passes_practical_gate'] else 'NO PASS'} |")
    lines += ["", "Paired group bootstrap 95% intervals for challenger−A:", "",
              "| Challenger | ΔH interval | ΔCE interval | Δread AUC interval |",
              "|---|---|---|---|"]
    for name, values in paired_boot.items():
        def interval(key):
            v = values[key]
            return f"[{v['lower_2_5']:+.4f}, {v['upper_97_5']:+.4f}]"
        lines.append(f"| {name} | {interval('H')} | {interval('rating_logloss')} | {interval('read_AUC')} |")
    lines += ["", f"Joint C−B: ΔCE {joint['CE_delta_C_vs_B']:+.4f}, Δread AUC {joint['AUC_delta_C_vs_B']:+.4f}; adds practical value = {joint['joint_adds_practical_value']}.",
              "", "Train probes are in-sample fitting diagnostics, not validation estimates:", "",
              "| Arm | Train-probe CE | Read AUC | H |", "|---|---:|---:|---:|"]
    for name, values in probe_summary.items():
        lines.append(f"| {name} | {values['rating_logloss']:.4f} | {values['read_AUC']:.4f} | {values['H']:.4f} |")
    lines += [
              "", "## Giới hạn và runtime", "", "Bootstrap publication-group của H là mô tả trên split đã xem, không biến split này thành holdout mới.",
              f"Training {complete['steps']} updates; load {complete['model_loading_seconds']:.1f}s, training từng epoch {complete['training_seconds_by_epoch']}, scoring {complete['scoring_seconds']:.1f}s, reload parity {complete['reload_probability_max_error']:.3g}; peak allocated {complete['max_peak_allocated_GiB']:.2f} GiB.",
              "Không train final 2,494, không score test 596, không submit hoặc thay incumbent.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "cpu-head", "smoke", "train", "evaluate"])
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(); root = args.output.resolve()
    cfg = json.loads((HERE / "config.json").read_text())
    base = json.loads((REMOTE / "config.json").read_text())
    docs, targets, parts, _ = load_inputs()
    if args.stage == "prepare":
        prepare(root, cfg, docs, targets, parts); return
    assert_lock(root)
    members = json.loads((root / "membership.json").read_text())
    try:
        if args.stage == "cpu-head": cpu_head(root, cfg, docs, targets, members)
        elif args.stage == "smoke": smoke(root, cfg, base, docs, targets, members)
        elif args.stage == "train": train(root, cfg, base, docs, members)
        elif args.stage == "evaluate": evaluate(root, cfg, docs, targets, members)
    except Exception as exc:
        write_json(root / f"{args.stage.replace('-', '_')}_FAILED.json",
                   {"error": f"{type(exc).__name__}: {exc}", "time": time.time()})
        raise


if __name__ == "__main__":
    main()
