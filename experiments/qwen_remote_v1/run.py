#!/usr/bin/env python3
"""Download, technical smoke, and fixed one-epoch five-class QLoRA pilot."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np

from common import HERE, check_probability, digest, fingerprint, jsonl, load_inputs, prompt, sha, write_json


def runtime(config, tag, mode):
    import torch
    packages = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy",
                "scikit-learn", "safetensors", "huggingface-hub")
    return dict(fingerprint=fingerprint(), partition=tag, mode=mode,
                versions={n: importlib.metadata.version(n) for n in packages},
                python=platform.python_version(), cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0),
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                capability=list(torch.cuda.get_device_capability()), seed=config["seed"])


def scores(model, tokens, answer, indices, docs, path, lock_sha, frozen=False):
    import torch
    from model_core import five_logits
    model.eval()
    cached = jsonl(path) if path.exists() else []
    assert [r["index"] for r in cached] == indices[:len(cached)]
    for r in cached:
        assert r["lock_sha"] == lock_sha and r["id"] == docs[r["index"]]["id"]
        assert r["token_sha"] == digest(tokens[r["index"]])
    if cached:
        check_probability([r["probabilities"] for r in cached], len(cached))
    ctx = model.disable_adapter() if frozen else contextlib.nullcontext()
    with ctx, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with path.open("a") as out:
            for index in indices[len(cached):]:
                ids = torch.tensor([tokens[index]], dtype=torch.long, device="cuda")
                logits = five_logits(model, ids, answer, train=False)
                p = logits.softmax(-1).squeeze(0).cpu().numpy().astype(float).tolist()
                check_probability([p], 1)
                row = dict(index=index, id=docs[index]["id"], lock_sha=lock_sha,
                           token_sha=digest(tokens[index]), probabilities=p)
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
                cached.append(row)
                if len(cached) % 25 == 0 or len(cached) == len(indices):
                    print(f"{path.name}: {len(cached)}/{len(indices)}", flush=True)
    return check_probability([r["probabilities"] for r in cached], len(indices))


def train(model, tokens, targets, fit, answer, config, out, smoke):
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup
    from model_core import five_logits
    cfg = config["training"]
    # Smoke is a separate technical run: two updates covering all five labels.
    if smoke:
        fit = [next(i for i in fit if targets[i] == y) for y in range(1, 6)] * 2
    else:
        fit = list(np.random.default_rng(config["seed"]).permutation(fit))
    accumulation = 5 if smoke else cfg["gradient_accumulation_steps"]
    updates = math.ceil(len(fit) / accumulation)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert parameters and all("lora_" in n for n, _ in parameters)
    before = {n: p.detach().cpu().clone() for n, p in parameters}
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(updates * cfg["warmup_ratio"]), num_training_steps=updates)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started, update = time.monotonic(), 0
    with (out / "training.jsonl").open("x") as log:
        for start in range(0, len(fit), accumulation):
            batch = fit[start:start + accumulation]
            losses = []
            for index in batch:
                ids = torch.tensor([tokens[index]], dtype=torch.long, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = five_logits(model, ids, answer, train=True)
                    loss = F.cross_entropy(logits, torch.tensor([targets[index] - 1], device="cuda"))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
            assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
            assert any(p.grad is not None for _, p in parameters)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], cfg["max_grad_norm"])
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            row = dict(update=update, updates=updates, examples=min(start+len(batch), len(fit)),
                       loss=float(np.mean(losses)), grad_norm=float(norm), lr=scheduler.get_last_lr()[0],
                       elapsed_seconds=time.monotonic()-started,
                       peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    changed = any(not torch.equal(before[n], p.detach().cpu()) for n, p in parameters)
    if not changed:
        raise RuntimeError("Adapter parameters did not change")
    write_json(out / "training_summary.json", dict(examples=len(fit), updates=updates,
               trainable_parameters=sum(p.numel() for _, p in parameters), adapter_changed=changed,
               seconds=time.monotonic()-started, smoke_only=smoke))
    del optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("download", "smoke", "train"))
    parser.add_argument("--partition", default="outer1_inner0")
    parser.add_argument("--resume", action="store_true",
                        help="Resume prediction caches / completed adapter only; never mid-training")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/pilot_v1")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    if args.command == "download":
        from huggingface_hub import snapshot_download
        print(snapshot_download(repo_id=config["model"]["id"], revision=config["model"]["revision"],
                                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]))
        return
    import torch
    from model_core import load_model, load_tokenizer, save_adapter
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA + BF16 required"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    docs, targets, parts, _ = load_inputs()
    assert args.partition in config["partitions"]
    assert config["training"]["epochs"] == 1, "This runner implements exactly one fixed epoch"
    tag = args.partition
    fit, pred = parts[tag]["fit"], parts[tag]["pred"]
    root = args.output.resolve()
    lock = runtime(config, tag, args.command)
    lock_sha = digest(lock)
    smoke_path = root / "smoke/COMPLETE.json"
    if args.command == "train":
        if not smoke_path.exists():
            raise RuntimeError("Run smoke with the same --output first")
        smoke_done = json.loads(smoke_path.read_text())
        smoke_lock = json.loads((smoke_path.parent / "runtime_lock.json").read_text())
        for key in ("fingerprint", "versions", "python", "cuda", "device", "capability", "gpu_memory_bytes"):
            assert smoke_lock[key] == lock[key], f"Smoke runtime mismatch: {key}"
        assert smoke_done["lock_sha"] == digest(smoke_lock) and smoke_done["status"] == "TECHNICAL_PASS"
    out = root / ("smoke" if args.command == "smoke" else tag)
    if out.exists():
        if not args.resume:
            raise RuntimeError(f"Output exists: {out}; use --resume or a fresh --output")
        assert json.loads((out / "runtime_lock.json").read_text()) == lock
    else:
        if args.resume:
            raise RuntimeError("Cannot resume an absent run")
        out.mkdir(parents=True)
        write_json(out / "runtime_lock.json", lock)
    if (out / "COMPLETE.json").exists():
        print(f"Already complete: {out}")
        return
    try:
        write_json(out / "status.json", dict(status="TOKENIZING", lock_sha=lock_sha))
        tokenizer, answer = load_tokenizer(config)
        tokens, audit = {}, []
        for index in sorted(set(fit + pred)):
            tokens[index], stats = prompt(tokenizer, docs[index], config)
            audit.append(dict(index=index, **stats))
        write_json(out / "prompt_audit.json", audit)
        adapter = out / "adapter"
        trained = out / "TRAIN_COMPLETE.json"
        if (out / "training.jsonl").exists() and not trained.exists():
            raise RuntimeError("Interrupted training cannot resume exactly; keep artifacts and use a fresh --output")
        if trained.exists():
            meta = json.loads(trained.read_text())
            assert meta["lock_sha"] == lock_sha
            assert all(sha(adapter / n) == h for n, h in meta["adapter_files"].items())
        model, _ = load_model(config, trainable=True, seed=config["seed"],
                              adapter_path=adapter if trained.exists() else None)
        if args.command == "train":
            write_json(out / "status.json", dict(status="FROZEN_SCORING", lock_sha=lock_sha))
            scores(model, tokens, answer, pred, docs, out / "frozen.jsonl", lock_sha, frozen=True)
        if not trained.exists():
            write_json(out / "status.json", dict(status="TRAINING", lock_sha=lock_sha))
            train(model, tokens, targets, fit, answer, config, out, smoke=args.command == "smoke")
            save_adapter(model, adapter)
            write_json(trained, dict(lock_sha=lock_sha,
                       adapter_files={p.name: sha(p) for p in adapter.iterdir() if p.is_file()}))
        sample = [next(i for i in fit if targets[i] == y) for y in range(1, 6)]
        pre = scores(model, tokens, answer, sample, docs, out / "reload_before.jsonl", lock_sha)
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        model, _ = load_model(config, trainable=True, seed=config["seed"], adapter_path=adapter)
        post = scores(model, tokens, answer, sample, docs, out / "reload_after.jsonl", lock_sha)
        error = float(np.max(np.abs(pre-post)))
        if error > 1e-4:
            raise RuntimeError(f"Adapter reload probability mismatch: {error}")
        if args.command == "train":
            write_json(out / "status.json", dict(status="TRAINED_SCORING", lock_sha=lock_sha))
            scores(model, tokens, answer, pred, docs, out / "trained.jsonl", lock_sha)
        status = "TECHNICAL_PASS" if args.command == "smoke" else "PREDICTIONS_COMPLETE"
        write_json(out / "COMPLETE.json", dict(status=status, lock_sha=lock_sha, reload_max_error=error))
        write_json(out / "status.json", dict(status=status, lock_sha=lock_sha))
    except Exception as exc:
        write_json(out / "status.json", dict(status="FAILED", lock_sha=lock_sha,
                                             error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
