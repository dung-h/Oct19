#!/usr/bin/env python3
"""Training-only hard-pair metric-learning pilot for SPECTER2."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
sys.path.insert(0, str(HERE.parent / "specter2_preference"))
import apm_context as ac
import modeling as old

SEED = 20261101
MAX_LENGTH = 256
BATCH_PAIRS = 8
EPOCHS = 2
MARGIN = 0.70


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(ctx: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(
        plan_sha256=digest(HERE / "PLAN.md"), source_sha256=digest(HERE / "run_preflight.py"),
        split_sha256=digest(ac.split_path()), seed=SEED, max_length=MAX_LENGTH,
        pair_rule="Each Label>1 training paper gets its closest Label1 frozen-retrieval neighbour; same venue preferred; duplicate group excluded.",
        loss="softplus(-(read(pos)-read(neg))) + 0.5*relu(cos(pos,neg)-0.70) + 0.1*(1-cos(adapted,frozen))",
        training="SPECTER2 regression adapter plus scalar head; two fixed epochs; batch eight hard pairs.",
        decision="Pilot only. Continue to new nested CV only if held-out hard-pair cosine decreases and held-out read AUC exceeds frozen logistic control.",
    )
    target = OUT / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == value, "Protocol changed; use a fresh directory."
    else:
        target.write_text(json.dumps(value, indent=2) + "\n")


def prepare(ctx: dict) -> None:
    _, _, tokenizer = old.resources()
    texts = (ctx["data"].title.astype(str) + tokenizer.sep_token + ctx["data"].abstract.astype(str)).tolist()
    raw = tokenizer(texts, truncation=False, return_token_type_ids=False)
    ctx["tokenizer"] = tokenizer
    ctx["tokens"] = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, return_token_type_ids=False)
    ctx["input_stats"] = dict(max_length=MAX_LENGTH, n=len(texts),
        truncated=int(sum(len(x) > MAX_LENGTH for x in raw["input_ids"])),
        token_sha256=hashlib.sha256(json.dumps(ctx["tokens"].data, sort_keys=True).encode()).hexdigest())


def hard_pairs(ctx: dict, fit: np.ndarray) -> np.ndarray:
    """One deterministic, no-leakage hard negative per worthwhile fit paper."""
    fit = np.asarray(fit, int)
    labels, groups, venues, ids = ctx["labels"], ctx["groups"], ctx["venues"], ctx["ids"]
    space = ctx["spaces"]["retrieval_margin"]
    positives = fit[labels[fit] > 1]
    negatives = fit[labels[fit] == 1]
    pairs = []
    for positive in positives:
        candidates = negatives[groups[negatives] != groups[positive]]
        same_venue = candidates[venues[candidates] == venues[positive]]
        pool = same_venue if len(same_venue) else candidates
        assert len(pool)
        cosine = space[pool] @ space[positive]
        negative = sorted(range(len(pool)), key=lambda i: (-float(cosine[i]), str(ids[pool[i]])))[0]
        pairs.append((int(positive), int(pool[negative])))
    result = np.asarray(pairs, int)
    assert len(result) == len(positives)
    assert np.all(labels[result[:, 0]] > 1) and np.all(labels[result[:, 1]] == 1)
    assert np.all(groups[result[:, 0]] != groups[result[:, 1]])
    return result


def batch(ctx: dict, indices: np.ndarray) -> dict:
    tokens = ctx["tokens"]
    value = ctx["tokenizer"].pad([{k: tokens[k][int(i)] for k in tokens} for i in indices], return_tensors="pt")
    return {k: v.to("cuda") for k, v in value.items()}


def embed(model: torch.nn.Module, ctx: dict, indices: np.ndarray) -> torch.Tensor:
    hidden = model(**batch(ctx, indices)).last_hidden_state[:, 0].float()
    return F.normalize(hidden, dim=1)


def encode(model: torch.nn.Module, ctx: dict, indices: np.ndarray) -> np.ndarray:
    result = np.zeros((len(indices), 768), np.float32)
    order = np.argsort([len(ctx["tokens"]["input_ids"][int(i)]) for i in indices], kind="stable")
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 16):
            positions = order[start:start + 16]
            with torch.autocast("cuda", dtype=torch.float16):
                result[positions] = embed(model, ctx, indices[positions]).cpu().numpy()
    return result


def main() -> None:
    torch.manual_seed(SEED)
    ctx = ac.context(); prepare(ctx); protocol(ctx)
    outer, fit, _, numbers = ac.outer_jobs(ctx)[0]
    train, probe = fit[numbers != 0], fit[numbers == 0]
    assert set(ctx["groups"][train]).isdisjoint(ctx["groups"][probe])
    pairs = hard_pairs(ctx, train)

    # The frozen vector is a fixed anchor only.  It contains no train labels.
    frozen = ctx["spaces"]["retrieval_margin"]
    model = old.encoder(True)
    head = nn.Linear(768, 1).to("cuda")
    adapter = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{ "params": adapter, "lr": 5e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=.01)
    rng = np.random.default_rng(SEED); trace = []; started = time.monotonic()
    initial = [p.detach().clone() for p in adapter]
    for epoch in range(EPOCHS):
        losses = []
        for positions in range(0, len(pairs), BATCH_PAIRS):
            take = rng.permutation(len(pairs))[positions:positions + BATCH_PAIRS] if positions == 0 else order[positions:positions + BATCH_PAIRS]
            if positions == 0:
                order = rng.permutation(len(pairs)); take = order[:BATCH_PAIRS]
            current = pairs[take]
            indices = current.reshape(-1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = embed(model, ctx, indices).float(); positive, negative = z[0::2], z[1::2]
                score_p, score_n = head(positive).squeeze(1), head(negative).squeeze(1)
                rank = F.softplus(-(score_p - score_n)).mean()
                cosine = (positive * negative).sum(1)
                separation = F.relu(cosine - MARGIN).mean()
                anchor = (1 - (z * torch.from_numpy(frozen[indices]).to("cuda")).sum(1)).mean()
                loss = rank + .5 * separation + .1 * anchor
            assert torch.isfinite(loss); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter + list(head.parameters()), 1.)
            opt.step(); losses.append(float(loss.detach()))
        trace.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), seconds=time.monotonic() - started))
        print(f"epoch={epoch + 1} loss={trace[-1]['loss']:.5f} seconds={trace[-1]['seconds']:.1f}", flush=True)

    probe_pairs = hard_pairs(ctx, probe)
    probe_index = np.unique(probe_pairs.reshape(-1))
    adapted_values = encode(model, ctx, probe_index)
    location = {int(index): pos for pos, index in enumerate(probe_index)}
    adapted_cosine = np.asarray([adapted_values[location[p]] @ adapted_values[location[n]] for p, n in probe_pairs])
    frozen_cosine = np.asarray([frozen[p] @ frozen[n] for p, n in probe_pairs])

    probe_vectors = encode(model, ctx, probe)
    with torch.inference_mode():
        read = torch.sigmoid(head(torch.from_numpy(probe_vectors).to("cuda")).squeeze(1)).cpu().numpy()
    control = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(frozen[train], ctx["labels"][train] > 1)
    control_read = control.predict_proba(frozen[probe])[:, 1]
    drift = float(sum((p.detach() - q).square().sum().item() for p, q in zip(adapter, initial)) ** .5)
    result = dict(status="COMPLETE", outer_fold=int(outer), train_rows=int(len(train)), probe_rows=int(len(probe)),
        train_pairs=int(len(pairs)), probe_pairs=int(len(probe_pairs)), input=ctx["input_stats"], trace=trace,
        trainable_adapter_parameters=int(sum(p.numel() for p in adapter)), adapter_l2_drift=drift,
        hard_pair_cosine=dict(frozen_mean=float(frozen_cosine.mean()), adapted_mean=float(adapted_cosine.mean()),
            delta=float(adapted_cosine.mean() - frozen_cosine.mean()), fraction_decreased=float((adapted_cosine < frozen_cosine).mean())),
        read_auc=dict(frozen_logistic=float(roc_auc_score(ctx["labels"][probe] > 1, control_read)),
            adapted_head=float(roc_auc_score(ctx["labels"][probe] > 1, read))),
        continuation_allowed=bool(adapted_cosine.mean() < frozen_cosine.mean() and roc_auc_score(ctx["labels"][probe] > 1, read) > roc_auc_score(ctx["labels"][probe] > 1, control_read)))
    (OUT / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    del model, head, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
