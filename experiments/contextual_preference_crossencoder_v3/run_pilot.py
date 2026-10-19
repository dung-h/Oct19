#!/usr/bin/env python3
"""Causal V3 pilot on the partition where V2's ReLU head failed."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
V2 = HERE.parent / "contextual_preference_crossencoder_v2"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
sys.path.insert(0, str(V2))
import apm_context as ac
import model_v2
import model_v3 as m

ARMS = ("query_only", "contextual")
FORMER_DEAD_IDS = {"31", "86", "211", "408", "476", "522", "560", "797", "1055", "1291", "1364", "1374", "1390"}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loss_forward(ctx, encoder, head, arm, queries, refs=None, monitor=False):
    output = m.forward(ctx, encoder, head, arm, queries, refs, monitor=monitor)
    labels = torch.tensor(ctx["labels"][queries], device="cuda", dtype=torch.long)
    if arm == "query_only":
        logits, hidden = output if monitor else (output, None)
        return m.query_loss(logits.float(), labels), logits, hidden
    logits, compare, hidden = output if monitor else (*output, None)
    ref_labels = torch.tensor(ctx["labels"][refs], device="cuda", dtype=torch.long)
    return m.contextual_loss(logits.float(), compare.float(), labels, ref_labels), logits, hidden


def instantiate_matching_heads(arm):
    """Prove that V2/V3 receive byte-identical initial parameters."""
    torch.manual_seed(m.SEED)
    encoder_v2 = model_v2.adapter()
    v2_head = model_v2.head(arm)
    v2_hash = state_hash(v2_head)
    del encoder_v2, v2_head
    torch.cuda.empty_cache()
    torch.manual_seed(m.SEED)
    encoder_v3 = m.adapter()
    v3_head = m.head(arm)
    v3_hash = state_hash(v3_head)
    assert v2_hash == v3_hash, f"Initial head differs for {arm}"
    return encoder_v3, v3_head, v2_hash


def train(ctx, fit, arm):
    encoder, head, initial_head_hash = instantiate_matching_heads(arm)
    params = [p for p in encoder.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": .001},
        {"params": params, "lr": .00005},
    ], weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    refs, _ = m.references(ctx, fit, fit) if arm == "contextual" else (None, None)
    rng = np.random.default_rng(m.SEED)
    trace = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, m.EPOCHS + 1):
        encoder.eval(); head.train(); losses = []; gradients = []
        for start in range(0, len(fit), m.QUERY_BATCH):
            order = rng.permutation(len(fit)) if start == 0 else order
            loc = order[start:start + m.QUERY_BATCH]
            q = fit[loc]; r = None if refs is None else refs[loc]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                value, _, _ = loss_forward(ctx, encoder, head, arm, q, r, monitor=True)
            if not torch.isfinite(value):
                raise RuntimeError(f"{arm}: non-finite loss")
            scaler.scale(value).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(list(head.parameters()) + params, 1.)
            gradients.append(bool(torch.isfinite(norm)) and any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in params))
            scaler.step(optimizer); scaler.update(); losses.append(float(value.detach()))
        trace.append(dict(epoch=epoch, loss=float(np.mean(losses)), adapter_gradient_all_batches=bool(all(gradients)), seconds=time.monotonic()-started))
        print(arm, "epoch", epoch, "loss", round(trace[-1]["loss"], 5), flush=True)
    return encoder, head, dict(initial_head_sha256=initial_head_hash, trace=trace,
                               adapter_l2_drift=float(sum((p.detach()-q).square().sum().item() for p, q in zip(params, initial))**.5),
                               seconds=time.monotonic()-started, peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()))


def cls_for_queries(ctx, encoder, arm, queries, refs=None):
    encoder.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(queries), m.QUERY_BATCH):
            q = queries[start:start+m.QUERY_BATCH]
            r = None if refs is None else refs[start:start+len(q)]
            with torch.autocast("cuda", dtype=torch.float16):
                if arm == "query_only":
                    output = encoder(**m.query_batch(ctx, q))
                else:
                    output = encoder(**m.pair_batch(ctx, q, r))
                values.append(m.normalized_cls(output).float().cpu())
    return torch.cat(values)


def predict(ctx, encoder, head, arm, fit, pred):
    refs, _ = m.references(ctx, fit, pred) if arm == "contextual" else (None, [])
    probability = []; embeddings = []; preactivations = []
    encoder.eval(); head.eval()
    with torch.no_grad():
        for start in range(0, len(pred), m.QUERY_BATCH):
            q = pred[start:start+m.QUERY_BATCH]; r = None if refs is None else refs[start:start+len(q)]
            with torch.autocast("cuda", dtype=torch.float16):
                _, logits, _ = loss_forward(ctx, encoder, head, arm, q, r, monitor=True)
                if arm == "query_only":
                    cls = m.normalized_cls(encoder(**m.query_batch(ctx, q)))
                    pre = head.in_layer(10 * cls)
                else:
                    cls = m.normalized_cls(encoder(**m.pair_batch(ctx, q, r)))
                    labels = torch.tensor(ctx["labels"][r], device="cuda", dtype=torch.long).reshape(-1)
                    features = torch.cat([10 * cls, head.ref_label(labels - 1)], 1)
                    pre = head.in_layer(features)
            probability.append(torch.softmax(logits.float(), 1).cpu())
            embeddings.append(cls.float().cpu()); preactivations.append(pre.float().cpu())
    return np.vstack([x.numpy() for x in probability]), torch.cat(embeddings), torch.cat(preactivations), refs


def per_row_input_gradient(ctx, encoder, head, arm, pred, refs, former_mask):
    """Gradient for each former-dead row independently; no batch cancellation."""
    rows = np.flatnonzero(former_mask)
    values = []
    encoder.eval(); head.eval()
    for row in rows:
        q = pred[row:row+1]; r = None if refs is None else refs[row:row+1]
        with torch.no_grad():
            if arm == "query_only":
                cls = m.normalized_cls(encoder(**m.query_batch(ctx, q))).float()
                labels = None
            else:
                cls = m.normalized_cls(encoder(**m.pair_batch(ctx, q, r))).float()
                labels = torch.tensor(ctx["labels"][r], device="cuda", dtype=torch.long).reshape(-1)
        cls = cls.detach().requires_grad_(True)
        if arm == "query_only": logits = head(cls)
        else: logits, _ = head(cls, labels)
        target = torch.tensor(ctx["labels"][q], device="cuda", dtype=torch.long)
        loss = m.query_loss(logits.float(), target)
        gradient = torch.autograd.grad(loss, cls)[0]
        values.append(float(gradient.abs().max()))
    return rows, values


def save_and_reload(ctx, encoder, head, arm, fit, pred, probability):
    checkpoint = OUT / f"{arm}_head.pt"; adapter_dir = OUT / f"{arm}_adapter"
    torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, checkpoint)
    encoder.save_adapter(str(adapter_dir), "preference")
    del encoder, head
    torch.cuda.empty_cache()
    replay_encoder = m.adapter()
    replay_encoder.delete_adapter("preference")
    replay_encoder.load_adapter(str(adapter_dir), load_as="preference", set_active=True)
    replay_encoder.to("cuda").eval()
    replay_head = m.head(arm)
    replay_head.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    replay, _, _, _ = predict(ctx, replay_encoder, replay_head, arm, fit, pred)
    delta = float(np.max(np.abs(replay - probability)))
    return delta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = ac.context(); input_stats = m.prepare(ctx)
    _, fit_outer, _, inner_numbers = ac.outer_jobs(ctx)[3]
    fit, pred = fit_outer[inner_numbers != 3], fit_outer[inner_numbers == 3]
    assert len(fit) == 1497 and len(pred) == 498
    assert set(ctx["groups"][fit]).isdisjoint(ctx["groups"][pred])
    former_mask = np.asarray([str(x) in FORMER_DEAD_IDS for x in ctx["ids"][pred]])
    assert int(former_mask.sum()) == 13
    protocol = dict(v3_change="ReLU to LeakyReLU only", negative_slope=m.NEGATIVE_SLOPE,
        partition="outer3_inner3", input_stats=input_stats,
        sources={"model_v3.py": sha_file(HERE / "model_v3.py"), "run_pilot.py": sha_file(__file__),
                 "v2_model.py": sha_file(V2 / "model_v2.py"), "v1_model.py": sha_file(V2.parent / "contextual_preference_crossencoder" / "model.py")})
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    reports = {}
    for arm in ARMS:
        encoder, head, train_report = train(ctx, fit, arm)
        probability, embeddings, pre, refs = predict(ctx, encoder, head, arm, fit, pred)
        rows, gradient_max = per_row_input_gradient(ctx, encoder, head, arm, pred, refs, former_mask)
        ids = [str(ctx["ids"][pred[x]]) for x in rows]
        dead_like = np.isclose(probability, probability[former_mask][0], atol=1e-7, rtol=0).all(1)
        unique, counts = np.unique(probability, axis=0, return_counts=True)
        # Fixed text-sensitivity probe: reverse the 498 prediction rows, retaining references for contextual arm.
        perm = np.arange(len(pred))[::-1]
        perm_probability, _, _, _ = predict(ctx, encoder, head, arm, fit, pred[perm])
        # Compare fixed batch positions after replacing every query with a different paper.
        # This tests dependence on the query content; it does not use target labels.
        reverse_delta = np.abs(perm_probability - probability).max(1)
        reload_delta = save_and_reload(ctx, encoder, head, arm, fit, pred, probability)
        report = dict(**train_report, n_train=len(fit), n_pred=len(pred),
            former_relu_dead_ids=ids, former_relu_dead_gradient_max_abs=gradient_max,
            former_relu_dead_all_gradients_nonzero=bool(all(x > 0 for x in gradient_max)),
            former_relu_dead_all_finite=bool(np.isfinite(gradient_max).all()),
            unique_probability_vectors=int(len(unique)), largest_duplicate=int(counts.max()),
            former_dead_exact_output_cluster=int((dead_like & former_mask).sum()),
            former_dead_max_preactivation=(
                pre[former_mask].max(1).values.tolist() if arm == "query_only"
                else pre.reshape(len(pred), m.N_REFERENCE, -1)[former_mask].amax(dim=(1, 2)).tolist()
            ),
            all_probability_finite=bool(np.isfinite(probability).all()),
            read_std=float((1-probability[:,0]).std()),
            reversed_query_max_abs_delta_quantiles=np.quantile(reverse_delta, [0,.25,.5,.75,1]).tolist(),
            text_sensitive_all_rows=bool(np.all(reverse_delta > 1e-7)),
            reload_max_abs_probability_delta=reload_delta)
        report["checks"] = dict(adapter_gradient_all_batches=bool(all(x["adapter_gradient_all_batches"] for x in train_report["trace"])),
            former_rows_gradient=report["former_relu_dead_all_gradients_nonzero"],
            output_variation=report["unique_probability_vectors"] >= max(8, len(pred)//10),
            no_dominant_vector=report["largest_duplicate"] < len(pred)//2,
            text_sensitive=report["text_sensitive_all_rows"], reload_parity=reload_delta <= 1e-6,
            finite=report["all_probability_finite"])
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
        np.savez_compressed(OUT / f"{arm}_prediction.npz", fit=fit, pred=pred, probability=probability,
                            embeddings=embeddings.numpy(), preactivation=pre.numpy(), former_dead_offsets=rows)
        (OUT / f"{arm}_report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports[arm] = report
        print(arm, report["status"], json.dumps(report["checks"]), flush=True)
    overall = dict(status="PASSED" if all(r["status"] == "PASSED" for r in reports.values()) else "FAILED", reports=reports)
    (OUT / "pilot_summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    if overall["status"] != "PASSED":
        raise SystemExit("V3 pilot failed its predeclared guardrails")


if __name__ == "__main__":
    main()
