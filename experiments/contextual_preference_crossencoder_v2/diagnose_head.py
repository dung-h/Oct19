#!/usr/bin/env python3
"""Replay one failed fit unchanged and audit its head; do not relax any guard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import evaluate as ev

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "head_diagnostic"


def encode(ctx, encoder, indices, amp=True):
    values = []
    raw_norms = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(indices), ev.m.QUERY_BATCH):
            ix = indices[start:start + ev.m.QUERY_BATCH]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                raw = encoder(**ev.m.query_batch(ctx, ix)).last_hidden_state[:, 0].float()
            values.append(F.normalize(raw, dim=1).cpu())
            raw_norms.append(raw.norm(dim=1).cpu())
    return torch.cat(values), torch.cat(raw_norms)


def preactivation_stats(x, weight, bias):
    pre = F.linear(x.float(), weight.float(), bias.float())
    active = pre > 0
    return dict(dead_rows=int((~active.any(1)).sum()),
                total_rows=int(len(x)), globally_never_active_units=int((~active.any(0)).sum()),
                active_units_per_row_quantiles=np.quantile(active.sum(1).numpy(), [0, .25, .5, .75, 1]).tolist(),
                max_preactivation_quantiles=np.quantile(pre.max(1).values.numpy(), [0, .25, .5, .75, 1]).tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = ev.ac.context()
    stats = ev.m.prepare(ctx)
    _, fit, _, nums = ev.ac.outer_jobs(ctx)[3]
    train, pred = fit[nums != 3], fit[nums == 3]
    np.testing.assert_array_equal(np.intersect1d(train, pred), [])
    assert set(ctx["groups"][train]).isdisjoint(ctx["groups"][pred])

    source_digest = hashlib.sha256(Path(ev.__file__).read_bytes()).hexdigest()
    expected = json.loads((HERE / "outputs" / "evaluation_manifest.json").read_text())["source_sha256"]
    assert source_digest == expected

    captured = {}
    original_adapter = ev.m.adapter
    original_head = ev.m.head
    original_predict = ev.predict

    def adapter_factory():
        model = original_adapter()
        captured["initial_embeddings"], _ = encode(ctx, model, pred)
        return model

    def head_factory(arm):
        head = original_head(arm)
        captured["initial_weight"] = head.in_layer.weight.detach().cpu().clone()
        captured["initial_bias"] = head.in_layer.bias.detach().cpu().clone()
        captured["train_activation_trace"] = []

        def hook(module, inputs, output):
            if head.training and torch.is_grad_enabled():
                pre = output.detach().float()
                captured["train_activation_trace"].append(dict(
                    step=len(captured["train_activation_trace"]) + 1,
                    dead_rows=int((~(pre > 0).any(1)).sum()), rows=len(pre),
                    mean_active_units=float((pre > 0).sum(1).float().mean()),
                    max_preactivation_mean=float(pre.max(1).values.mean())))

        head.in_layer.register_forward_hook(hook)
        return head

    def predict_capture(context, encoder, head, arm, ref_pool, indices):
        assert arm == "query_only"
        p, audit = original_predict(context, encoder, head, arm, ref_pool, indices)
        captured["probability"] = p
        captured["original_prediction_audit"] = audit
        captured["final_embeddings"], captured["raw_cls_norms"] = encode(context, encoder, indices)
        captured["final_weight"] = head.in_layer.weight.detach().cpu().clone()
        captured["final_bias"] = head.in_layer.bias.detach().cpu().clone()
        head.eval()
        hidden_rows = []
        pre_rows = []
        with torch.no_grad():
            for start in range(0, len(indices), ev.m.QUERY_BATCH):
                z = captured["final_embeddings"][start:start + ev.m.QUERY_BATCH].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    pre = head.in_layer(10 * z)
                    hidden = F.relu(pre)
                hidden_rows.append(hidden.float().cpu()); pre_rows.append(pre.float().cpu())
        captured["hidden"] = torch.cat(hidden_rows)
        captured["preactivation_amp"] = torch.cat(pre_rows)
        dead = np.flatnonzero(captured["hidden"].abs().sum(1).numpy() == 0)
        captured["dead_offsets"] = dead
        if len(dead):
            fp32_embeddings, _ = encode(context, encoder, indices[dead], amp=False)
            with torch.no_grad():
                fp32_pre = F.linear(10 * fp32_embeddings, captured["final_weight"], captured["final_bias"])
            captured["dead_fp32"] = dict(dead_rows=int((~(fp32_pre > 0).any(1)).sum()),
                                         max_preactivation=fp32_pre.max(1).values.tolist())
            z = captured["final_embeddings"][dead].cuda().requires_grad_(True)
            logits = head(z).float()
            targets = torch.tensor(context["labels"][indices[dead]], dtype=torch.long, device="cuda")
            loss = ev.m.query_loss(logits, targets)
            gradient = torch.autograd.grad(loss, z)[0]
            captured["dead_input_gradient_max_abs"] = float(gradient.abs().max())

        # Save a diagnostic checkpoint without changing the failed fit's cache.
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, OUT / "head.pt")
        encoder.save_adapter(str(OUT / "adapter"), "preference")
        return p, audit

    ev.m.adapter = adapter_factory
    ev.m.head = head_factory
    ev.predict = predict_capture
    try:
        ev.fit_predict(ctx, train, pred, "query_only", HERE / "outputs" / "outer3_inner3_query_only")
    except RuntimeError as error:
        captured["error"] = str(error)
        tb = error.__traceback__
        while tb is not None:
            value = tb.tb_frame.f_locals.get("info")
            if isinstance(value, dict) and "prediction_audit" in value:
                captured["failure_info"] = value
            tb = tb.tb_next
        # The exception retains the training frame; release GPU state before reload.
        error.__traceback__ = None
        tb = None
    finally:
        ev.m.adapter = original_adapter
        ev.m.head = original_head
        ev.predict = original_predict

    assert "failure_info" in captured, "Expected the original, unchanged guard failure"
    probability = captured["probability"]
    dead = captured["dead_offsets"]
    initial, final = captured["initial_embeddings"], captured["final_embeddings"]
    wi, bi = captured["initial_weight"], captured["initial_bias"]
    wf, bf = captured["final_weight"], captured["final_bias"]
    pre = captured["preactivation_amp"].numpy()
    token_lengths = np.asarray([len(ctx["side_ids"][int(q)]) for q in pred])
    token_hashes = [ev.m.v1.sha(ctx["side_ids"][int(q)]) for q in pred]
    token_counts = pd.Series(token_hashes).value_counts()
    samples = ctx["data"].iloc[pred][["id", "title", "venue", "abstract"]].copy()
    samples["row_index"] = pred
    samples["label"] = ctx["labels"][pred]
    samples["dead_relu"] = np.isin(np.arange(len(pred)), dead)
    samples["input_tokens"] = token_lengths
    samples["raw_tokens"] = ctx["raw_token_lengths"][pred]
    samples["truncated"] = samples.raw_tokens > ev.m.SIDE_TOKENS
    samples["token_hash"] = token_hashes
    samples["identical_token_input_count"] = [int(token_counts[h]) for h in token_hashes]
    samples["active_units"] = (pre > 0).sum(1)
    samples["max_preactivation"] = pre.max(1)
    samples["read_probability"] = 1 - probability[:, 0]
    samples["raw_cls_norm"] = captured["raw_cls_norms"].numpy()
    samples.to_csv(OUT / "sample_audit.csv", index=False)

    counterfactuals = {
        "initial_embeddings_initial_head": preactivation_stats(10 * initial, wi, bi),
        "final_embeddings_final_head": preactivation_stats(10 * final, wf, bf),
        "final_embeddings_initial_head": preactivation_stats(10 * final, wi, bi),
        "initial_embeddings_final_head": preactivation_stats(10 * initial, wf, bf),
        "final_embeddings_final_weights_initial_bias": preactivation_stats(10 * final, wf, bi),
        "final_embeddings_initial_weights_final_bias": preactivation_stats(10 * final, wi, bf),
    }
    np.savez_compressed(OUT / "representations.npz", train=train, pred=pred, probability=probability,
                        initial_embeddings=initial.numpy(), final_embeddings=final.numpy(),
                        preactivation_amp=pre, dead_offsets=dead)

    # Reload the saved adapter and head, verifying real inference on all held-out inputs.
    encoder = original_adapter()
    encoder.delete_adapter("preference")
    encoder.load_adapter(str(OUT / "adapter"), load_as="preference", set_active=True)
    encoder.to("cuda").eval()
    head = original_head("query_only")
    head.load_state_dict(torch.load(OUT / "head.pt", map_location="cuda", weights_only=True))
    replay, _ = original_predict(ctx, encoder, head, "query_only", train, pred)
    reload_delta = float(np.max(np.abs(replay - probability)))
    assert np.allclose(replay, probability, atol=1e-6, rtol=1e-5)

    report = dict(
        status="DIAGNOSED_ORIGINAL_FAILURE", source_sha256=source_digest,
        input_stats=stats, partition="outer3_inner3_query_only", n_train=len(train), n_pred=len(pred),
        original_failure_info=captured["failure_info"], dead_rows=int(len(dead)),
        dead_ids=samples[samples.dead_relu].id.astype(str).tolist(),
        dead_label_counts=samples[samples.dead_relu].label.value_counts().sort_index().to_dict(),
        dead_venue_counts=samples[samples.dead_relu].venue.value_counts().to_dict(),
        dead_unique_token_inputs=len(set(np.asarray(token_hashes)[dead])),
        dead_duplicate_input_rows=int((samples[samples.dead_relu].identical_token_input_count > 1).sum()),
        dead_missing_abstracts=int(samples[samples.dead_relu].abstract.fillna("").eq("").sum()),
        dead_truncated=int(samples[samples.dead_relu].truncated.sum()),
        dead_raw_token_lengths=samples[samples.dead_relu].raw_tokens.tolist(),
        dead_fp32=captured.get("dead_fp32"),
        dead_input_gradient_max_abs=captured.get("dead_input_gradient_max_abs"),
        counterfactuals=counterfactuals,
        head_bias_initial_quantiles=np.quantile(bi.numpy(), [0, .25, .5, .75, 1]).tolist(),
        head_bias_final_quantiles=np.quantile(bf.numpy(), [0, .25, .5, .75, 1]).tolist(),
        head_weight_l2_drift=float((wf - wi).norm()), head_bias_l2_drift=float((bf - bi).norm()),
        reload_max_abs_probability_delta=reload_delta,
        activation_trace=captured["train_activation_trace"],
        caveat="Counterfactual head/embedding swaps diagnose this checkpoint; they are not fitted candidates or estimates of H improvement.",
    )
    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    printable = {key: value for key, value in report.items() if key not in ["activation_trace", "original_failure_info"]}
    print(json.dumps(printable, indent=2), flush=True)


if __name__ == "__main__":
    main()
