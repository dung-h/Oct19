#!/usr/bin/env python3
"""Resumable, grouped nested evaluation for the locked contextual cross-encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
import apm_context as ac
import model

ARMS = ("query_only", "contextual")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y, probability, ids, threshold):
    selected = 1 - probability[:, 0] > threshold
    priority = np.where(selected, 1 + 9 * (probability @ (np.arange(5) / 4)), 0.)
    order = np.asarray(sorted(np.flatnonzero(selected), key=lambda i: (-priority[i], str(ids[i]))), dtype=int)
    gain = (y[order] - 1) / 4
    q = float(np.average(gain, weights=1 / np.log2(np.arange(len(order)) + 2))) if len(order) else 0.
    r = float(((y > 1) & selected).sum() / (y > 1).sum())
    return dict(H=float(2*q*r/(q+r)) if q+r else 0., Q=q, R=r, accepted=int(selected.sum()),
                TP=int(((y > 1) & selected).sum()), FP=int(((y == 1) & selected).sum()),
                FN=int(((y > 1) & ~selected).sum()), missed_label4=int(((y == 4) & ~selected).sum()),
                missed_label5=int(((y == 5) & ~selected).sum()), priority=priority)


def _head(arm):
    return model.QueryHead().to("cuda") if arm == "query_only" else model.ContextHead().to("cuda")


def _forward(ctx, encoder, head, arm, queries, refs=None):
    if arm == "query_only":
        cls = encoder(**model.query_batch(ctx, queries)).last_hidden_state[:, 0].float()
        return head(cls), None
    cls = encoder(**model.pair_batch(ctx, queries, refs)).last_hidden_state[:, 0].float()
    ref_labels = torch.tensor(ctx["labels"][refs], device="cuda", dtype=torch.long)
    return head(cls, ref_labels.reshape(-1))


def predict(ctx, encoder, head, arm, fit, indices):
    values = np.empty((len(indices), 5), dtype=np.float32)
    refs, _ = model.references(ctx, fit, indices) if arm == "contextual" else (None, None)
    encoder.eval(); head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), model.QUERY_BATCH):
            q = indices[start:start + model.QUERY_BATCH]
            r = None if refs is None else refs[start:start + len(q)]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = _forward(ctx, encoder, head, arm, q, r)
            values[start:start+len(q)] = torch.softmax(logits.float(), 1).cpu().numpy()
    assert np.isfinite(values).all() and np.allclose(values.sum(1), 1, atol=1e-6)
    return values


def fit_predict(ctx, fit, pred, arm, prefix):
    """Fit exactly two fixed epochs and cache held-out probabilities and audit trace."""
    cache = Path(str(prefix) + ".npz"); meta = Path(str(prefix) + ".json")
    if cache.exists() and meta.exists():
        z = np.load(cache)
        np.testing.assert_array_equal(z["fit"], fit); np.testing.assert_array_equal(z["pred"], pred)
        info = json.loads(meta.read_text())
        if info["arm"] == arm and info["epochs"] == model.EPOCHS:
            return z["probability"], info
    assert set(fit).isdisjoint(pred)
    if np.all(pred < len(ctx["labels"])):
        assert set(ctx["groups"][fit]).isdisjoint(ctx["groups"][pred])
    torch.manual_seed(model.SEED); rng = np.random.default_rng(model.SEED)
    encoder = model._adapter(); head = _head(arm)
    adapter_params = [p for p in encoder.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in adapter_params]
    opt = torch.optim.AdamW([{ "params": list(head.parameters()), "lr": .001},
                             { "params": adapter_params, "lr": .00005}], weight_decay=.01)
    train_refs, train_audit = model.references(ctx, fit, fit) if arm == "contextual" else (None, [])
    trace=[]; nonzero_adapter_steps=0; started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, model.EPOCHS + 1):
        # AutoAdapterModel's train_adapter setup is kept in eval mode in the
        # repository's validated SPECTER2 implementation: adapter parameters
        # still receive gradients, while frozen encoder dropout stays disabled.
        encoder.eval(); head.train(); losses=[]
        order = rng.permutation(len(fit))
        for start in range(0, len(fit), model.QUERY_BATCH):
            loc = order[start:start+model.QUERY_BATCH]
            q = fit[loc]; refs = None if train_refs is None else train_refs[loc]
            labels = torch.tensor(ctx["labels"][q], device="cuda", dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, compare = _forward(ctx, encoder, head, arm, q, refs)
                if arm == "contextual":
                    ref_labels = torch.tensor(ctx["labels"][refs], device="cuda", dtype=torch.long)
                    value = model.contextual_loss(logits.float(), compare.float(), labels, ref_labels)
                else:
                    value = model.query_loss(logits.float(), labels)
            assert torch.isfinite(value); value.backward()
            nonzero_adapter_steps += int(any(p.grad is not None and torch.count_nonzero(p.grad) for p in adapter_params))
            norm = torch.nn.utils.clip_grad_norm_(list(head.parameters()) + adapter_params, 1.)
            assert torch.isfinite(norm)
            opt.step(); losses.append(float(value.detach()))
        trace.append(dict(epoch=epoch, loss=float(np.mean(losses)), seconds=time.monotonic()-started))
        print(Path(prefix).name, arm, "epoch", epoch, "loss", round(trace[-1]["loss"], 5), "seconds", round(trace[-1]["seconds"], 1), flush=True)
    probability = predict(ctx, encoder, head, arm, fit, pred)
    info = dict(arm=arm, epochs=model.EPOCHS, seed=model.SEED, fit=fit.tolist(), pred=pred.tolist(),
                seconds=time.monotonic()-started, trace=trace, peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                adapter_l2_drift=float(sum((p.detach()-a).square().sum().item() for p,a in zip(adapter_params, initial))**.5),
                trained_adapter_parameters=int(sum(p.numel() for p in adapter_params)),
                reference_audit=dict(queries=len(train_audit), negative_venue_only=int(sum(x["negative_same_venue"] for x in train_audit)),
                                     positive_venue_only=int(sum(x["positive_same_venue"] for x in train_audit)),
                                     repeated=int(sum(x["negative_repeated"]+x["positive_repeated"] for x in train_audit))))
    assert nonzero_adapter_steps > 0 and info["adapter_l2_drift"] > 0
    np.savez_compressed(cache, fit=fit, pred=pred, probability=probability)
    meta.write_text(json.dumps(info, indent=2)+"\n")
    del encoder, head, opt, adapter_params, initial; torch.cuda.empty_cache()
    return probability, info


def inner(ctx, fit, nums, prefix):
    allp={arm:np.full((len(fit),5), np.nan, np.float32) for arm in ARMS}
    for inner_fold in range(4):
        mask = nums == inner_fold; a,b=fit[~mask],fit[mask]
        for arm in ARMS:
            p,_=fit_predict(ctx,a,b,arm,OUT/f"{prefix}_inner{inner_fold}_{arm}")
            allp[arm][mask]=p
        print(prefix, "inner", inner_fold, "complete", flush=True)
    assert all(np.isfinite(v).all() for v in allp.values())
    choices={}
    for arm,p in allp.items():
        candidates=[]
        for threshold in model.THRESHOLDS:
            folds=[score(ctx["labels"][fit[nums==k]],p[nums==k],ctx["ids"][fit[nums==k]],threshold) for k in range(4)]
            candidates.append(dict(arm=arm,threshold=float(threshold),inner_H=float(np.mean([x["H"] for x in folds])),inner_Q=float(np.mean([x["Q"] for x in folds])),inner_R=float(np.mean([x["R"] for x in folds]))))
        choices[arm]=max(candidates,key=lambda x:x["inner_H"])
    return choices


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--fold", type=int, choices=range(5), action="append")
    args=parser.parse_args(); ctx=ac.context(); stats=model.prepare(ctx)
    locked=json.loads((OUT/"protocol.json").read_text())
    assert locked["input_stats"] == stats, "Input preprocessing does not match preflight."
    manifest=dict(protocol_sha256=digest(OUT/"protocol.json"), evaluation_source_sha256=digest(__file__), status="started")
    (OUT/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    requested=set(args.fold) if args.fold else set(range(5)); rows=[]
    for fold,fit,val,nums in ac.outer_jobs(ctx):
        if fold not in requested: continue
        choices=inner(ctx,fit,nums,f"outer{fold}")
        (OUT/f"outer{fold}_choices.json").write_text(json.dumps(choices,indent=2)+"\n")
        for arm,choice in choices.items():
            p,info=fit_predict(ctx,fit,val,arm,OUT/f"outer{fold}_validation_{arm}")
            result=score(ctx["labels"][val],p,ctx["ids"][val],choice["threshold"])
            rows.append(dict(outer_fold=fold, arm=arm, threshold=choice["threshold"], inner_H=choice["inner_H"], **{k:v for k,v in result.items() if k!="priority"}, seconds=info["seconds"]))
        pd.DataFrame(rows).to_csv(OUT/"fold_metrics_partial.csv",index=False)
        print(pd.DataFrame(rows).query("outer_fold==@fold").to_string(index=False),flush=True)
    manifest["status"]="complete"; manifest["completed_folds"]=sorted(requested)
    (OUT/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
