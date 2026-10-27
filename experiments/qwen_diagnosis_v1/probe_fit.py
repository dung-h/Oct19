"""Fit-only inference and gradient probes of existing, immutable Qwen adapters."""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from analyze import HERE, PILOT, metrics, write_csv
from common import check_probability, digest, fingerprint, jsonl, load_inputs, prompt, sha, write_json
from model_core import five_logits, load_model, load_tokenizer


def score(model, ids, answer, frozen=False):
    model.eval()
    ctx = model.disable_adapter() if frozen else contextlib.nullcontext()
    with ctx, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = five_logits(model, torch.tensor([ids],device="cuda"), answer, train=False).softmax(-1)
        result = p.squeeze(0).cpu().numpy().astype(float)
    # Retain numerical behavior while releasing unused reserved segments for
    # the concurrent user job. Saved fit-probe parity is checked before use.
    torch.cuda.empty_cache()
    return check_probability([result],1)[0]


def adapter_norms(path):
    state=load_file(str(path/"adapter_model.safetensors"),device="cpu")
    rows=[]
    for name,a in state.items():
        if "lora_A" not in name:
            continue
        b=state[name.replace("lora_A","lora_B")].float()
        a=a.float()
        norm=float(torch.sqrt(torch.trace((b.T@b)@(a@a.T)).clamp_min(0)))
        rows.append(dict(module=name.split(".lora_A")[0],A_norm=float(a.norm()),
                         B_norm=float(b.norm()),unscaled_BA_norm=norm))
    assert len(rows)==56 and all(r["B_norm"]>0 for r in rows)
    return rows


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument("--source-output",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--runtime-smoke-output",type=Path,required=True)
    args=ap.parse_args()
    source,out=args.source_output.resolve(),args.output.resolve()
    if out.exists():
        raise RuntimeError("Probe output exists; retain artifacts and use a new namespace")
    docs,targets,parts,_=load_inputs()
    config=json.loads((PILOT/"config.json").read_text())
    tokenizer,answer=load_tokenizer(config)
    source_fp=fingerprint()
    smoke=args.runtime_smoke_output.resolve()/"smoke"
    smoke_lock=json.loads((smoke/"runtime_lock.json").read_text())
    smoke_done=json.loads((smoke/"COMPLETE.json").read_text())
    assert smoke_lock["fingerprint"]==source_fp
    assert smoke_done["status"]=="TECHNICAL_PASS" and smoke_done["lock_sha"]==digest(smoke_lock)
    assert all(importlib.metadata.version(k)==v for k,v in smoke_lock["versions"].items())
    out.mkdir(parents=True)
    write_json(out/"protocol_lock.json",dict(source=str(source),source_fingerprint=source_fp,
        runtime_smoke=str(smoke),
        protocol_sha256=sha(HERE/"PROTOCOL.md"),probe_code_sha256=sha(Path(__file__)),
        seed=20261028,samples=128,sensitivity_samples=32,optimizer_steps=0,
        allocator_environment={k:os.environ.get(k) for k in ("PYTORCH_ALLOC_CONF","PYTORCH_CUDA_ALLOC_CONF")},
        allocator_cache_released_between_probes=True))
    summaries={}
    try:
        for tag,part in parts.items():
            dest=out/tag
            dest.mkdir()
            lock=json.loads((source/tag/"runtime_lock.json").read_text())
            assert lock["fingerprint"]==source_fp
            assert all(importlib.metadata.version(k)==v for k,v in lock["versions"].items())
            done=json.loads((source/tag/"COMPLETE.json").read_text())
            assert done["status"]=="PREDICTIONS_COMPLETE" and done["lock_sha"]==digest(lock)
            train_done=json.loads((source/tag/"TRAIN_COMPLETE.json").read_text())
            adapter=source/tag/"adapter"
            assert all(sha(adapter/name)==h for name,h in train_done["adapter_files"].items())
            rng=np.random.default_rng(20261028)
            fit=sorted(part["fit"])
            indices=[int(i) for i in rng.choice(fit,size=128,replace=False)]
            assert set(indices)<=set(fit) and set(indices).isdisjoint(part["pred"])
            y=np.array([targets[i] for i in indices])
            tokens={i:prompt(tokenizer,docs[i],config)[0] for i in indices}
            write_json(dest/"membership.json",dict(fit_indices=indices,source_lock_sha=digest(lock)))
            write_csv(dest/"adapter_norms.csv",adapter_norms(adapter))
            started=time.monotonic()
            model,_=load_model(config,trainable=True,seed=config["seed"],adapter_path=adapter)
            loading_seconds=time.monotonic()-started
            torch.cuda.empty_cache()
            parity=[]
            for row in jsonl(source/tag/"reload_after.jsonl"):
                index=row["index"]
                assert index in fit
                ids=prompt(tokenizer,docs[index],config)[0]
                p=score(model,ids,answer)
                parity.append(float(np.max(np.abs(p-np.array(row["probabilities"])))))
            assert max(parity)<=1e-4, f"Saved fit parity failed: {max(parity)}"
            rows=[]
            timing={}
            probabilities={}
            for arm in ("frozen","trained"):
                started=time.monotonic()
                probabilities[arm]=[]
                for i in indices:
                    p=score(model,tokens[i],answer,frozen=arm=="frozen")
                    probabilities[arm].append(p)
                    rows.append(dict(index=i,id=docs[i]["id"],rating=targets[i],arm=arm,
                        token_sha=digest(tokens[i]),probabilities=p.tolist()))
                probabilities[arm]=np.array(probabilities[arm])
                timing[arm]=time.monotonic()-started
                print(f"{tag} {arm} fit probes 128/128",flush=True)
            sensitivity=[]
            started=time.monotonic()
            for mode in ("title_only","abstract_only"):
                for i in indices[:32]:
                    changed=dict(docs[i])
                    changed["abstract" if mode=="title_only" else "title"]=""
                    ids=prompt(tokenizer,changed,config)[0]
                    p=score(model,ids,answer)
                    original=probabilities["trained"][indices.index(i)]
                    sensitivity.append(dict(index=i,mode=mode,probabilities=p.tolist(),
                        full_to_ablation_TV=float(abs(original-p).sum()/2)))
            blank=score(model,prompt(tokenizer,dict(docs[indices[0]],title="",abstract=""),config)[0],answer)
            timing["ablations"]=time.monotonic()-started
            prior=np.bincount([targets[i] for i in fit],minlength=6)[1:]/len(fit)
            fit_metrics={arm:metrics(y,p) for arm,p in probabilities.items()}
            fit_metrics["fit_prior"]=metrics(y,np.tile(prior,(len(y),1)))
            fit_metrics["empty_prompt"]=metrics(y,np.tile(blank,(len(y),1)))
            shuffled=[]
            for _ in range(200):
                shuffled.append(metrics(rng.permutation(y),probabilities["trained"]))
            # One fit-only backward, without an optimizer or checkpoint mutation.
            model.train()
            model.zero_grad(set_to_none=True)
            i=indices[0]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=five_logits(model,torch.tensor([tokens[i]],device="cuda"),answer,train=True)
                loss=F.cross_entropy(logits,torch.tensor([targets[i]-1],device="cuda"))
            loss.backward()
            grads=[(name,p.grad) for name,p in model.named_parameters() if p.requires_grad]
            assert grads and all("lora_" in name for name,_ in grads)
            assert all(p.grad is None for name,p in model.named_parameters() if "lora_" not in name)
            assert all(g is not None and torch.isfinite(g).all() for _,g in grads)
            grad_norm=float(torch.sqrt(sum(g.float().square().sum() for _,g in grads)))
            assert np.isfinite(grad_norm) and grad_norm>0
            summary=dict(fit_metrics=fit_metrics,empty_probabilities=blank.tolist(),
                fit_saved_parity_max=max(parity),loading_seconds=loading_seconds,
                scoring_seconds=timing,gradient_probe=dict(index=i,loss=float(loss.detach()),
                    LoRA_tensors_with_grad=len(grads),LoRA_tensors_nonzero=sum(bool(g.count_nonzero()) for _,g in grads),
                    grad_norm=grad_norm,base_gradients_absent=True,optimizer_steps=0),
                mean_full_to_title_only_TV=float(np.mean([r["full_to_ablation_TV"] for r in sensitivity if r["mode"]=="title_only"])),
                mean_full_to_abstract_only_TV=float(np.mean([r["full_to_ablation_TV"] for r in sensitivity if r["mode"]=="abstract_only"])),
                shuffled_label_logloss_quantiles=np.quantile([r["rating_logloss"] for r in shuffled],[.025,.5,.975]).tolist(),
                shuffled_label_read_AUC_quantiles=np.quantile([r["read_AUC"] for r in shuffled],[.025,.5,.975]).tolist())
            write_json(dest/"summary.json",summary)
            (dest/"probabilities.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            (dest/"sensitivity.jsonl").write_text("".join(json.dumps(r)+"\n" for r in sensitivity))
            summaries[tag]=summary
            print(json.dumps(dict(partition=tag,summary=summary)),flush=True)
            del grads,model,logits,loss
            gc.collect()
            torch.cuda.empty_cache()
        assert fingerprint()==source_fp
        write_json(out/"summary.json",summaries)
        write_json(out/"status.json",dict(status="DIAGNOSTIC_COMPLETE",source_unchanged=True))
    except Exception as exc:
        write_json(out/"status.json",dict(status="FAILED",error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__=="__main__":
    main()
