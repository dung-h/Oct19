"""Fixed fit-internal controls for Qwen representation, head, and training length."""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'qwen_remote_v1'
sys.path.insert(0,str(PILOT))
from common import check_probability, digest, fingerprint, jsonl, load_inputs, official, prompt, sha, write_json
from model_core import five_logits, load_model, load_tokenizer, save_adapter


def runtime():
    packages=('torch','transformers','peft','accelerate','bitsandbytes','numpy','scikit-learn','safetensors','huggingface-hub')
    return dict(source=fingerprint(),code={n:sha(HERE/n) for n in ('run.py','config.json','PROTOCOL.md','run_check.sh')},
        versions={n:importlib.metadata.version(n) for n in packages},python=platform.python_version(),
        cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        allocator={k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')})


def score(model,tokens,answer,indices):
    model.eval()
    values=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in indices:
            z=five_logits(model,torch.tensor([tokens[i]],device='cuda'),answer,train=False)
            values.append(z.softmax(-1).squeeze(0).cpu().numpy().astype(float))
            torch.cuda.empty_cache()
    return check_probability(values,len(indices))


def prepare(root,cfg,docs,targets,parts):
    if root.exists():
        raise RuntimeError('Output exists; retain artifacts and choose a fresh namespace')
    pool=parts[cfg['pool_partition']]['fit']
    groups=np.array([docs[i]['group'] for i in pool])
    train_pos,val_pos=next(GroupShuffleSplit(n_splits=1,test_size=cfg['validation_group_fraction'],random_state=cfg['seed']).split(pool,groups=groups))
    train=[pool[int(i)] for i in train_pos]
    val=[pool[int(i)] for i in val_pos]
    probes=np.random.default_rng(cfg['seed']+1).choice(train,cfg['train_probes'],replace=False).tolist()
    assert set(train).isdisjoint(val) and set(train+val)==set(pool)
    assert set(pool).isdisjoint(parts[cfg['pool_partition']]['pred'])
    assert {docs[i]['group'] for i in train}.isdisjoint({docs[i]['group'] for i in val})
    assert {targets[i] for i in train}=={1,2,3,4,5}
    root.mkdir(parents=True)
    write_json(root/'runtime_lock.json',runtime())
    write_json(root/'membership.json',dict(pool=pool,train=train,validation=val,train_probes=probes,
        seed=cfg['seed'],validation_labels_used_for_fitting=False))
    print(json.dumps(dict(train=len(train),validation=len(val),train_groups=len(set(groups[train_pos])),validation_groups=len(set(groups[val_pos])))),flush=True)


def train(root,cfg,base,docs,targets,members,smoke=False):
    dest=root/('smoke' if smoke else 'training')
    if dest.exists():
        raise RuntimeError('Training namespace exists; optimizer resume is unsupported')
    if not smoke:
        assert json.loads((root/'smoke/COMPLETE.json').read_text())['status']=='TECHNICAL_PASS'
    dest.mkdir()
    tokenizer,answer=load_tokenizer(base)
    fit=members['train']
    anchors=[next(i for i in fit if targets[i]==label) for label in range(1,6)]
    required=anchors if smoke else sorted(set(fit+members['validation']))
    tokens={i:prompt(tokenizer,docs[i],base)[0] for i in required}
    audit={str(i):digest(tokens[i]) for i in required}
    write_json(dest/'token_audit.json',audit)
    started=time.monotonic()
    model,_=load_model(base,trainable=True,seed=cfg['seed'])
    load_seconds=time.monotonic()-started
    torch.cuda.empty_cache()
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert parameters and all('lora_' in n for n,_ in parameters)
    before={n:p.detach().cpu().clone() for n,p in parameters}
    tc=base['training']; epochs=1 if smoke else cfg['epochs']; accumulation=5 if smoke else tc['gradient_accumulation_steps']
    count=10 if smoke else len(fit)
    updates_per_epoch=math.ceil(count/accumulation)
    total=epochs*updates_per_epoch
    opt=torch.optim.AdamW([p for _,p in parameters],lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=math.ceil(total*tc['warmup_ratio']),num_training_steps=total)
    step=0;train_seconds=0;score_seconds=0;checkpoint_hashes={}
    opt.zero_grad(set_to_none=True)
    with (dest/'training.jsonl').open('x') as log:
        for epoch in range(1,epochs+1):
            order=anchors*2 if smoke else np.random.default_rng(cfg['seed']+epoch).permutation(fit).tolist()
            model.train(); epoch_start=time.monotonic()
            for start in range(0,len(order),accumulation):
                batch=order[start:start+accumulation]; losses=[]; update_start=time.monotonic()
                for i in batch:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        z=five_logits(model,torch.tensor([tokens[i]],device='cuda'),answer,train=True)
                        loss=F.cross_entropy(z,torch.tensor([targets[i]-1],device='cuda'))
                    assert torch.isfinite(loss)
                    (loss/len(batch)).backward(); losses.append(float(loss.detach()))
                assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
                assert any(p.grad is not None for _,p in parameters)
                norm=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],tc['max_grad_norm'])
                assert torch.isfinite(norm)
                opt.step();sched.step();opt.zero_grad(set_to_none=True);step+=1
                row=dict(epoch=epoch,step=step,total_steps=total,loss=float(np.mean(losses)),grad_norm=float(norm),
                    lr=sched.get_last_lr()[0],seconds=time.monotonic()-update_start,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step%10==0 or step==total or smoke: print(json.dumps(row),flush=True)
                torch.cuda.empty_cache()
            train_seconds+=time.monotonic()-epoch_start
            adapter=dest/f'epoch{epoch}_adapter'
            save_adapter(model,adapter)
            checkpoint_hashes[str(epoch)]={p.name:sha(p) for p in adapter.iterdir() if p.is_file()}
            if not smoke:
                begun=time.monotonic()
                vp=score(model,tokens,answer,members['validation'])
                tp=score(model,tokens,answer,members['train_probes'])
                np.savez_compressed(dest/f'epoch{epoch}_predictions.npz',validation=vp,train_probes=tp)
                score_seconds+=time.monotonic()-begun
                print(f'EPOCH {epoch}/{epochs} predictions saved; no validation metrics inspected',flush=True)
    assert any(not torch.equal(before[n],p.detach().cpu()) for n,p in parameters)
    pre=score(model,tokens,answer,anchors)
    del model,opt,sched,parameters,loss,z,norm
    gc.collect();torch.cuda.empty_cache()
    begun=time.monotonic()
    model,_=load_model(base,trainable=True,seed=cfg['seed'],adapter_path=dest/f'epoch{epochs}_adapter')
    reload_seconds=time.monotonic()-begun
    post=score(model,tokens,answer,anchors)
    error=float(abs(pre-post).max()); assert error<=1e-4
    write_json(dest/'COMPLETE.json',dict(status='TECHNICAL_PASS' if smoke else 'TRAINING_COMPLETE',epochs=epochs,
        steps=step,reload_max_error=error,loading_seconds=load_seconds,reloading_seconds=reload_seconds,
        training_seconds=train_seconds,scoring_seconds=score_seconds,checkpoint_hashes=checkpoint_hashes,
        adapter_changed=True,base_gradients_absent=True))
    print(f'{dest.name} complete parity={error}',flush=True)


def extract(root,cfg,base,docs,members):
    assert json.loads((root/'smoke/COMPLETE.json').read_text())['status']=='TECHNICAL_PASS'
    if (root/'features.npz').exists(): raise RuntimeError('Features already exist')
    tok,answer=load_tokenizer(base)
    started=time.monotonic(); model,_=load_model(base,trainable=False,seed=cfg['seed'])
    loading=time.monotonic()-started;model.eval();torch.cuda.empty_cache()
    capture={}
    def hook(module,args):
        capture['h']=args[0][:,-1,:].detach().float().cpu().numpy().copy()
    handle=model.get_output_embeddings().register_forward_pre_hook(hook)
    hs=[];zs=[];texts=[];started=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for j,i in enumerate(members['pool']):
            ids,stats=prompt(tok,docs[i],base)
            z=five_logits(model,torch.tensor([ids],device='cuda'),answer,train=False).squeeze(0).cpu().numpy()
            h=capture.pop('h').squeeze(0)
            assert h.shape==(3584,) and z.shape==(5,) and np.isfinite(h).all() and np.isfinite(z).all()
            hs.append(h);zs.append(z)
            fields=[tok.decode(tok.encode(docs[i][name],add_special_tokens=False)[:base['input'][name+'_tokens']],skip_special_tokens=True).strip() for name in ('title','abstract')]
            texts.append(dict(index=i,text=fields[0]+'\n'+fields[1],token_sha=stats['token_sha256']))
            torch.cuda.empty_cache()
            if (j+1)%100==0:print(f'Frozen features {j+1}/{len(members["pool"])}',flush=True)
    handle.remove()
    np.savez_compressed(root/'features.npz',indices=members['pool'],hidden=np.array(hs),logits=np.array(zs))
    (root/'feature_texts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in texts))
    write_json(root/'EXTRACT_COMPLETE.json',dict(status='FEATURES_COMPLETE',loading_seconds=loading,
        scoring_seconds=time.monotonic()-started,features_sha=sha(root/'features.npz')))


def controls(root,cfg,targets,members):
    assert json.loads((root/'EXTRACT_COMPLETE.json').read_text())['status']=='FEATURES_COMPLETE'
    if (root/'CONTROLS_COMPLETE.json').exists():raise RuntimeError('Controls already exist')
    f=np.load(root/'features.npz');pos={int(i):j for j,i in enumerate(f['indices'])}
    ti=np.array([pos[i] for i in members['train']]);vi=np.array([pos[i] for i in members['validation']]);pi=np.array([pos[i] for i in members['train_probes']])
    y=np.array([targets[i] for i in members['train']]);zy=f['logits'][ti].astype(float)
    prior=np.bincount(y,minlength=6)[1:]/len(y)
    predictions={};metadata={};begin=time.monotonic()
    def retain(name,allp):
        check_probability(allp,len(members['pool']))
        predictions[name+'_validation']=allp[vi];predictions[name+'_train_probes']=allp[pi]
    retain('fit_prior',np.tile(prior,(len(pos),1)))
    retain('frozen_digit',softmax(f['logits'].astype(float),axis=1))
    def objective(b):
        z=zy+b;p=softmax(z,axis=1)
        ce=np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y-1])+.5*cfg['bias_L2']*np.dot(b,b)
        p[np.arange(len(y)),y-1]-=1
        return ce,p.mean(0)+cfg['bias_L2']*b
    initial=np.log(prior)-zy.mean(0);initial-=initial.mean()
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
    assert result.success,result.message
    retain('bias_only',softmax(f['logits'].astype(float)+result.x,axis=1))
    metadata['bias_only']=dict(bias=result.x.tolist(),objective=float(result.fun),iterations=int(result.nit))
    for name,labels in (('hidden_head',y),('shuffled_hidden_head',np.random.default_rng(cfg['seed']+2).permutation(y))):
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=cfg['linear_head_C'],max_iter=cfg['linear_head_max_iter'],solver='lbfgs'))
        clf.fit(f['hidden'][ti],labels)
        assert clf[-1].classes_.tolist()==[1,2,3,4,5]
        retain(name,clf.predict_proba(f['hidden']))
        joblib.dump(clf,root/(name+'.joblib'))
        metadata[name]=dict(iterations=clf[-1].n_iter_.tolist())
    text={r['index']:r['text'] for r in jsonl(root/'feature_texts.jsonl')}
    clf=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=cfg['tfidf_max_features'],sublinear_tf=True),
        LogisticRegression(C=cfg['linear_head_C'],max_iter=cfg['linear_head_max_iter'],solver='lbfgs'))
    clf.fit([text[i] for i in members['train']],y)
    retain('tfidf',clf.predict_proba([text[i] for i in members['pool']]))
    joblib.dump(clf,root/'tfidf.joblib')
    np.savez_compressed(root/'control_predictions.npz',**predictions)
    write_json(root/'CONTROLS_COMPLETE.json',dict(status='CONTROLS_COMPLETE',seconds=time.monotonic()-begin,fit=metadata,
        inner_train_class_prior=prior.tolist(),validation_labels_used=False))


def evaluate(root,cfg,docs,targets,members):
    import csv
    assert json.loads((root/'CONTROLS_COMPLETE.json').read_text())['status']=='CONTROLS_COMPLETE'
    assert json.loads((root/'training/COMPLETE.json').read_text())['status']=='TRAINING_COMPLETE'
    if (root/'decision.json').exists():raise RuntimeError('Evaluation already complete')
    data=np.load(root/'control_predictions.npz')
    probabilities={name:data[name] for name in data.files}
    for epoch in range(1,cfg['epochs']+1):
        pred=np.load(root/'training'/f'epoch{epoch}_predictions.npz')
        for split in ('validation','train_probes'):probabilities[f'lora_epoch{epoch}_{split}']=pred[split]
    arms=['fit_prior','frozen_digit','bias_only','hidden_head','shuffled_hidden_head','tfidf']+[f'lora_epoch{i}' for i in range(1,cfg['epochs']+1)]
    rows=[];lookup={}
    for split in ('train_probes','validation'):
        indices=members[split]; y=np.array([targets[i] for i in indices]); ids=[docs[i]['id'] for i in indices]
        for arm in arms:
            p=check_probability(probabilities[f'{arm}_{split}'],len(indices));p=p/p.sum(1,keepdims=True)
            read=1-p[:,0];g=p@(np.arange(5)/4)
            row=dict(split=split,arm=arm,n=len(y),rating_logloss=float(log_loss(y,p,labels=[1,2,3,4,5])),
                read_AUC=float(roc_auc_score(y>1,read)),read_Brier=float(np.mean((read-(y>1))**2)),
                **official(y,np.where(read>cfg['read_cutoff'],1+9*g,0),ids))
            rows.append(row);lookup[split,arm]=row
    with (root/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    y=np.array([targets[i] for i in members['validation']]);groups={}
    for j,i in enumerate(members['validation']):groups.setdefault(docs[i]['group'],[]).append(j)
    keys=list(groups);rng=np.random.default_rng(cfg['seed']+3);uncertainty={}
    for a,b in (('lora_epoch3','lora_epoch1'),('hidden_head','lora_epoch3'),('hidden_head','shuffled_hidden_head'),('tfidf','hidden_head')):
        pa=probabilities[a+'_validation'];pb=probabilities[b+'_validation'];deltas=[]
        for _ in range(cfg['bootstrap_repeats']):
            idx=np.concatenate([groups[keys[int(k)]] for k in rng.choice(len(keys),len(keys),replace=True)])
            yy=y[idx]
            if len(np.unique(yy>1))<2:continue
            ce=lambda p:float(-np.log(np.clip(p[idx,yy-1],1e-15,1)).mean())
            deltas.append([ce(pa)-ce(pb),roc_auc_score(yy>1,1-pa[idx,0])-roc_auc_score(yy>1,1-pb[idx,0])])
        uncertainty[a+'_minus_'+b]=dict(CE_delta_quantiles=np.quantile(np.array(deltas)[:,0],[.025,.5,.975]).tolist(),
            AUC_delta_quantiles=np.quantile(np.array(deltas)[:,1],[.025,.5,.975]).tolist())
    one=lookup['validation','lora_epoch1'];three=lookup['validation','lora_epoch3'];head=lookup['validation','hidden_head']
    decision=dict(interpretation='Exploratory fit-internal diagnostic; not promotion',
        extra_epochs_promising=bool(three['rating_logloss']<=one['rating_logloss']-.03 and three['read_AUC']>=one['read_AUC']),
        hidden_head_advantage=bool(head['rating_logloss']<=three['rating_logloss']-.03 and head['read_AUC']>=three['read_AUC']+.02),
        validation_CE_epoch3_minus_epoch1=three['rating_logloss']-one['rating_logloss'],
        validation_AUC_epoch3_minus_epoch1=three['read_AUC']-one['read_AUC'],bootstrap=uncertainty)
    write_json(root/'decision.json',decision)
    print(json.dumps(rows,indent=2));print(json.dumps(decision,indent=2))


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('stage',choices=['prepare','smoke','extract','controls','train','evaluate']);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.output.resolve();cfg=json.loads((HERE/'config.json').read_text())
    base=json.loads((PILOT/'config.json').read_text());docs,targets,parts,_=load_inputs()
    if args.stage=='prepare':prepare(root,cfg,docs,targets,parts);return
    assert json.loads((root/'runtime_lock.json').read_text())==runtime(),'Locked runtime/source changed; use new output and smoke'
    members=json.loads((root/'membership.json').read_text())
    try:
        if args.stage in ('smoke','train'):train(root,cfg,base,docs,targets,members,args.stage=='smoke')
        elif args.stage=='extract':extract(root,cfg,base,docs,members)
        elif args.stage=='controls':controls(root,cfg,targets,members)
        elif args.stage=='evaluate':evaluate(root,cfg,docs,targets,members)
    except Exception as exc:
        write_json(root/(args.stage+'_FAILED.json'),dict(error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':main()
