#!/usr/bin/env python3
"""Fixed-protocol frozen/learned adapter and random/hard-pair preference controls."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'input_ablation'))
import run_ablation as ab
from run_diagnostics import MINILM_REV
REPO='sentence-transformers/all-MiniLM-L6-v2'
SEED=20261016
SECOND_SEED=20261017
ARMS=['frozen','adapter_random','adapter_hard']
EPOCHS=3
PAIR_BATCH=16
MAX_LENGTH=256
PROB_THRESHOLDS=np.arange(0,.951,.05)


def setup_model(seed):
    from adapters import AutoAdapterModel,AdapterConfig
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    torch.manual_seed(seed)
    path=snapshot_download(REPO,revision=MINILM_REV,allow_patterns=['*.json','vocab.txt','model.safetensors'])
    tokenizer=AutoTokenizer.from_pretrained(path)
    model=AutoAdapterModel.from_pretrained(path)
    model.add_adapter('preference',config=AdapterConfig.load('seq_bn',reduction_factor=16))
    model.train_adapter('preference');model.set_active_adapters('preference')
    model.to('cuda').eval()
    assert all('adapters.preference' in name for name,p in model.named_parameters() if p.requires_grad)
    return model,tokenizer


def prepare(data):
    OUT.mkdir(exist_ok=True,parents=True)
    protocol=dict(arms=ARMS,seed=SEED,stability_seed=SECOND_SEED,epochs=EPOCHS,pair_batch=PAIR_BATCH,
        repo=REPO,revision=MINILM_REV,max_length=MAX_LENGTH,adapter='seq_bn reduction16',
        input='Mode/weight chosen by stage1 inner CV per outer fold; MiniLM mean pooling. Missing abstract block exactly zero. No grade/label in tokens.',
        architecture='Same initial adapter in all arms. Frozen arm freezes adapter too. Base transformer weights always frozen; all encoder dropout disabled. MLP head dim64 dropout0.1.',
        loss='BCE(read) + 0.25*MSE(conditional gain | positive) + 0.25*softplus(-(read_logit_positive-read_logit_negative))',
        training='Each epoch samples each positive once, paired with a negative. Random arms sample uniform negative; hard arm samples among5 closest training negatives of same venue when available. No mining on validation labels.',
        optimizer='AdamW head lr0.003, adapter lr0.0001, weight_decay0.01, clip_grad_norm1, fp16 autocast+GradScaler',
        priority='10*sigmoid(read_logit)*(0.25+0.75*sigmoid(gain_logit)); select by read probability threshold chosen from mean inner H',
        thresholds=PROB_THRESHOLDS.tolist(),
        evaluation='Same 5outer membership. Fixed training schedule with no hyperparameter search; use saved inner_fold0 as held-out calibration for all three arms and select cutoff by calibration H. Secondary seed outer-only refits use primary-seed calibration cutoffs held fixed; stability diagnostic only.',
        calibration_revision='Use one held-out calibration group instead of four full neural refits per outer fold; decided before any neural outer score was inspected. Other preflight inner runs archived and unused.',
        data_sha256=ab.baseline.input_digest(data),
        split_sha256=hashlib.sha256((ab.V2/'nested_splits.csv').read_bytes()).hexdigest(),
        caution='Different backbone/max length from SPECTER reference, so causal adaptation claim uses only matched neural arms. Short fixed training schedule; a negative result does not rule out all fine-tuning.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    # Cache the immutable frozen adapter's embeddings independently of labels.
    token_cache={};frozen_cache={}
    for seed in [SEED,SECOND_SEED]:
        path=OUT/f'frozen_{seed}.npz'
        model,tokenizer=setup_model(seed)
        sequences={
            'title':(data.title+tokenizer.sep_token).tolist(),
            'abstract':(tokenizer.sep_token+data.abstract).tolist(),
            'joint':(data.title+tokenizer.sep_token+data.abstract).tolist()}
        for field,texts in sequences.items():
            if seed==SEED:token_cache[field]=tokenizer(texts,truncation=True,max_length=MAX_LENGTH,padding=False,return_token_type_ids=False)
        if path.exists():
            cache=np.load(path);frozen_cache[seed]={k:cache[k] for k in sequences}
        else:
            arrays={}
            for field in sequences:
                tokens=token_cache[field]
                indices=np.arange(len(data)) if field!='abstract' else np.flatnonzero(data.abstract.str.len().gt(0))
                indices=indices[np.argsort([len(tokens['input_ids'][i]) for i in indices],kind='stable')]
                vectors=np.zeros((len(data),384),dtype=np.float32)
                for start in range(0,len(indices),64):
                    ix=indices[start:start+64];batch=tokenizer.pad([{k:tokens[k][i] for k in tokens} for i in ix],return_tensors='pt')
                    batch={k:v.to('cuda') for k,v in batch.items()}
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        hidden=model(**batch).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
                        value=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp(min=1),dim=1)
                    vectors[ix]=value.cpu().numpy()
                arrays[field]=vectors
            np.savez_compressed(path,**arrays,keys=(data.split+':'+data.id).to_numpy(dtype=str))
            frozen_cache[seed]=arrays
        del model;torch.cuda.empty_cache()
    return tokenizer,token_cache,frozen_cache


class Head(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,2))
    def forward(self,x):return self.net(10*x)


def field_representation(arrays,mode,weight,indices):
    if mode!='separate':return arrays[mode][indices]
    a=arrays['title'][indices];b=arrays['abstract'][indices]*weight
    value=np.hstack([a,b]);return value/np.maximum(np.linalg.norm(value,axis=1,keepdims=True),1e-12)


def hard_candidates(fit,y,representations,venues,k=5):
    positives=fit[y[fit]>1];negatives=fit[y[fit]==1]
    sims=representations[positives]@representations[negatives].T
    result={}
    for i,positive in enumerate(positives):
        same=venues[negatives]==venues[positive]
        scores=sims[i].copy()
        if same.any():scores[~same]=-np.inf
        order=np.argsort(-scores,kind='stable')
        good=order[np.isfinite(scores[order])][:k]
        result[int(positive)]=negatives[good]
    return result


def preference_loss(logits,labels):
    positive=labels>1
    read=F.binary_cross_entropy_with_logits(logits[:,0],positive.float())
    gain=.25+.75*torch.sigmoid(logits[:,1])
    conditional=F.mse_loss(gain[positive],(labels[positive]-1)/4)
    n=labels.numel()//2
    rank=F.softplus(-(logits[:n,0]-logits[n:,0])).mean()
    return read+.25*conditional+.25*rank


def fit_one(data,y,fit,pred,base,arm,seed,tokenizer,tokens,frozen,save_dir=None):
    mode=base['mode'];weight=base.get('weight') or 1.
    full_indices=np.arange(len(data));fixed=field_representation(frozen[seed],mode,weight,full_indices)
    width=fixed.shape[1];torch.manual_seed(seed)
    head=Head(width).to('cuda')
    model=None
    if arm!='frozen':
        model,_=setup_model(seed)
        # Re-create the head after adapter initialization: same initialization for every arm.
    torch.manual_seed(seed);head=Head(width).to('cuda')
    parameters=[{'params':list(head.parameters()),'lr':.003}]
    if model is not None:parameters.append({'params':[p for p in model.parameters() if p.requires_grad],'lr':.0001})
    optimizer=torch.optim.AdamW(parameters,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler()
    train_positive=fit[y[fit]>1];train_negative=fit[y[fit]==1]
    assert len(train_positive) and len(train_negative)
    hard=hard_candidates(fit,y,fixed,data.venue.to_numpy()) if arm=='adapter_hard' else None
    if hard is not None:assert all(set(v).issubset(set(fit)) for v in hard.values())
    rng=np.random.default_rng(seed)
    has_abstract=data.abstract.str.len().gt(0).to_numpy()
    def embed(indices):
        if model is None:return torch.from_numpy(fixed[indices]).to('cuda')
        def field(which):
            tok=tokens[which]
            batch=tokenizer.pad([{k:tok[k][i] for k in tok} for i in indices],return_tensors='pt')
            batch={k:v.to('cuda') for k,v in batch.items()}
            hidden=model(**batch).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
            result=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp(min=1),dim=1)
            if which=='abstract':result=result*torch.from_numpy(has_abstract[indices]).to('cuda').unsqueeze(-1)
            return result
        if mode!='separate':return field(mode)
        return F.normalize(torch.cat([field('title'),weight*field('abstract')],dim=1),dim=1)
    trace=[];started=time.monotonic()
    for epoch in range(EPOCHS):
        head.train();order=rng.permutation(train_positive);losses=[]
        for start in range(0,len(order),PAIR_BATCH):
            pi=order[start:start+PAIR_BATCH]
            ni=np.array([rng.choice(hard[int(p)]) for p in pi]) if hard is not None else rng.choice(train_negative,size=len(pi),replace=True)
            ix=np.concatenate([pi,ni]);target=torch.from_numpy(y[ix].astype(np.float32)).to('cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):logits=head(embed(ix));loss=preference_loss(logits,target)
            assert torch.isfinite(loss),'Nonfinite loss'
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],1.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
    head.eval();outputs=[]
    # Sort by sequence length for inference, then restore the requested index order.
    lengths=np.array([len(tokens['joint']['input_ids'][i]) for i in pred]);order=np.argsort(lengths,kind='stable')
    result=np.zeros((len(pred),2),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(pred),64):
            offsets=order[start:start+64];ix=pred[offsets]
            with torch.autocast('cuda',dtype=torch.float16):raw=head(embed(ix)).float()
            probability=torch.sigmoid(raw[:,0]);gain=probability*(.25+.75*torch.sigmoid(raw[:,1]))
            result[offsets,0]=probability.cpu().numpy();result[offsets,1]=gain.cpu().numpy()
    info=dict(loss_by_epoch=trace,elapsed_seconds=time.monotonic()-started,seed=seed,arm=arm,
        mode=mode,weight=weight,fit_ids_sha256=hashlib.sha256('\n'.join(data.id.iloc[fit]).encode()).hexdigest(),
        trained_encoder_parameters=0 if model is None else sum(p.numel() for p in model.parameters() if p.requires_grad))
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True,parents=True)
        torch.save(head.cpu().state_dict(),save_dir/'head.pt')
        if model is not None:model.save_adapter(str(save_dir/'adapter'),'preference')
        (save_dir/'manifest.json').write_text(json.dumps(dict(info,width=width,base=base),indent=2)+'\n')
    del model,head,optimizer;torch.cuda.empty_cache()
    assert np.isfinite(result).all()
    return result,info


def make_priority(values,threshold):
    return np.where(values[:,0]>threshold,10*values[:,1],0.)


def predict_saved(data,indices,directory,tokenizer,tokens,frozen):
    info=json.loads((directory/'manifest.json').read_text())
    base=info['base'];mode=base['mode'];weight=base.get('weight') or 1.
    model=None
    head=Head(info['width']);head.load_state_dict(torch.load(directory/'head.pt',map_location='cpu',weights_only=True));head.to('cuda').eval()
    fixed=field_representation(frozen[info['seed']],mode,weight,np.arange(len(data)))
    if info['arm']!='frozen':
        model,_=setup_model(info['seed']);model.delete_adapter('preference')
        model.load_adapter(str(directory/'adapter'),load_as='preference',set_active=True)
        model.to('cuda').eval()
    has_abstract=data.abstract.str.len().gt(0).to_numpy()
    def field(which,ix):
        tok=tokens[which];batch=tokenizer.pad([{k:tok[k][i] for k in tok} for i in ix],return_tensors='pt')
        batch={k:v.to('cuda') for k,v in batch.items()}
        hidden=model(**batch).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
        result=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp(min=1),dim=1)
        return result*torch.from_numpy(has_abstract[ix]).to('cuda').unsqueeze(-1) if which=='abstract' else result
    order=np.argsort([len(tokens['joint']['input_ids'][i]) for i in indices],kind='stable')
    result=np.zeros((len(indices),2),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(indices),64):
            offsets=order[start:start+64];ix=indices[offsets]
            with torch.autocast('cuda',dtype=torch.float16):
                if model is None:representation=torch.from_numpy(fixed[ix]).to('cuda')
                elif mode!='separate':representation=field(mode,ix)
                else:representation=F.normalize(torch.cat([field('title',ix),weight*field('abstract',ix)],dim=1),dim=1)
                raw=head(representation).float()
            prob=torch.sigmoid(raw[:,0]);gain=prob*(.25+.75*torch.sigmoid(raw[:,1]))
            result[offsets,0]=prob.cpu().numpy();result[offsets,1]=gain.cpu().numpy()
    del model,head;torch.cuda.empty_cache()
    return result


def choose(data,y,fit,nums,pred):
    choices={}
    for arm,values in pred.items():
        best=None
        for cutoff in PROB_THRESHOLDS:
            p=make_priority(values,cutoff)
            scores=[ab.evaluate(y[fit[nums==k]],p[nums==k],data.id.iloc[fit[nums==k]])['score'] for k in np.unique(nums)]
            row=dict(arm=arm,threshold=float(cutoff),inner_score=float(np.mean(scores)))
            if best is None or row['inner_score']>best['inner_score']:best=row
        choices[arm]=best
    return choices


def run():
    data,_=ab.load_clean_data();tokenizer,tokens,frozen=prepare(data)
    y=np.full(len(data),np.nan);y[:2494]=data.Label.iloc[:2494].astype(int)
    groups=ab.baseline.grouping(data.iloc[:2494].reset_index(drop=True))
    split=pd.read_csv(ab.V2/'nested_splits.csv',dtype={'id':str});lookup=dict(zip(data.id.iloc[:2494],range(2494)))
    search=pd.read_csv(ab.OUT/'inner_search.csv');configs={c['key']:c for c in ab.configurations()}
    oof=data.iloc[:2494][['id','title','venue','year','Label']].copy();oof['outer_fold']=-1
    metrics=[];selections=[];traces=[];started=time.monotonic()
    for fold in range(5):
        best=search[search.outer_fold.eq(fold)].sort_values('inner_score',ascending=False,kind='stable').iloc[0]
        base=configs[best['key']]
        s=split[split.outer_fold.eq(fold)];fr=s[s.partition.eq('outer_train')];vr=s[s.partition.eq('outer_validation')]
        fit=np.array([lookup[x] for x in fr.id]);val=np.array([lookup[x] for x in vr.id]);nums=fr.inner_fold.to_numpy()
        assert set(groups[fit]).isdisjoint(groups[val])
        inner={arm:np.zeros((len(fit),2),dtype=np.float32) for arm in ARMS}
        pred={}
        for arm in ARMS:
            for k in [0]:
                mask=nums==k;fi,vi=fit[~mask],fit[mask]
                assert set(groups[fi]).isdisjoint(groups[vi])
                path=OUT/f'outer{fold}_inner{k}_{arm}.npz'
                if path.exists():values=np.load(path)['values']
                else:
                    values,info=fit_one(data,y,fi,vi,base,arm,SEED,tokenizer,tokens,frozen)
                    np.savez_compressed(path,values=values,fit=fi,val=vi)
                    (path.with_suffix('.json')).write_text(json.dumps(info,indent=2)+'\n')
                inner[arm][mask]=values
            path=OUT/f'outer{fold}_{arm}.npz'
            if path.exists():pred[arm]=np.load(path)['values']
            else:
                values,info=fit_one(data,y,fit,val,base,arm,SEED,tokenizer,tokens,frozen)
                pred[arm]=values;np.savez_compressed(path,values=values,fit=fit,val=val)
                path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
            print(f'Neural outer{fold+1}/5 {arm} done, elapsed={time.monotonic()-started:.1f}s',flush=True)
        calibration=nums==0
        selected=choose(data,y,fit[calibration],nums[calibration],{arm:v[calibration] for arm,v in inner.items()})
        selections.extend(dict(outer_fold=fold,**x) for x in selected.values())
        for arm,c in selected.items():
            values=pred[arm];p=make_priority(values,c['threshold'])
            result=ab.evaluate(y[val],p,data.id.iloc[val])
            metrics.append(dict(outer_fold=fold,family=arm,seed=SEED,**c,**result,
                read_auc=roc_auc_score(y[val]>1,values[:,0]),
                missed_high=int(((y[val]>=4)&(p==0)).sum())))
            oof.loc[val,arm+'_prob']=values[:,0];oof.loc[val,arm+'_gain']=values[:,1];oof.loc[val,arm+'_priority']=p
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(selections).to_csv(OUT/'inner_choices.csv',index=False)
        oof.to_csv(OUT/'oof_predictions.csv',index=False)
        print(pd.DataFrame(metrics).query('outer_fold==@fold')[['family','score','FP','FN']].to_string(index=False),flush=True)
    # Check initialization sensitivity without a second hyperparameter search.
    stability=[]
    for fold in range(5):
        best=search[search.outer_fold.eq(fold)].sort_values('inner_score',ascending=False,kind='stable').iloc[0];base=configs[best['key']]
        s=split[split.outer_fold.eq(fold)];fit=np.array([lookup[x] for x in s[s.partition.eq('outer_train')].id]);val=np.array([lookup[x] for x in s[s.partition.eq('outer_validation')].id])
        for arm in ARMS:
            path=OUT/f'stability_outer{fold}_{arm}.npz'
            if path.exists():values=np.load(path)['values']
            else:
                values,info=fit_one(data,y,fit,val,base,arm,SECOND_SEED,tokenizer,tokens,frozen)
                np.savez_compressed(path,values=values,fit=fit,val=val);path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
            cutoff=next(c['threshold'] for c in selections if c['outer_fold']==fold and c['arm']==arm)
            p=make_priority(values,cutoff)
            stability.append(dict(outer_fold=fold,family=arm,seed=SECOND_SEED,threshold=cutoff,**ab.evaluate(y[val],p,data.id.iloc[val])))
        pd.DataFrame(stability).to_csv(OUT/'stability_metrics.csv',index=False)
    m=pd.DataFrame(metrics)
    m.groupby('family').agg(H=('score','mean'),H_std=('score','std'),Q=('relevance','mean'),R=('coverage','mean'),
        FP=('FP','sum'),FN=('FN','sum'),missed_high=('missed_high','sum')).to_csv(OUT/'summary.csv')
    (OUT/'run_manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
        primary_seed=SEED,stability_seed=SECOND_SEED,original_labels_preserved=True),indent=2)+'\n')
    print(pd.read_csv(OUT/'summary.csv').to_string(index=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):run()
