"""Matched title/joint SPECTER2 heads with effective batch32 and microbatch16."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'specter2_preference'))
import modeling as old

SEED=old.SEED
ARMS=['title_adapted','joint_frozen','joint_adapted']
EPOCHS=[1,2,4,8]
BATCH=32
MICRO=16
MAX_LENGTH=512


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens_hash(tokens):
    return hashlib.sha256(json.dumps(dict(tokens),sort_keys=True).encode()).hexdigest()


def prepare(ctx,out):
    _,_,tokenizer=old.resources()
    data=ctx['data'];keys=(data.split+':'+data.id).to_numpy(dtype=str)
    modes={};rows=[]
    for mode in ['title','joint']:
        texts=(data.title+tokenizer.sep_token+(data.abstract if mode=='joint' else '')).tolist()
        raw=tokenizer(texts,truncation=False,return_token_type_ids=False)
        tokens=tokenizer(texts,truncation=True,max_length=MAX_LENGTH,return_token_type_ids=False)
        lengths=np.array([len(v) for v in raw['input_ids']])
        modes[mode]=dict(tokenizer=tokenizer,tokens=tokens,keys=keys,
            text_hash=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),
            token_hash=tokens_hash(tokens),lengths=lengths)
        for i,text in enumerate(texts):
            rows.append(dict(split=data.split.iloc[i],id=data.id.iloc[i],mode=mode,text=text,
                tokens=int(lengths[i]),truncated=bool(lengths[i]>MAX_LENGTH),
                abstract_available=bool(data.abstract.iloc[i])))
    ctx['modes']=modes
    out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(out/'input_audit.csv',index=False)
    stats={mode:dict(n=len(data),max_length=MAX_LENGTH,max_tokens=int(c['lengths'].max()),
        mean_tokens=float(c['lengths'].mean()),truncated=int((c['lengths']>MAX_LENGTH).sum()),
        token_sha256=c['token_hash'],text_sha256=c['text_hash']) for mode,c in modes.items()}
    (out/'input_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    return ctx


def mode_for(arm):
    return 'title' if arm=='title_adapted' else 'joint'


def tokens(ctx,indices):
    batch=ctx['tokenizer'].pad([{k:ctx['tokens'][k][i] for k in ctx['tokens']} for i in indices],return_tensors='pt')
    return {k:v.to('cuda') for k,v in batch.items()}


def embedding(model,mode,indices):
    if model is None:
        return torch.from_numpy(mode['vectors'][indices]).to('cuda')
    return F.normalize(model(**tokens(mode,indices)).last_hidden_state[:,0].float(),dim=1)


def prepare_vectors(ctx,out,protocol_hash):
    for mode,value in ctx['modes'].items():
        target=out/f'{mode}_vectors.npz'
        if target.exists():
            z=np.load(target)
            assert str(z['token_sha256'])==value['token_hash']
            assert str(z['protocol_sha256'])==protocol_hash
            np.testing.assert_array_equal(z['keys'],value['keys'])
            value['vectors']=z['vectors']
            continue
        started=time.monotonic();encoder=old.encoder(False)
        vectors=np.zeros((len(value['keys']),768),np.float32)
        order=np.argsort([len(v) for v in value['tokens']['input_ids']],kind='stable')
        with torch.inference_mode():
            for start in range(0,len(order),MICRO):
                ix=order[start:start+MICRO]
                with torch.autocast('cuda',dtype=torch.float16):
                    vectors[ix]=embedding(encoder,value,ix).cpu().numpy()
        assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        np.savez_compressed(target,vectors=vectors,keys=value['keys'],token_sha256=value['token_hash'],protocol_sha256=protocol_hash)
        value['vectors']=vectors
        del encoder;torch.cuda.empty_cache()
        print('ENCODED',mode,round(time.monotonic()-started,1),'seconds',flush=True)


def predict(model,head,mode,indices):
    result=np.zeros((len(indices),5),np.float32)
    order=np.argsort([len(mode['tokens']['input_ids'][i]) for i in indices],kind='stable')
    head.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),MICRO):
            loc=order[start:start+MICRO];ix=indices[loc]
            with torch.autocast('cuda',dtype=torch.float16):
                logits=head(embedding(model,mode,ix)).float()
            result[loc]=torch.softmax(logits,dim=1).cpu().numpy()
    assert np.isfinite(result).all() and np.allclose(result.sum(axis=1),1,atol=1e-6)
    return result


def fit(ctx,fit,pred,arm,epochs,prefix,protocol_hash,include_train=False,save_dir=None):
    prefix=Path(prefix);path=prefix.with_suffix('.npz');meta_path=prefix.with_suffix('.json')
    if path.exists():
        z=np.load(path)
        assert str(z['protocol_sha256'])==protocol_hash
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['pred'],pred)
        np.testing.assert_array_equal(z['epochs'],epochs)
        if save_dir is not None:assert (save_dir/'manifest.json').exists()
        return {e:z[f'epoch{e}'] for e in epochs},json.loads(meta_path.read_text()),z['train'] if include_train else None
    assert set(fit).isdisjoint(pred) and set(fit)<=set(range(2494))
    if np.all(pred<2494):
        assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][pred])
    else:
        assert save_dir is not None
    started=time.monotonic();adapted=arm!='joint_frozen'
    model=old.encoder(True) if adapted else None
    mode=ctx['modes'][mode_for(arm)]
    torch.manual_seed(SEED);head=old.Head().to('cuda')
    initial_head=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()
    params=[] if model is None else [p for p in model.parameters() if p.requires_grad]
    initial=[p.detach().clone() for p in params]
    parameter_groups=[dict(params=list(head.parameters()),lr=.001)]
    if params:parameter_groups.append(dict(params=params,lr=.00005))
    optimizer=torch.optim.AdamW(parameter_groups,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    rng=np.random.default_rng(SEED);predictions={};trace=[];nonzero=overflows=0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,max(epochs)+1):
        head.train();permutation=rng.permutation(fit);losses=[]
        for start in range(0,len(permutation),BATCH):
            batch=permutation[start:start+BATCH];optimizer.zero_grad(set_to_none=True)
            logical_loss=0.
            for micro_start in range(0,len(batch),MICRO):
                ix=batch[micro_start:micro_start+MICRO]
                targets=torch.tensor(ctx['y'][ix],device='cuda',dtype=torch.long)
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=old.loss(head(embedding(model,mode,ix)).float(),targets)
                assert torch.isfinite(loss)
                weighted=loss*(len(ix)/len(batch))
                scaler.scale(weighted).backward();logical_loss+=float(weighted.detach())
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],1.)
            if not bool(torch.isfinite(norm)):
                overflows+=1
            elif params:
                assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in params)
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                nonzero+=1
            scaler.step(optimizer);scaler.update();losses.append(logical_loss)
        if epoch in epochs:predictions[epoch]=predict(model,head,mode,pred)
        trace.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.monotonic()-started))
        print(prefix.name,'epoch',epoch,'loss',round(trace[-1]['loss'],4),'seconds',round(trace[-1]['seconds'],1),flush=True)
    train=predict(model,head,mode,fit) if include_train else None
    drift=float(sum((p.detach()-a).square().sum().item() for p,a in zip(params,initial))**.5)
    assert (drift>0)==adapted
    info=dict(arm=arm,epochs=epochs,seed=SEED,fit=fit.tolist(),pred=pred.tolist(),
        trained_encoder_parameters=sum(p.numel() for p in params),adapter_l2_drift=drift,
        nonzero_adapter_gradient_steps=nonzero,skipped_overflow_steps=overflows,
        initial_head_sha256=initial_head,trace=trace,seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=protocol_hash)
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True,parents=True)
        torch.save(head.cpu().state_dict(),save_dir/'head.pt')
        if model is not None:model.save_adapter(str(save_dir/'adapter'),'preference')
        (save_dir/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    values={f'epoch{e}':p for e,p in predictions.items()}
    if train is not None:values['train']=train
    np.savez_compressed(path,fit=fit,pred=pred,epochs=epochs,protocol_sha256=protocol_hash,**values)
    meta_path.write_text(json.dumps(info,indent=2)+'\n')
    del model,head,optimizer,params,initial
    torch.cuda.empty_cache()
    return predictions,info,train


def reload_predict(ctx,indices,directory):
    info=json.loads((directory/'manifest.json').read_text())
    model=None
    if info['arm']!='joint_frozen':
        model=old.encoder(False)
        model.delete_adapter('preference')
        model.load_adapter(str(directory/'adapter'),load_as='preference',set_active=True)
        model.to('cuda').eval()
    head=old.Head();head.load_state_dict(torch.load(directory/'head.pt',map_location='cpu',weights_only=True))
    head.to('cuda').eval()
    result=predict(model,head,ctx['modes'][mode_for(info['arm'])],indices)
    del head,model;torch.cuda.empty_cache()
    return result


def preflight(ctx,out):
    """Memory/gradient test on training inputs only, without heldout scores."""
    mode=ctx['modes']['joint']
    longest=np.argsort(mode['lengths'][:2494],kind='stable')[-MICRO:]
    model=old.encoder(True);torch.manual_seed(SEED);head=old.Head().to('cuda')
    base_name,base=next((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    before=base.detach().clone();started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.float16):
        logits=head(embedding(model,mode,longest)).float()
        value=old.loss(logits,torch.tensor(ctx['y'][longest],device='cuda',dtype=torch.long))
    value.backward()
    trained=[p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in trained)
    assert torch.equal(base,before) and torch.isfinite(value)
    result=dict(status='PASSED',max_length=MAX_LENGTH,microbatch=MICRO,effective_batch=BATCH,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        longest_batch_seconds=time.monotonic()-started,
        encoder_trainable_parameters=sum(p.numel() for p in trained),base_unchanged=True,
        note='No outer or calibration model score computed')
    (out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
