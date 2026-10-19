#!/usr/bin/env python3
"""Title/abstract ablation, with all choices made on the saved inner folds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'diagnose_bottleneck'))
import run_diagnostics as diag
from run_diagnostics import baseline, load_clean_data, evaluate, priority
V2 = diag.V2
ALPHAS = diag.ALPHAS
WEIGHTS = [.25,.5,1.,2.]
THRESHOLDS = diag.THRESHOLDS


def configurations():
    configs=[]
    for representation in ['tfidf','specter']:
        for mode in ['title','joint','separate']:
            for weight in WEIGHTS if mode=='separate' else [None]:
                for alpha in ALPHAS:
                    configs.append(dict(family=f'{representation}_{mode}',representation=representation,
                                        mode=mode,weight=weight,alpha=alpha,key=f'c{len(configs):03d}'))
    return configs


def protocol(data):
    value=dict(configurations=configurations(),thresholds=THRESHOLDS.tolist(),
        data_sha256=baseline.input_digest(data),
        split_sha256=hashlib.sha256((V2/'nested_splits.csv').read_bytes()).hexdigest(),
        encoder=dict(base=baseline.BASE,base_revision=baseline.BASE_REV,
                     adapter=baseline.ADAPTER,adapter_revision=baseline.ADAPTER_REV),
        embedding='CLS, fp16 GPU, max512. Title: title+SEP. Abstract block: SEP+abstract; missing blocks zero. Joint vectors reused verbatim from v2.',
        representation='L2 each block; separate [title,weight*abstract] then L2; same weights for dense and TFIDF. TFIDF vocab fit only in training partition.',
        model='Ridge lsqr tol1e-6, float64 normalized dense features',
        evaluation='Exact saved 5 outer/4 inner groups. Choose config and cutoff per family, plus global selection policy, by inner OOF H. No outer labels used to choose.',
        final='Select again by 4 grouped folds on all train, then fit all labels and predict test.',
        continuation='Metadata audit follows. Fine-tune comparison follows unless input+metadata improves H>=0.01, reduces FP by>=10%, preserves recall within0.01 and does not increase missed labels4/5 vs diagnostic baseline.',
        limitation='Adaptive research on repeatedly inspected labels; outer score is not a pristine independent benchmark.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==value,'Protocol changed'
    else:path.write_text(json.dumps(value,indent=2)+'\n')


def encode(data):
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    target=OUT/'field_embeddings.npz'
    digest=baseline.input_digest(data)
    if target.exists():
        assert json.loads((OUT/'embedding_manifest.json').read_text())['input_sha256']==digest
        return
    torch.manual_seed(20261016);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    base=snapshot_download(baseline.BASE,revision=baseline.BASE_REV,allow_patterns=['*.bin','*.json','vocab.txt'])
    adapter=snapshot_download(baseline.ADAPTER,revision=baseline.ADAPTER_REV,allow_patterns=['*.bin','*.json'])
    tokenizer=AutoTokenizer.from_pretrained(base)
    model=AutoAdapterModel.from_pretrained(base)
    model.load_adapter(adapter,load_as='regression',set_active=True)
    model.to(device).eval().requires_grad_(False)
    arrays={};stats={};started=time.monotonic()
    for field,texts in [('title',(data.title+tokenizer.sep_token).tolist()),
                         ('abstract',(tokenizer.sep_token+data.abstract).tolist())]:
        vectors=np.zeros((len(data),768),dtype=np.float32)
        indices=np.arange(len(data)) if field=='title' else np.flatnonzero(data.abstract.str.len().gt(0))
        lengths=np.array([len(x) for x in tokenizer([texts[i] for i in indices],truncation=False)['input_ids']])
        indices=indices[np.argsort(lengths,kind='stable')]
        for start in range(0,len(indices),12):
            ix=indices[start:start+12]
            batch=tokenizer([texts[i] for i in ix],padding=True,truncation=True,max_length=512,
                            return_tensors='pt',return_token_type_ids=False)
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.inference_mode(),torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):
                value=model(**batch).last_hidden_state[:,0,:].float()
            vectors[ix]=value.cpu().numpy()
        arrays[field]=vectors
        stats[field]=dict(encoded=len(indices),truncated=int((lengths>512).sum()))
        print('Encoded',field,stats[field],flush=True)
    arrays['keys']=(data.split+':'+data.id).to_numpy(dtype=str)
    np.savez_compressed(target,**arrays)
    (OUT/'embedding_manifest.json').write_text(json.dumps(dict(input_sha256=digest,
        stats=stats,elapsed_seconds=time.monotonic()-started,device=device,
        base_revision=baseline.BASE_REV,adapter_revision=baseline.ADAPTER_REV),indent=2)+'\n')


def load_features(data):
    cache=np.load(OUT/'field_embeddings.npz');joint=np.load(V2/'embeddings_clean.npz')
    keys=(data.split+':'+data.id).to_numpy(dtype=str)
    assert np.array_equal(cache['keys'],keys) and np.array_equal(joint['keys'],keys)
    assert json.loads((V2/'embedding_manifest.json').read_text())['input_sha256']==baseline.input_digest(data)
    return {key:normalize(cache[key].astype(np.float64)) for key in ['title','abstract']}|{
            'joint':normalize(joint['embeddings'].astype(np.float64))}


def vectorizer():
    return TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=50000,sublinear_tf=True)


def build_spaces(data,features,fit,pred):
    spaces={};vocab={}
    for mode,column in [('title','title'),('joint','text'),('abstract','abstract')]:
        vocab[mode]=vectorizer()
        a=vocab[mode].fit_transform(data[column].iloc[fit]);b=vocab[mode].transform(data[column].iloc[pred])
        spaces[('tfidf',mode)]=(a,b)
        spaces[('specter',mode)]=(features[mode][fit],features[mode][pred])
    return spaces,vocab


def space_for(config,spaces):
    rep,mode=config['representation'],config['mode']
    if mode!='separate':return spaces[(rep,mode)]
    weight=config['weight']
    if rep=='tfidf':
        return tuple(normalize(sparse.hstack([spaces[(rep,'title')][i],weight*spaces[(rep,'abstract')][i]],format='csr')) for i in [0,1])
    return tuple(normalize(np.hstack([spaces[(rep,'title')][i],weight*spaces[(rep,'abstract')][i]])) for i in [0,1])


def fit_predict(data,features,labels,fit,pred,configs):
    spaces,_=build_spaces(data,features,fit,pred)
    predictions={};train_mae={}
    for config in configs:
        xf,xp=space_for(config,spaces)
        model=Ridge(alpha=config['alpha'],solver='lsqr',tol=1e-6).fit(xf,labels[fit])
        predictions[config['key']]=model.predict(xp)
        train_mae[config['key']]=float(mean_absolute_error(labels[fit],model.predict(xf)))
    return predictions,train_mae


def select(data,labels,fit,fold_numbers,predictions,configs):
    chosen={};rows=[]
    for config in configs:
        best=None
        for cutoff in THRESHOLDS:
            p=priority(predictions[config['key']],cutoff)
            scores=[]
            for fold in range(4):
                mask=fold_numbers==fold
                scores.append(evaluate(labels[fit[mask]],p[mask],data.id.iloc[fit[mask]])['score'])
            row=dict(config,threshold=float(cutoff),inner_score=float(np.mean(scores)))
            if best is None or row['inner_score']>best['inner_score']:best=row
        rows.append(best)
        family=config['family']
        if family not in chosen or best['inner_score']>chosen[family]['inner_score']:chosen[family]=best
    chosen['selection_procedure']=max(rows,key=lambda x:x['inner_score'])
    return chosen,rows


def run(data):
    features=load_features(data);configs=configurations();started=time.monotonic()
    y=np.full(len(data),np.nan);train=np.flatnonzero(data.split.eq('train'));y[train]=data.Label.iloc[train].astype(int)
    assert np.array_equal(train,np.arange(2494))
    groups=baseline.grouping(data.iloc[train].reset_index(drop=True))
    split=pd.read_csv(V2/'nested_splits.csv',dtype={'id':str});lookup=dict(zip(data.id.iloc[train],train))
    oof=data.iloc[train][['id','title','venue','year','Label']].reset_index(drop=True).copy();oof['outer_fold']=-1
    metrics=[];search=[]
    for fold in range(5):
        s=split[split.outer_fold.eq(fold)];fr=s[s.partition.eq('outer_train')];vr=s[s.partition.eq('outer_validation')]
        fit=np.array([lookup[x] for x in fr.id]);val=np.array([lookup[x] for x in vr.id]);foldnums=fr.inner_fold.to_numpy()
        assert set(groups[fit]).isdisjoint(groups[val])
        cachepath=OUT/f'outer_{fold}.npz'
        if cachepath.exists():
            cache=np.load(cachepath);assert np.array_equal(cache['fit'],fit) and np.array_equal(cache['val'],val)
            inner={c['key']:cache['inner_'+c['key']] for c in configs}
            pred={c['key']:cache['outer_'+c['key']] for c in configs}
            train_mae=json.loads((OUT/f'outer_{fold}_train_mae.json').read_text())
        else:
            inner={c['key']:np.full(len(fit),np.nan) for c in configs}
            for inner_fold in range(4):
                mask=foldnums==inner_fold;fi,vi=fit[~mask],fit[mask]
                assert set(groups[fi]).isdisjoint(groups[vi])
                values,_=fit_predict(data,features,y,fi,vi,configs)
                for key in inner:inner[key][mask]=values[key]
                print(f'Outer {fold+1}/5 inner {inner_fold+1}/4; {time.monotonic()-started:.1f}s',flush=True)
            pred,train_mae=fit_predict(data,features,y,fit,val,configs)
            np.savez_compressed(cachepath,fit=fit,val=val,**{'inner_'+k:v for k,v in inner.items()},**{'outer_'+k:v for k,v in pred.items()})
            (OUT/f'outer_{fold}_train_mae.json').write_text(json.dumps(train_mae,indent=2)+'\n')
        choices,rows=select(data,y,fit,foldnums,inner,configs)
        search.extend(dict(outer_fold=fold,**r) for r in rows)
        for family,choice in choices.items():
            ratings=pred[choice['key']];p=priority(ratings,choice['threshold'])
            result=evaluate(y[val],p,data.id.iloc[val])
            metrics.append(dict(outer_fold=fold,family=family,config=choice['key'],
                alpha=choice['alpha'],weight=choice['weight'],mode=choice['mode'],representation=choice['representation'],
                threshold=choice['threshold'],inner_score=choice['inner_score'],**result,
                train_mae=train_mae[choice['key']],validation_mae=mean_absolute_error(y[val],ratings),
                read_auc=roc_auc_score(y[val]>1,ratings),missed_high=int(((y[val]>=4)&(p==0)).sum())))
            oof.loc[val,family+'_rating']=ratings;oof.loc[val,family+'_priority']=p
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(search).to_csv(OUT/'inner_search.csv',index=False)
        oof.to_csv(OUT/'oof_predictions.csv',index=False)
        print(pd.DataFrame(metrics).query('outer_fold==@fold')[['family','score','FP','FN','missed_high']].to_string(index=False),flush=True)
    m=pd.DataFrame(metrics)
    summary=m.groupby('family').agg(H=('score','mean'),H_std=('score','std'),Q=('relevance','mean'),R=('coverage','mean'),
        FP=('FP','sum'),FN=('FN','sum'),missed_high=('missed_high','sum'),AUC=('read_auc','mean'),
        train_mae=('train_mae','mean'),validation_mae=('validation_mae','mean'))
    summary.to_csv(OUT/'summary.csv');print(summary.to_string(),flush=True)
    (OUT/'run_manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
        configs=len(configs),n=2494,data_sha256=baseline.input_digest(data)),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','embed','train'],nargs='?',default='all');args=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);data,_=load_clean_data();protocol(data)
    if args.stage in ['all','embed']:encode(data)
    if args.stage in ['all','train']:
        with threadpool_limits(limits=4):run(data)


if __name__=='__main__':main()
