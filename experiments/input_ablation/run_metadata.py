#!/usr/bin/env python3
"""Controlled original/verified metadata experiment on inner-selected text inputs."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse,linalg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from threadpoolctl import threadpool_limits

import run_ablation as ab

OUT=ab.HERE/'metadata_outputs'


def configurations():
    rows=[dict(key='reference',family='input_policy',kind='none',weight=0.,alpha=None)]
    for kind in ['original_all','clean_all','clean_authors','venue','clean_year']:
        for weight in [.25,1.]:
            for alpha in ab.ALPHAS:
                rows.append(dict(key=f'm{len(rows):03d}',family=kind,kind=kind,weight=weight,alpha=alpha))
    return rows


def meta_spaces(data,audit,fit,pred,return_fitters=False):
    blocks={};fitters={}
    for kind,col in [('original_authors','author_tokens_original'),('clean_authors','author_tokens_clean')]:
        vec=CountVectorizer(token_pattern=r'(?u)\b\w+\b',binary=True)
        af=normalize(vec.fit_transform(audit[col].iloc[fit]));ap=normalize(vec.transform(audit[col].iloc[pred]))
        blocks[kind]=(af,ap)
        fitters[kind]=vec
    v=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    blocks['venue']=(v.fit_transform(data[['venue']].iloc[fit]),v.transform(data[['venue']].iloc[pred]))
    fitters['venue']=v
    for kind,col in [('original_year','year_original'),('clean_year','year_clean')]:
        scaler=StandardScaler()
        blocks[kind]=(sparse.csr_matrix(.25*scaler.fit_transform(audit[[col]].iloc[fit])),
                      sparse.csr_matrix(.25*scaler.transform(audit[[col]].iloc[pred])))
        fitters[kind]=scaler
    for kind,prefix in [('original_all','original'),('clean_all','clean')]:
        blocks[kind]=tuple(normalize(sparse.hstack([blocks['venue'][i],blocks[prefix+'_authors'][i],
                    blocks[prefix+'_year'][i]],format='csr')) for i in [0,1])
    return (blocks,fitters) if return_fitters else blocks


def fit_predict(data,audit,features,y,fit,pred,base,configs):
    if base['representation']=='specter':
        spaces={('specter',key):(value[fit],value[pred]) for key,value in features.items()}
    else:spaces,_=ab.build_spaces(data,features,fit,pred)
    xf,xp=ab.space_for(base,spaces)
    blocks=meta_spaces(data,audit,fit,pred);values={};train_mae={}
    ref=Ridge(alpha=base['alpha'],solver='lsqr',tol=1e-6).fit(xf,y[fit])
    values['reference']=ref.predict(xp);train_mae['reference']=float(mean_absolute_error(y[fit],ref.predict(xf)))
    to_dense=lambda a:a.toarray() if sparse.issparse(a) else np.asarray(a)
    ktext=to_dense(xf@xf.T);cross_text=to_dense(xp@xf.T)
    text_train_norm=np.asarray(xf.multiply(xf).sum(axis=1)).ravel() if sparse.issparse(xf) else (xf*xf).sum(axis=1)
    text_pred_norm=np.asarray(xp.multiply(xp).sum(axis=1)).ravel() if sparse.issparse(xp) else (xp*xp).sum(axis=1)
    # Same normalized linear Ridge objective. Factor each feature-space Gram once for all alphas.
    for kind in ['original_all','clean_all','clean_authors','venue','clean_year']:
        mf,mp=blocks[kind];km=to_dense(mf@mf.T);cm=to_dense(mp@mf.T)
        train_norm=np.asarray(mf.multiply(mf).sum(axis=1)).ravel()
        pred_norm=np.asarray(mp.multiply(mp).sum(axis=1)).ravel()
        for weight in [.25,1.]:
            nf=np.sqrt(np.maximum(text_train_norm+weight*weight*train_norm,1e-24))
            npred=np.sqrt(np.maximum(text_pred_norm+weight*weight*pred_norm,1e-24))
            kernel=(ktext+weight*weight*km)/(nf[:,None]*nf[None,:])
            cross=(cross_text+weight*weight*cm)/(npred[:,None]*nf[None,:])
            outputs,train_outputs=linear_kernel_ridge_grid(kernel,cross,y[fit],ab.ALPHAS)
            for c in configs:
                if c['kind']==kind and c['weight']==weight:
                    values[c['key']]=outputs[c['alpha']]
                    train_mae[c['key']]=float(mean_absolute_error(y[fit],train_outputs[c['alpha']]))
    return values,train_mae


def linear_kernel_ridge_grid(kernel,cross,y,alphas):
    mean=kernel.mean(axis=0);grand=mean.mean()
    centered=kernel-mean[None,:]-mean[:,None]+grand
    cross_center=cross-cross.mean(axis=1,keepdims=True)-mean[None,:]+grand
    eig,vectors=linalg.eigh(centered,check_finite=False)
    eig=np.maximum(eig,0.)
    target=vectors.T@(y-y.mean());projection=cross_center@vectors
    pred={};train={}
    for alpha in alphas:
        dual=target/(eig+alpha)
        pred[alpha]=projection@dual+y.mean()
        train[alpha]=vectors@(eig*dual)+y.mean()
    return pred,train


def run():
    OUT.mkdir(exist_ok=True);data,_=ab.load_clean_data();features=ab.load_features(data)
    audit_path=ab.OUT/'metadata_audit.csv'
    audit=pd.read_csv(audit_path,dtype={'id':str},keep_default_na=False)
    assert np.array_equal(audit.id,data.id) and np.array_equal(audit.split,data.split)
    configs=configurations()
    protocol=dict(configs=configs,thresholds=ab.THRESHOLDS.tolist(),metadata_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        data_sha256=ab.baseline.input_digest(data),input_protocol_sha256=hashlib.sha256((ab.OUT/'protocol.json').read_bytes()).hexdigest(),
        input_policy='For each outer fold choose text representation/mode/weight/alpha using stage1 INNER scores only; compare metadata conditional on this selection.',
        encoding='Original vs verified author surnames use same CountVectorizer binary + L2; unknown verified data fallback to original. Venue onehot. Train-scaled years times0.25.',
        choice='Each metadata family and global policy choose params/threshold by mean inner H; original_all and clean_all have identical search spaces.',
        solver='Exact centered linear-kernel eigensolve, reusing eigensystem across alphas; mathematically same normalized Ridge with intercept. Reference uses original lsqr tol1e-6. All configs/thresholds unchanged.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    y=np.full(len(data),np.nan);y[:2494]=data.Label.iloc[:2494].astype(int)
    groups=ab.baseline.grouping(data.iloc[:2494].reset_index(drop=True))
    splits=pd.read_csv(ab.V2/'nested_splits.csv',dtype={'id':str});lookup=dict(zip(data.id.iloc[:2494],range(2494)))
    previous=pd.read_csv(ab.OUT/'inner_search.csv')
    base_configs={c['key']:c for c in ab.configurations()}
    metrics=[];search=[];choices_out={};oof=data.iloc[:2494][['id','title','venue','year','Label']].copy();oof['outer_fold']=-1
    started=time.monotonic()
    for fold in range(5):
        b=previous[previous.outer_fold.eq(fold)].sort_values('inner_score',ascending=False,kind='stable').iloc[0]
        base=base_configs[b['key']];choices_out[str(fold)]=base
        s=splits[splits.outer_fold.eq(fold)];fr=s[s.partition.eq('outer_train')];vr=s[s.partition.eq('outer_validation')]
        fit=np.array([lookup[x] for x in fr.id]);val=np.array([lookup[x] for x in vr.id]);nums=fr.inner_fold.to_numpy()
        assert set(groups[fit]).isdisjoint(groups[val])
        path=OUT/f'outer_{fold}.npz'
        if path.exists():
            cache=np.load(path);assert np.array_equal(fit,cache['fit']) and np.array_equal(val,cache['val'])
            inner={c['key']:cache['inner_'+c['key']] for c in configs}
            pred={c['key']:cache['outer_'+c['key']] for c in configs}
            tr=json.loads((OUT/f'outer_{fold}_train_mae.json').read_text())
        else:
            inner={c['key']:np.full(len(fit),np.nan) for c in configs}
            for k in range(4):
                mask=nums==k;fi,vi=fit[~mask],fit[mask]
                assert set(groups[fi]).isdisjoint(groups[vi])
                vals,_=fit_predict(data,audit,features,y,fi,vi,base,configs)
                for key in inner:inner[key][mask]=vals[key]
            pred,tr=fit_predict(data,audit,features,y,fit,val,base,configs)
            np.savez_compressed(path,fit=fit,val=val,**{'inner_'+k:v for k,v in inner.items()},**{'outer_'+k:v for k,v in pred.items()})
            (OUT/f'outer_{fold}_train_mae.json').write_text(json.dumps(tr,indent=2)+'\n')
        selected,rows=ab.select(data,y,fit,nums,inner,configs)
        search.extend(dict(outer_fold=fold,base=base['key'],**r) for r in rows)
        for family,c in selected.items():
            ratings=pred[c['key']];p=ab.priority(ratings,c['threshold'])
            metrics.append(dict(outer_fold=fold,family=family,base=base['key'],config=c['key'],
                threshold=c['threshold'],inner_score=c['inner_score'],**ab.evaluate(y[val],p,data.id.iloc[val]),
                AUC=roc_auc_score(y[val]>1,ratings),validation_mae=mean_absolute_error(y[val],ratings),train_mae=tr[c['key']],
                missed_high=int(((y[val]>=4)&(p==0)).sum())))
            oof.loc[val,family+'_rating']=ratings;oof.loc[val,family+'_priority']=p
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(search).to_csv(OUT/'inner_search.csv',index=False)
        oof.to_csv(OUT/'oof_predictions.csv',index=False)
        print(f'Metadata outer {fold+1}/5; {time.monotonic()-started:.1f}s',flush=True)
        print(pd.DataFrame(metrics).query('outer_fold==@fold')[['family','score','FP','FN']].to_string(index=False),flush=True)
    m=pd.DataFrame(metrics)
    m.groupby('family').agg(H=('score','mean'),H_std=('score','std'),Q=('relevance','mean'),R=('coverage','mean'),
        FP=('FP','sum'),FN=('FN','sum'),missed_high=('missed_high','sum'),AUC=('AUC','mean'),
        validation_mae=('validation_mae','mean'),train_mae=('train_mae','mean')).to_csv(OUT/'summary.csv')
    (OUT/'base_choices.json').write_text(json.dumps(choices_out,indent=2)+'\n')
    print(pd.read_csv(OUT/'summary.csv').to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
