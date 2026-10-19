#!/usr/bin/env python3
"""Controlled, resumable augmentation experiment; no outer-driven tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'
sys.path.insert(0, str(HERE.parent/'input_ablation'))
import run_ablation as ab
import features as feat

FAMILIES = ['base', 'content', 'neighbors', 'both']
CONFIGS = [dict(key=f'{family}_d{depth}', family=family, depth=depth)
           for family in FAMILIES for depth in [4, 6]]
CONFIGS += [dict(key='ridge', family='ridge', depth=None)]


def context():
    data, template = ab.load_clean_data()
    assert len(data)==3090 and data.split.iloc[:2494].eq('train').all()
    labels = np.full(len(data), np.nan)
    labels[:2494] = data.Label.iloc[:2494].astype(int)
    groups = ab.baseline.grouping(data.reset_index(drop=True))
    keys = (data.split+':'+data.id).to_numpy(dtype=str)
    z = np.load(ab.OUT/'field_embeddings.npz')
    assert np.array_equal(keys, z['keys'])
    title = normalize(z['title'].astype(float))
    path = HERE.parent/'diagnose_bottleneck/outputs/embeddings_retrieval.npz'
    z = np.load(path)
    assert np.array_equal(keys, z['keys'])
    manifest = json.loads(path.with_suffix('.json').read_text())
    assert manifest['input_sha256']==ab.baseline.input_digest(data)
    title_manifest = json.loads((ab.OUT/'embedding_manifest.json').read_text())
    assert title_manifest['input_sha256']==ab.baseline.input_digest(data)
    retrieval = normalize(z['embeddings'].astype(float))
    content, evidence = feat.content_features(data)
    return data, template, labels, groups, title, retrieval, content, evidence


def write_protocol(data):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(plan_sha256=feat.digest_file(HERE/'PLAN.md'),
        source_sha256={p:feat.digest_file(HERE/p) for p in ['features.py', 'run_augmentation.py']},
        data_sha256=ab.baseline.input_digest(data),
        split_sha256=feat.digest_file(ab.V2/'nested_splits.csv'),
        configs=CONFIGS, seed=feat.SEED, iterations=300, learning_rate=.05, l2_leaf_reg=8,
        rules=feat.RULES, crossfit_splits=3, neighbor_k=[5,15],
        reference_manifest_sha256=feat.digest_file(HERE.parent/'diagnose_bottleneck/outputs/embeddings_retrieval.json'),
        title_manifest_sha256=feat.digest_file(ab.OUT/'embedding_manifest.json'),
        cat_thresholds=np.arange(0,.901,.05).tolist(), ridge_thresholds=ab.THRESHOLDS.tolist(),
        score='detailed Evaluation: harmonic mean discounted gain and worthwhile recall',
        limitations=['same outer data inspected in earlier exploratory work',
                     'lexical categories are evidence of wording, not verified semantic classifications',
                     'crossfit training has fewer history references than full-reference prediction'])
    target=OUT/'protocol.json'
    if target.exists():
        assert json.loads(target.read_text())==protocol,'Protocol/source changed; use a new output directory'
    else:
        target.write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def export_content(data, content, evidence):
    frame=pd.concat([data[['split','id']].reset_index(drop=True),content],axis=1)
    frame.to_csv(OUT/'content_features.csv',index=False)
    evidence.to_csv(OUT/'content_evidence.csv',index=False)
    coverage=[]
    for split in ['train','test']:
        mask=data.split.eq(split)
        for field in ['title','abstract']:
            for aspect in feat.RULES:
                unknown=content.loc[mask,f'{field}_{aspect}_unknown'].sum()
                coverage.append(dict(split=split,field=field,aspect=aspect,n=int(mask.sum()),
                                     with_evidence=int(mask.sum()-unknown),unknown=int(unknown)))
    pd.DataFrame(coverage).to_csv(OUT/'content_coverage.csv',index=False)


def estimator(config):
    return CatBoostClassifier(iterations=300,depth=config['depth'],learning_rate=.05,
        l2_leaf_reg=8,loss_function='MultiClass',classes_count=5,random_seed=feat.SEED,
        thread_count=4,verbose=False,allow_writing_files=False)


def partition(ctx, fit, pred, name, save_key=None):
    data,_,y,groups,title,retrieval,content,_=ctx
    path=OUT/(name+'.npz')
    if path.exists() and save_key is None:
        with np.load(path) as z:
            assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
            return {c['key']:z[c['key']] for c in CONFIGS}
    feature_path=OUT/(name+'_features.joblib')
    if feature_path.exists():
        pack=joblib.load(feature_path)
        assert np.array_equal(pack['fit'],fit) and np.array_equal(pack['pred'],pred)
        a,b,ridge,lineage,recipe=[pack[k] for k in ['a','b','ridge','lineage','recipe']]
    else:
        started=time.monotonic()
        a,b,ridge,lineage,recipe=feat.build_features(data,y,groups,title,retrieval,content,fit,pred)
        joblib.dump(dict(a=a,b=b,ridge=ridge,lineage=lineage,recipe=recipe,fit=fit,pred=pred),feature_path,compress=3)
        (OUT/(name+'_lineage.json')).write_text(json.dumps(dict(fit=fit.tolist(),pred=pred.tolist(),crossfits=lineage)))
        print(name,'features',a.shape,'seconds',round(time.monotonic()-started,1),flush=True)
    predictions={'ridge':ridge}
    trainlog=[]
    for config in CONFIGS[:-1]:
        if save_key is not None and config['key']!=save_key:
            continue
        cols=feat.columns_for(a,config['family'])
        started=time.monotonic()
        model=estimator(config)
        model.fit(a[cols],y[fit].astype(int)-1,cat_features=['venue'])
        predictions[config['key']]=model.predict_proba(b[cols])
        assert np.array_equal(model.classes_,np.arange(5))
        trainlog.append(dict(key=config['key'],n_fit=len(fit),n_pred=len(pred),n_features=len(cols),
                             seconds=time.monotonic()-started))
        if save_key==config['key']:
            model.save_model(str(OUT/'final_classifier.cbm'))
            importance=pd.DataFrame(dict(feature=cols,importance=model.feature_importances_))
            importance.sort_values('importance',ascending=False).to_csv(OUT/'final_feature_importance.csv',index=False)
    if save_key is not None:
        joblib.dump(dict(recipe=recipe,fit=fit,pred=pred,columns=feat.columns_for(a,next(c['family'] for c in CONFIGS if c['key']==save_key)),
                         key=save_key),OUT/'final_recipe.joblib',compress=3)
    else:
        np.savez_compressed(path,fit=fit,pred=pred,**predictions)
        (OUT/(name+'_training.json')).write_text(json.dumps(trainlog,indent=2)+'\n')
    return predictions


def priorities(config, values, cutoff):
    if config['family']=='ridge':
        return ab.priority(values,cutoff)
    gain=np.asarray(values) @ (np.arange(5)/4)
    return np.where((1-values[:,0]>cutoff)&(gain>0),10*gain,0.)


def search(ctx, fit, foldnums, predictions):
    data,_,y,*_=ctx
    rows=[]
    for config in CONFIGS:
        best=None
        thresholds=ab.THRESHOLDS if config['family']=='ridge' else np.arange(0,.901,.05)
        for cutoff in thresholds:
            p=priorities(config,predictions[config['key']],cutoff)
            scores=[ab.evaluate(y[fit[foldnums==k]],p[foldnums==k],data.id.iloc[fit[foldnums==k]])['score']
                    for k in range(4)]
            row=dict(config,threshold=float(cutoff),inner_score=float(np.mean(scores)))
            if best is None or row['inner_score']>best['inner_score']:
                best=row
        rows.append(best)
    choices={}
    for row in rows:
        if row['family'] not in choices or row['inner_score']>choices[row['family']]['inner_score']:
            choices[row['family']]=row
    choices['selection_procedure']=max(rows,key=lambda r:r['inner_score'])
    return choices,rows


def outer_members(data):
    splits=pd.read_csv(ab.V2/'nested_splits.csv',dtype={'id':str})
    lookup=dict(zip(data.id.iloc[:2494],range(2494)))
    for fold in range(5):
        rows=splits[splits.outer_fold.eq(fold)]
        f=rows[rows.partition.eq('outer_train')];v=rows[rows.partition.eq('outer_validation')]
        yield fold,np.array([lookup[i] for i in f.id]),np.array([lookup[i] for i in v.id]),f.inner_fold.to_numpy()


def inner_predictions(ctx,fit,nums,prefix):
    result={c['key']:np.full((len(fit),5),np.nan) if c['family']!='ridge' else np.full(len(fit),np.nan) for c in CONFIGS}
    for k in range(4):
        mask=nums==k
        pred=partition(ctx,fit[~mask],fit[mask],prefix+f'_inner{k}')
        for key in result:
            result[key][mask]=pred[key]
        print(prefix,'inner',k,'complete',flush=True)
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def run_outer(ctx):
    data,_,y,*_=ctx
    oof=data[['id','title','venue','year','Label']].iloc[:2494].copy()
    metrics,searches=[] ,[]
    for fold,fit,val,nums in outer_members(data):
        inner=inner_predictions(ctx,fit,nums,f'outer{fold}')
        choices,rows=search(ctx,fit,nums,inner)
        searches.extend(dict(outer_fold=fold,**r) for r in rows)
        pred=partition(ctx,fit,val,f'outer{fold}_validation')
        for family,choice in choices.items():
            p=priorities(choice,pred[choice['key']],choice['threshold'])
            raw=pred[choice['key']]
            read=1-raw[:,0] if choice['family']!='ridge' else raw
            result=ab.evaluate(y[val],p,data.id.iloc[val])
            metrics.append(dict(outer_fold=fold,family=family,key=choice['key'],threshold=choice['threshold'],
                inner_score=choice['inner_score'],**result,read_auc=roc_auc_score(y[val]>1,read),
                missed_high=int(((y[val]>=4)&(p==0)).sum())))
            oof.loc[val,family+'_priority']=p
        oof.loc[val,'outer_fold']=fold
        pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(searches).to_csv(OUT/'inner_search.csv',index=False)
        oof.to_csv(OUT/'oof_predictions.csv',index=False)
        print(pd.DataFrame(metrics).query('outer_fold==@fold')[['family','score','FP','FN','missed_high']].to_string(index=False),flush=True)
    m=pd.DataFrame(metrics)
    summary=m.groupby('family',sort=False).agg(H=('score','mean'),H_std=('score','std'),
        Q=('relevance','mean'),R=('coverage','mean'),FP=('FP','sum'),FN=('FN','sum'),
        missed_high=('missed_high','sum'),AUC=('read_auc','mean'))
    summary.to_csv(OUT/'summary.csv');print(summary.to_string(),flush=True)


def run_final(ctx):
    data,template,y,groups,*_=ctx
    fit=np.arange(2494);pred=np.arange(2494,len(data));nums=np.full(2494,-1)
    for k,(_,v) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=20261118).split(fit,y[fit],groups[fit])):
        nums[v]=k
    pd.DataFrame(dict(id=data.id.iloc[fit],inner_fold=nums)).to_csv(OUT/'final_inner_splits.csv',index=False)
    inner=inner_predictions(ctx,fit,nums,'final')
    choices,rows=search(ctx,fit,nums,inner)
    choice=choices['selection_procedure']
    pd.DataFrame(rows).to_csv(OUT/'final_search.csv',index=False)
    (OUT/'final_choice.json').write_text(json.dumps(choice,indent=2)+'\n')
    result=partition(ctx,fit,pred,'final_test',save_key=choice['key'])
    values=result[choice['key']];p=priorities(choice,values,choice['threshold'])
    output=pd.DataFrame(dict(id=data.id.iloc[pred].to_numpy(),priority=p))
    output=template[['id']].merge(output,on='id',validate='one_to_one')
    output.to_csv(OUT/'submission_candidate.csv',index=False)
    details=data[['id','title','venue']].iloc[pred].copy();details['priority']=p
    if choice['family']!='ridge':
        for k in range(5):details[f'p{k+1}']=values[:,k]
    details.to_csv(OUT/'test_prediction_details.csv',index=False)
    print('FINAL',choice,'selected',int((p>0).sum()),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','outer','final','all'],nargs='?',default='all')
    args=parser.parse_args();started=time.monotonic()
    with threadpool_limits(4):
        ctx=context();write_protocol(ctx[0]);export_content(ctx[0],ctx[6],ctx[7])
        if args.stage in ['outer','all']:run_outer(ctx)
        if args.stage in ['final','all']:run_final(ctx)
    print('DONE',args.stage,'seconds',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':main()
