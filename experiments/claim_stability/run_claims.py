#!/usr/bin/env python3
"""Five fixed contrasts, nested threshold/arm selection and head-seed sensitivity."""
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
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'preference_augmentation'))
import run_augmentation as previous
import claim_features as cf

ARMS=['reference','regularized','repaired','prefix','claims']
SEEDS=[20261018,20261019,20261020]
THRESHOLDS=np.arange(0,.901,.05)


def context():
    prior=previous.context();data=prior[0]
    old=json.loads((previous.OUT/'protocol.json').read_text())
    for name,value in old['source_sha256'].items():assert cf.digest(previous.HERE/name)==value
    texts,evidence=cf.extract(data)
    repaired,ledger=cf.repair(data,prior[6])
    return dict(prior=prior,texts=texts,evidence=evidence,repaired=repaired,ledger=ledger)


def protocol(ctx):
    OUT.mkdir(parents=True,exist_ok=True)
    mini=json.loads((HERE.parent/'diagnose_bottleneck/outputs/embeddings_minilm.json').read_text())
    value=dict(plan_sha256=cf.digest(HERE/'PLAN.md'),
        source_sha256={f:cf.digest(HERE/f) for f in ['run_claims.py','claim_features.py']},
        previous_protocol_sha256=cf.digest(previous.OUT/'protocol.json'),
        data_sha256=previous.ab.baseline.input_digest(ctx['prior'][0]),
        text_sha256=hashlib.sha256(ctx['texts'].to_csv(index=False).encode()).hexdigest(),
        arms=ARMS,seeds=SEEDS,primary_seed=SEEDS[0],thresholds=THRESHOLDS.tolist(),
        model=dict(iterations=300,learning_rate=.05,reference_depth=4,reference_l2=8,
                   regularized_depth=3,regularized_l2=32),
        encoder=dict(model=mini['repo'],revision=mini['revision'],max_tokens_per_block=128,
                     blocks=3,projection_svd=16,frozen=True),
        cues=cf.CUES,education=cf.EDUCATION,tutorial=cf.TUTORIAL,
        sensitivity='Refit outer heads only; hold primary inner arm/cutoff choices fixed; no seed picking',
        limitation='Repeatedly inspected outer data; prefix matched by max budget, not equal actual text length')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==value,'Protocol changed; use a new experiment directory'
    else:p.write_text(json.dumps(value,indent=2)+'\n')
    return value


def prepare(ctx,proto):
    data=ctx['prior'][0]
    ctx['texts'].to_csv(OUT/'extracted_texts.csv',index=False)
    ctx['evidence'].to_csv(OUT/'sentence_evidence.csv',index=False)
    ctx['ledger'].to_csv(OUT/'repair_ledger.csv',index=False)
    pd.concat([data[['split','id']],ctx['repaired']],axis=1).to_csv(OUT/'repaired_content.csv',index=False)
    cf.encode(ctx['texts'],OUT,proto['encoder']['model'],proto['encoder']['revision'])
    z=np.load(OUT/'sentence_embeddings.npz')
    assert np.array_equal(z['keys'],(data.split+':'+data.id).to_numpy(dtype=str))
    ctx['vectors']={k:z[k] for k in ['prefix','claims']}
    coverage=[]
    for split in ['train','test']:
        mask=data.split.eq(split)
        for kind in ['prefix','claims']:
            for j,role in enumerate(cf.ASPECTS):
                coverage.append(dict(split=split,kind=kind,role=role,n=int(mask.sum()),
                    available=int((np.linalg.norm(ctx['vectors'][kind][mask,j],axis=1)>0).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    probes=pd.read_csv(HERE.parent/'plateau_diagnosis/outputs/content_pairs.csv',dtype={'a_id':str,'b_id':str},keep_default_na=False)
    rows=[];lookup=dict(zip(data.id.iloc[:2494],range(2494)))
    for row in probes.to_dict('records'):
        a,b=lookup[row['a_id']],lookup[row['b_id']]
        v=ctx['vectors']['claims']
        item={k:row[k] for k in ['pair_id','selection','a_id','b_id','category_flag_differences']}
        item['different_claim_blocks']=sum(ctx['texts'].iloc[a]['claims_'+r]!=ctx['texts'].iloc[b]['claims_'+r] for r in cf.ASPECTS)
        for j,r in enumerate(cf.ASPECTS):
            item['a_'+r]=ctx['texts'].iloc[a]['claims_'+r]
            item['b_'+r]=ctx['texts'].iloc[b]['claims_'+r]
            item[r+'_cosine']=float(v[a,j]@v[b,j]) if np.linalg.norm(v[a,j])*np.linalg.norm(v[b,j])>0 else np.nan
        rows.append(item)
    pd.DataFrame(rows).to_csv(OUT/'prior_probe_audit.csv',index=False)


def repair_frame(original,fixed,indices):
    frame=original.copy()
    for col in fixed:
        frame['content_'+col]=fixed[col].iloc[indices].to_numpy()
    return frame


def feature_pack(ctx,name,fit,pred):
    path=OUT/(name+'_features.joblib')
    if path.exists():
        pack=joblib.load(path)
        assert np.array_equal(pack['fit'],fit) and np.array_equal(pack['pred'],pred)
        return pack
    old=joblib.load(previous.OUT/(name+'_features.joblib'))
    assert np.array_equal(old['fit'],fit) and np.array_equal(old['pred'],pred)
    pack=dict(fit=fit,pred=pred,original_a=old['a'],original_b=old['b'],lineage=old['lineage'],base_recipe=old['recipe'])
    pack['repaired_a']=repair_frame(old['a'],ctx['repaired'],fit)
    pack['repaired_b']=repair_frame(old['b'],ctx['repaired'],pred)
    pack['projections']={}
    for kind in ['prefix','claims']:
        a,b,models=cf.project(ctx['vectors'][kind],fit,pred)
        pack['projections'][kind]=dict(a=a,b=b,models=models)
    joblib.dump(pack,path,compress=3)
    (OUT/(name+'_lineage.json')).write_text(json.dumps(dict(fit=fit.tolist(),pred=pred.tolist(),
        crossfits=old['lineage'],projection_fit=fit.tolist(),inherited_from=str(previous.OUT/(name+'_features.joblib')))))
    return pack


def frame(pack,arm,partition):
    if arm in ['reference','regularized']:return pack['original_'+partition]
    base=pack['repaired_'+partition]
    if arm in ['prefix','claims']:
        return pd.concat([base,pack['projections'][arm][partition]],axis=1)
    return base


def estimator(arm,seed):
    return CatBoostClassifier(iterations=300,depth=4 if arm=='reference' else 3,
        l2_leaf_reg=8 if arm=='reference' else 32,learning_rate=.05,loss_function='MultiClass',
        classes_count=5,random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)


def train_predictions(ctx,pack,arms,seed,save_model=False):
    y=ctx['prior'][2];fit=pack['fit'];result={};logs=[]
    for arm in arms:
        a,b=frame(pack,arm,'a'),frame(pack,arm,'b');started=time.monotonic()
        model=estimator(arm,seed);model.fit(a,y[fit].astype(int)-1,cat_features=['venue'])
        result[arm]=model.predict_proba(b);result['train_'+arm]=model.predict_proba(a)
        logs.append(dict(arm=arm,seed=seed,n_fit=len(fit),n_pred=len(pack['pred']),n_features=a.shape[1],
                         seconds=time.monotonic()-started))
        if save_model:
            model.save_model(str(OUT/'final_classifier.cbm'))
            pd.DataFrame(dict(feature=a.columns,importance=model.feature_importances_)).sort_values('importance',ascending=False).to_csv(OUT/'final_feature_importance.csv',index=False)
    return result,logs


def primary_predictions(ctx,name,fit,pred):
    path=OUT/(name+'_predictions.npz')
    if path.exists():
        with np.load(path) as z:
            assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
            return {k:z[k] for k in z.files if k not in ['fit','pred']}
    pack=feature_pack(ctx,name,fit,pred)
    result,logs=train_predictions(ctx,pack,ARMS[1:],SEEDS[0])
    with np.load(previous.OUT/(name+'.npz')) as z:
        assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
        result['reference']=z['both_d4']
    if name.endswith('_validation'):
        fold=int(name[5:name.index('_')])
        with np.load(HERE.parent/f'plateau_diagnosis/outputs/outer{fold}_full.npz') as z:
            assert np.array_equal(z['fit'],fit) and np.array_equal(z['val'],pred)
            result['train_reference']=z['catboost_train']
    np.savez_compressed(path,fit=fit,pred=pred,**result)
    (OUT/(name+'_training.json')).write_text(json.dumps(logs,indent=2)+'\n')
    return result


def priority(probability,threshold):
    gain=probability@(np.arange(5)/4)
    return np.where((1-probability[:,0]>threshold)&(gain>0),10*gain,0.)


def select(ctx,fit,nums,values):
    data,_,y,*_=ctx['prior'];choices={};rows=[]
    for arm in ARMS:
        candidates=[]
        for cutoff in THRESHOLDS:
            p=priority(values[arm],cutoff)
            score=np.mean([previous.ab.evaluate(y[fit[nums==k]],p[nums==k],data.id.iloc[fit[nums==k]])['score'] for k in range(4)])
            item=dict(family=arm,threshold=float(cutoff),inner_score=float(score));rows.append(item);candidates.append(item)
        choices[arm]=max(candidates,key=lambda r:r['inner_score'])
    choices['selection_procedure']=max(choices.values(),key=lambda r:r['inner_score'])
    return choices,rows


def inner(ctx,fit,nums,prefix):
    result={arm:np.full((len(fit),5),np.nan) for arm in ARMS}
    for k in range(4):
        mask=nums==k
        values=primary_predictions(ctx,prefix+f'_inner{k}',fit[~mask],fit[mask])
        for arm in ARMS:result[arm][mask]=values[arm]
        print(prefix,'inner',k,'complete',flush=True)
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def metrics(y,values,train_y,train_values,ids,cutoff):
    p=priority(values,cutoff)
    result=previous.ab.evaluate(y,p,ids)
    val_auc=roc_auc_score(y>1,1-values[:,0]);train_auc=roc_auc_score(train_y>1,1-train_values[:,0])
    result.update(read_auc=val_auc,train_auc=train_auc,auc_gap=train_auc-val_auc,
        validation_logloss=log_loss(y-1,values,labels=np.arange(5)),
        train_logloss=log_loss(train_y-1,train_values,labels=np.arange(5)),
        missed_high=int(((y>=4)&(p==0)).sum()))
    return result


def run_outer(ctx):
    data,_,y,*_=ctx['prior'];oof=data[['id','title','venue','Label']].iloc[:2494].copy()
    rows=[];searches=[]
    for fold,fit,val,nums in previous.outer_members(data):
        primary=inner(ctx,fit,nums,f'outer{fold}')
        choices,search=select(ctx,fit,nums,primary)
        searches.extend(dict(outer_fold=fold,**r) for r in search)
        values=primary_predictions(ctx,f'outer{fold}_validation',fit,val)
        pack=None
        for seed in SEEDS:
            if seed!=SEEDS[0]:
                path=OUT/f'outer{fold}_seed{seed}.npz'
                if path.exists():
                    with np.load(path) as z:
                        assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],val)
                        values={k:z[k] for k in z.files if k not in ['fit','pred']}
                else:
                    if pack is None:pack=feature_pack(ctx,f'outer{fold}_validation',fit,val)
                    values,logs=train_predictions(ctx,pack,ARMS,seed)
                    np.savez_compressed(path,fit=fit,pred=val,**values)
                    (OUT/f'outer{fold}_seed{seed}_training.json').write_text(json.dumps(logs,indent=2)+'\n')
            for family,choice in choices.items():
                arm=choice['family'];p=priority(values[arm],choice['threshold'])
                result=metrics(y[val],values[arm],y[fit],values['train_'+arm],data.id.iloc[val],choice['threshold'])
                rows.append(dict(outer_fold=fold,seed=seed,family=family,chosen_arm=arm,
                                 threshold=choice['threshold'],inner_score=choice['inner_score'],**result))
                oof.loc[val,f'{family}_seed{seed}_priority']=p
            oof.loc[val,'outer_fold']=fold
            print(pd.DataFrame(rows).query('outer_fold==@fold and seed==@seed')[['family','score','FP','FN','missed_high','auc_gap']].to_string(index=False),flush=True)
        pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.DataFrame(searches).to_csv(OUT/'inner_search.csv',index=False)
        oof.to_csv(OUT/'oof_predictions.csv',index=False)
    m=pd.DataFrame(rows)
    summary=m.groupby(['seed','family'],sort=False).agg(H=('score','mean'),H_fold_sd=('score','std'),
        Q=('relevance','mean'),R=('coverage','mean'),FP=('FP','sum'),FN=('FN','sum'),
        missed_high=('missed_high','sum'),AUC=('read_auc','mean'),train_AUC=('train_auc','mean'),
        AUC_gap=('auc_gap','mean'),logloss=('validation_logloss','mean'))
    summary.to_csv(OUT/'summary.csv');print(summary.to_string(),flush=True)


def final(ctx):
    data,template,y,*_=ctx['prior'];fit=np.arange(2494);pred=np.arange(2494,len(data))
    membership=pd.read_csv(previous.OUT/'final_inner_splits.csv',dtype={'id':str})
    assert np.array_equal(membership.id,data.id.iloc[fit])
    nums=membership.inner_fold.to_numpy()
    choices,rows=select(ctx,fit,nums,inner(ctx,fit,nums,'final'))
    choice=choices['selection_procedure']
    pd.DataFrame(rows).to_csv(OUT/'final_search.csv',index=False)
    (OUT/'final_choice.json').write_text(json.dumps(choice,indent=2)+'\n')
    pack=feature_pack(ctx,'final_test',fit,pred)
    values,logs=train_predictions(ctx,pack,[choice['family']],SEEDS[0],save_model=True)
    probabilities=values[choice['family']];p=priority(probabilities,choice['threshold'])
    joblib.dump(dict(choice=choice,fit=fit,pred=pred,base_recipe=pack['base_recipe'],
        projections={k:pack['projections'][k]['models'] for k in ['prefix','claims']},
        columns=frame(pack,choice['family'],'a').columns.tolist(),protocol_sha256=cf.digest(OUT/'protocol.json')),
        OUT/'final_recipe.joblib',compress=3)
    export=pd.DataFrame(dict(id=data.id.iloc[pred].to_numpy(),priority=p))
    template[['id']].merge(export,on='id',validate='one_to_one').to_csv(OUT/'submission_candidate.csv',index=False)
    details=data[['id','title','venue']].iloc[pred].copy();details['priority']=p
    for k in range(5):details[f'p{k+1}']=probabilities[:,k]
    details.to_csv(OUT/'test_predictions.csv',index=False)
    print('FINAL',choice,'positive',int((p>0).sum()),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','outer','final','all'],nargs='?',default='all')
    args=parser.parse_args();started=time.monotonic()
    with threadpool_limits(4):
        ctx=context();proto=protocol(ctx);prepare(ctx,proto)
        if args.stage in ['outer','all']:run_outer(ctx)
        if args.stage in ['final','all']:final(ctx)
    print('FITS COMPLETE',args.stage,'seconds',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':main()
