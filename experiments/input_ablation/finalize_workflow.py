#!/usr/bin/env python3
"""Choose the complete workflow from full-train inner CV and write reviewable artifacts."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

import run_ablation as ab
import run_metadata as meta

OUT=ab.HERE/'final_outputs'
CV_SEED=20261117


def stage_decision():
    baseline=pd.read_csv(ab.HERE.parent/'diagnose_bottleneck/outputs/summary.csv').set_index('family').loc['regression_ridge']
    new=pd.read_csv(meta.OUT/'summary.csv').set_index('family').loc['selection_procedure']
    checks=dict(H_gain_ge_0_01=bool(new.H>=baseline.score_mean+.01),
        FP_reduced_at_least10percent=bool(new.FP<=.9*baseline.FP),
        recall_loss_at_most0_01=bool(new.R>=baseline.R-.01),
        missed_high_not_increased=bool(new.missed_high<=40))
    value=dict(checks=checks,advance_to_adapter=not all(checks.values()),
        policy='Predeclared in input protocol. Gates guide experiment continuation, not claim independent confirmation.',
        baseline_H=float(baseline.score_mean),metadata_policy_H=float(new.H),metadata_policy_FP=int(new.FP),
        metadata_policy_FN=int(new.FN),metadata_policy_R=float(new.R),metadata_policy_missed_high=int(new.missed_high))
    (ab.OUT/'stage_decision.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


def load_neural():
    sys.path.insert(0,str(ab.HERE.parent/'preference_adapter'))
    import run_adapter
    return run_adapter


def combined_outer(neural):
    data,_=ab.load_clean_data()
    metrics=pd.read_csv(meta.OUT/'fold_metrics.csv')
    preds=pd.read_csv(meta.OUT/'oof_predictions.csv',dtype={'id':str})
    neural_metrics=None;neural_preds=None
    if neural:
        root=ab.HERE.parent/'preference_adapter/outputs'
        neural_metrics=pd.read_csv(root/'fold_metrics.csv');neural_preds=pd.read_csv(root/'oof_predictions.csv',dtype={'id':str})
        assert np.array_equal(preds.id,neural_preds.id)
    result=preds[['id','title','venue','year','Label','outer_fold']].copy();rows=[]
    for fold in range(5):
        choice=metrics[metrics.outer_fold.eq(fold)&metrics.family.eq('selection_procedure')].iloc[0]
        source='metadata';priority=preds.selection_procedure_priority
        if neural:
            candidate=neural_metrics[neural_metrics.outer_fold.eq(fold)].sort_values('inner_score',ascending=False,kind='stable').iloc[0]
            if candidate.inner_score>choice.inner_score:
                choice=candidate;source='adapter';priority=neural_preds[choice.family+'_priority']
        mask=result.outer_fold.eq(fold);p=priority[mask].to_numpy()
        rows.append(dict(outer_fold=fold,source=source,chosen_family=choice.family,inner_score=choice.inner_score,
                         **ab.evaluate(result.Label[mask],p,result.id[mask]),
                         missed_high=int(((result.Label[mask]>=4)&(p==0)).sum())))
        result.loc[mask,'priority']=p;result.loc[mask,'source']=source
    result.to_csv(OUT/'workflow_oof.csv',index=False);m=pd.DataFrame(rows);m.to_csv(OUT/'workflow_fold_metrics.csv',index=False)
    stats=dict(H=float(m.score.mean()),H_std=float(m.score.std()),Q=float(m.relevance.mean()),R=float(m.coverage.mean()),
               FP=int(m.FP.sum()),FN=int(m.FN.sum()),missed_high=int(m.missed_high.sum()))
    return stats


def fit_classical(data,audit,features,y,train,test,base,choice):
    spaces,vectors=ab.build_spaces(data,features,train,test);a,b=ab.space_for(base,spaces)
    fitters=None
    if choice['kind']!='none':
        blocks,fitters=meta.meta_spaces(data,audit,train,test,return_fitters=True)
        af,bf=blocks[choice['kind']]
        a=ab.normalize(sparse.hstack([sparse.csr_matrix(a),choice['weight']*af],format='csr'))
        b=ab.normalize(sparse.hstack([sparse.csr_matrix(b),choice['weight']*bf],format='csr'))
    alpha=base['alpha'] if choice['kind']=='none' else choice['alpha']
    if choice['kind']=='none':model=Ridge(alpha=alpha,solver='lsqr',tol=1e-6).fit(a,y[train])
    else:model=Ridge(alpha=alpha,solver='cholesky').fit(a.toarray() if sparse.issparse(a) else a,y[train])
    ratings=model.predict(b);p=ab.priority(ratings,choice['threshold'])
    bundle=dict(model=model,base=base,metadata_choice=choice,vectorizers=vectors,metadata_transformers=fitters,
        data_sha256=ab.baseline.input_digest(data),metadata_sha256=hashlib.sha256((ab.OUT/'metadata_audit.csv').read_bytes()).hexdigest(),
        input_protocol=json.loads((ab.OUT/'protocol.json').read_text()),
        note='Field embeddings keyed by split:id are in input_ablation outputs; full source pins and encoder recipe in protocol.')
    joblib.dump(bundle,OUT/'model_workflow.joblib')
    if sparse.issparse(b):sparse.save_npz(OUT/'test_features.npz',b)
    else:np.savez_compressed(OUT/'test_features.npz',dense=b)
    np.testing.assert_allclose(joblib.load(OUT/'model_workflow.joblib')['model'].predict(b),ratings,atol=1e-12)
    return ratings,p


def main():
    OUT.mkdir(exist_ok=True,parents=True);decision=stage_decision()
    neural=decision['advance_to_adapter']
    prepare_only='--prepare-only' in sys.argv
    if neural and not prepare_only:assert (ab.HERE.parent/'preference_adapter/outputs/summary.csv').exists(),'Complete the required adapter control before finalizing'
    data,template=ab.load_clean_data();features=ab.load_features(data)
    audit=pd.read_csv(ab.OUT/'metadata_audit.csv',dtype={'id':str},keep_default_na=False)
    assert np.array_equal(audit.id,data.id)
    y=np.full(len(data),np.nan);train=np.arange(2494);test=np.arange(2494,len(data));y[train]=data.Label.iloc[train].astype(int)
    groups=ab.baseline.grouping(data.iloc[:2494].reset_index(drop=True))
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=CV_SEED).split(train,y[train],groups))
    nums=np.full(len(train),-1)
    for k,(fi,vi) in enumerate(folds):
        assert set(groups[fi]).isdisjoint(groups[vi]);nums[vi]=k
    p=dict(seed=CV_SEED,inner_folds=4,neural_calibration_group=0,
           selection='Highest inner evaluation H: 4-fold mean for text/metadata, fixed group0 calibration for fixed-schedule neural arms. Outer scores do not choose final config.',
           input_protocol_sha256=hashlib.sha256((ab.OUT/'protocol.json').read_bytes()).hexdigest(),
           metadata_protocol_sha256=hashlib.sha256((meta.OUT/'protocol.json').read_bytes()).hexdigest(),neural_included=neural)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:path.write_text(json.dumps(p,indent=2)+'\n')
    configs=ab.configurations();inner={c['key']:np.full(len(train),np.nan) for c in configs}
    for k,(fi,vi) in enumerate(folds):
        path=OUT/f'input_inner{k}.npz'
        if path.exists():values=dict(np.load(path))
        else:
            values,_=ab.fit_predict(data,features,y,fi,vi,configs);np.savez_compressed(path,**values)
        for key in inner:inner[key][vi]=values[key]
    choices,search=ab.select(data,y,train,nums,inner,configs)
    pd.DataFrame(search).to_csv(OUT/'input_inner_search.csv',index=False)
    base=choices['selection_procedure']
    print('FINAL TEXT INPUT',base,flush=True)
    mc=meta.configurations();mi={c['key']:np.full(len(train),np.nan) for c in mc}
    for k,(fi,vi) in enumerate(folds):
        path=OUT/f'metadata_inner{k}.npz'
        if path.exists():values=dict(np.load(path))
        else:
            values,_=meta.fit_predict(data,audit,features,y,fi,vi,base,mc);np.savez_compressed(path,**values)
        for key in mi:mi[key][vi]=values[key]
    choices,search=ab.select(data,y,train,nums,mi,mc)
    pd.DataFrame(search).to_csv(OUT/'metadata_inner_search.csv',index=False)
    chosen=choices['selection_procedure'];source='metadata'
    if prepare_only:
        (OUT/'classical_final_search.json').write_text(json.dumps(dict(base=base,choice=chosen),indent=2)+'\n')
        print('Classical full-train inner search cached; waiting for neural comparison before final choice.',flush=True)
        return
    neural_predictions=None
    if neural:
        n=load_neural();tokenizer,tokens,frozen=n.prepare(data);ni={arm:np.zeros((2494,2),dtype=np.float32) for arm in n.ARMS}
        for arm in n.ARMS:
            for k,(fi,vi) in list(enumerate(folds))[:1]:
                path=OUT/f'neural_inner{k}_{arm}.npz'
                if path.exists():values=np.load(path)['values']
                else:
                    values,info=n.fit_one(data,y,fi,vi,base,arm,n.SEED,tokenizer,tokens,frozen)
                    np.savez_compressed(path,values=values)
                ni[arm][vi]=values
        calibration=nums==0
        choices=n.choose(data,y,train[calibration],nums[calibration],{arm:v[calibration] for arm,v in ni.items()})
        pd.DataFrame(choices.values()).to_csv(OUT/'neural_inner_search.csv',index=False)
        candidate=max(choices.values(),key=lambda x:x['inner_score'])
        if candidate['inner_score']>chosen['inner_score']:chosen=candidate;source='adapter'
    print('FINAL WINNER',source,chosen,flush=True)
    if source=='metadata':ratings,priorities=fit_classical(data,audit,features,y,train,test,base,chosen)
    else:
        values,info=n.fit_one(data,y,train,test,base,chosen['arm'],n.SEED,tokenizer,tokens,frozen,save_dir=OUT/'neural_model')
        reloaded=n.predict_saved(data,test,OUT/'neural_model',tokenizer,tokens,frozen)
        np.testing.assert_allclose(reloaded,values,atol=1e-6)
        np.testing.assert_array_equal(n.make_priority(reloaded,chosen['threshold'])>0,n.make_priority(values,chosen['threshold'])>0)
        np.savez_compressed(OUT/'neural_test_predictions.npz',values=values)
        ratings=1+4*values[:,1];priorities=n.make_priority(values,chosen['threshold'])
    submission=template[['id']].merge(pd.DataFrame(dict(id=data.id.iloc[test],priority=priorities)),on='id',how='left',validate='one_to_one')
    assert len(submission)==596 and submission.id.is_unique and np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
    submission.to_csv(OUT/'submission_workflow.csv',index=False)
    pd.DataFrame(dict(id=data.id.iloc[test],expected_rating=ratings,priority=priorities)).to_csv(OUT/'test_prediction_details.csv',index=False)
    stats=combined_outer(neural)
    manifest=dict(source=source,base=base,chosen=chosen,workflow_outer_metrics=stats,test_selected=int(submission.priority.gt(0).sum()),
        stage_decision=decision,not_submitted_to_kaggle=True,original_labels_preserved=True,
        limitation='Research reused these outer labels. Family results are diagnostic; full workflow choice is evaluated separately by choosing within each outer training set.')
    (OUT/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
