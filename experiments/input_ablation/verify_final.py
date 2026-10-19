#!/usr/bin/env python3
"""Independently rebuild selected test features and recompute the staged OOF policy."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

import run_ablation as ab
import run_metadata as meta
from finalize_workflow import OUT,CV_SEED,load_neural


def rebuild_classical(bundle,data,audit,test):
    base=bundle['base'];spaces={}
    if base['representation']=='specter':
        features=ab.load_features(data)
        for key,value in features.items():spaces[('specter',key)]=(value[test],value[test])
    else:
        for key,column in [('title','title'),('abstract','abstract'),('joint','text')]:
            value=bundle['vectorizers'][key].transform(data[column].iloc[test]);spaces[('tfidf',key)]=(value,value)
    x,_=ab.space_for(base,spaces);choice=bundle['metadata_choice']
    if choice['kind']=='none':return x
    fitters=bundle['metadata_transformers'];blocks={}
    for kind,col in [('original_authors','author_tokens_original'),('clean_authors','author_tokens_clean')]:
        blocks[kind]=ab.normalize(fitters[kind].transform(audit[col].iloc[test]))
    blocks['venue']=fitters['venue'].transform(data[['venue']].iloc[test])
    for kind,col in [('original_year','year_original'),('clean_year','year_clean')]:
        blocks[kind]=sparse.csr_matrix(.25*fitters[kind].transform(audit[[col]].iloc[test]))
    for kind,prefix in [('original_all','original'),('clean_all','clean')]:
        blocks[kind]=ab.normalize(sparse.hstack([blocks['venue'],blocks[prefix+'_authors'],blocks[prefix+'_year']],format='csr'))
    return ab.normalize(sparse.hstack([sparse.csr_matrix(x),choice['weight']*blocks[choice['kind']]],format='csr'))


def main():
    data,template=ab.load_clean_data();results=json.loads((OUT/'results.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    train=np.arange(2494);test=np.arange(2494,len(data));y=data.Label.iloc[:2494].astype(int).to_numpy()
    groups=ab.baseline.grouping(data.iloc[:2494].reset_index(drop=True))
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=CV_SEED).split(train,y,groups))
    nums=np.full(2494,-1);configs=ab.configurations();inner={c['key']:np.zeros(2494) for c in configs}
    for k,(fi,vi) in enumerate(folds):
        assert set(groups[fi]).isdisjoint(groups[vi]);nums[vi]=k;cache=np.load(OUT/f'input_inner{k}.npz')
        for c in configs:inner[c['key']][vi]=cache[c['key']]
    choices,_=ab.select(data,y,train,nums,inner,configs);base=choices['selection_procedure']
    assert base==results['base']
    mc=meta.configurations();mi={c['key']:np.zeros(2494) for c in mc}
    for k,(_,vi) in enumerate(folds):
        cache=np.load(OUT/f'metadata_inner{k}.npz')
        for c in mc:mi[c['key']][vi]=cache[c['key']]
    choices,_=ab.select(data,y,train,nums,mi,mc);chosen=choices['selection_procedure'];source='metadata'
    if results['stage_decision']['advance_to_adapter']:
        n=load_neural();cal=nums==0
        ni={arm:np.load(OUT/f'neural_inner0_{arm}.npz')['values'] for arm in n.ARMS}
        choices=n.choose(data,y,train[cal],nums[cal],ni);candidate=max(choices.values(),key=lambda x:x['inner_score'])
        if candidate['inner_score']>chosen['inner_score']:chosen=candidate;source='adapter'
    assert source==results['source'] and chosen==results['chosen']
    submission=pd.read_csv(OUT/'submission_workflow.csv',dtype={'id':str})
    assert len(submission)==596 and submission.id.is_unique and np.array_equal(submission.id,template.id)
    assert np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
    if source=='metadata':
        bundle=joblib.load(OUT/'model_workflow.joblib');audit_path=ab.OUT/'metadata_audit.csv'
        assert bundle['metadata_sha256']==hashlib.sha256(audit_path.read_bytes()).hexdigest()
        assert bundle['data_sha256']==ab.baseline.input_digest(data)
        audit=pd.read_csv(audit_path,dtype={'id':str},keep_default_na=False)
        x=rebuild_classical(bundle,data,audit,test)
        cache=np.load(OUT/'test_features.npz')
        saved=cache['dense'] if 'dense' in cache.files else sparse.load_npz(OUT/'test_features.npz')
        a=x.toarray() if sparse.issparse(x) else x;b=saved.toarray() if sparse.issparse(saved) else saved
        np.testing.assert_allclose(a,b,atol=1e-12)
        ratings=bundle['model'].predict(x);p=ab.priority(ratings,chosen['threshold'])
    else:
        tokenizer,tokens,frozen=n.prepare(data)
        values=n.predict_saved(data,test,OUT/'neural_model',tokenizer,tokens,frozen)
        saved=np.load(OUT/'neural_test_predictions.npz')['values'];np.testing.assert_allclose(values,saved,atol=1e-6)
        ratings=1+4*values[:,1];p=n.make_priority(values,chosen['threshold'])
    pred=pd.DataFrame(dict(id=data.id.iloc[test],priority=p))
    aligned=template[['id']].merge(pred,on='id',validate='one_to_one')
    np.testing.assert_allclose(aligned.priority,submission.priority,atol=1e-6 if source=='adapter' else 1e-10)
    np.testing.assert_array_equal(aligned.priority.gt(0),submission.priority.gt(0))
    np.testing.assert_array_equal(np.lexsort((aligned.id.to_numpy(),-aligned.priority.to_numpy())),
                                  np.lexsort((submission.id.to_numpy(),-submission.priority.to_numpy())))
    workflow=pd.read_csv(OUT/'workflow_oof.csv',dtype={'id':str});fm=pd.read_csv(OUT/'workflow_fold_metrics.csv')
    assert len(workflow)==2494 and np.array_equal(workflow.Label,y) and np.array_equal(workflow.id,data.id.iloc[:2494])
    stages=pd.read_csv(meta.OUT/'fold_metrics.csv');stagepred=pd.read_csv(meta.OUT/'oof_predictions.csv',dtype={'id':str})
    if results['stage_decision']['advance_to_adapter']:
        nm=pd.read_csv(n.OUT/'fold_metrics.csv');npred=pd.read_csv(n.OUT/'oof_predictions.csv',dtype={'id':str})
    for fold,row in fm.set_index('outer_fold').iterrows():
        selected=stages[stages.outer_fold.eq(fold)&stages.family.eq('selection_procedure')].iloc[0]
        expected='metadata';values=stagepred.selection_procedure_priority
        if results['stage_decision']['advance_to_adapter']:
            alternative=nm[nm.outer_fold.eq(fold)].sort_values('inner_score',ascending=False,kind='stable').iloc[0]
            if alternative.inner_score>selected.inner_score:selected=alternative;expected='adapter';values=npred[selected.family+'_priority']
        mask=workflow.outer_fold.eq(fold)
        assert row.source==expected and row.chosen_family==selected.family
        np.testing.assert_allclose(workflow.loc[mask,'priority'],values[mask],atol=1e-10)
        metric=ab.evaluate(y[mask],workflow.priority[mask],workflow.id[mask])
        for key,value in metric.items():assert abs(value-row[key])<1e-9
    verified=dict(original_labels_and_grouped_final_cv=True,final_selection_recomputed_from_inner_only=True,
        model_reload_and_test_features_rebuilt=True,submission_n=596,template_order_exact=True,
        finite_nonnegative_priorities=True,identical_selection_and_full_ranking_after_reload=True,
        full_workflow_outer_choices_and_metrics_recomputed=5,not_submitted_to_kaggle=True)
    (OUT/'verification.json').write_text(json.dumps(verified,indent=2)+'\n');print(json.dumps(verified,indent=2))


if __name__=='__main__':main()
