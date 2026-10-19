#!/usr/bin/env python3
"""Independent score/selection checks and feature provenance/inference verification."""
import json
import importlib.metadata
import re

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import run_repairs as r
feat=r.feat


def metric(y,priority,ids):
    chosen=sorted(np.flatnonzero(priority>0),key=lambda i:(-priority[i],str(ids[i])))
    weights=1/np.log2(np.arange(len(chosen))+2)
    q=float(np.sum(weights*(y[chosen]-1)/4)/weights.sum()) if chosen else 0.
    tp=int(sum(y[i]>1 for i in chosen));total=int(sum(y>1))
    recall=tp/total if total else 0.
    return dict(score=2*q*recall/(q+recall) if q+recall else 0.,relevance=q,
        coverage=recall,selected=len(chosen),TP=tp,FP=len(chosen)-tp,FN=total-tp,
        missed_high=int(((y>=4)&(priority==0)).sum()))


def independent_choices(ctx,fit,nums,values):
    choices={}
    for arm in r.ARMS:
        scores=[]
        for threshold in r.THRESHOLDS:
            p=r.priority(values[arm],threshold)
            scores.append(np.mean([metric(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]].to_numpy())['score'] for k in range(4)]))
        i=int(np.argmax(scores))
        choices[arm]=dict(arm=arm,threshold=float(r.THRESHOLDS[i]),inner_score=float(scores[i]))
    choices['selection_procedure']=max(choices.values(),key=lambda v:v['inner_score']).copy()
    return choices


def check_choices(computed,path):
    saved=json.loads(path.read_text())
    assert computed.keys()==saved.keys()
    for key in computed:
        assert saved[key]['arm']==computed[key]['arm']
        for field in ['threshold','inner_score']:
            assert abs(saved[key][field]-computed[key][field])<1e-12


def main():
    with threadpool_limits(4):
        ctx=r.context();locked=r.protocol(ctx)
        raw,_=r.prior.ab.baseline.load_data()
        np.testing.assert_array_equal(ctx['data'].Label.iloc[:2494],raw.Label.iloc[:2494])
        assert np.isnan(ctx['y'][2494:]).all()
        # These generic phrases test the observed meaning collision; they do
        # not assign preference labels to any validation or test paper.
        probes=['predicate transformer','abstract transformers','distribution transformer','monad transformer']
        for text in probes:
            assert re.search(r.prior.feat.RULES['method']['neural'],text,re.I)
            assert not re.search(feat.NEURAL,text,re.I)
        assert re.search(feat.NEURAL,'large language models',re.I)
        assert re.search(feat.NEURAL,'neural networks',re.I)
        assert not re.search(feat.EDUCATION,'PhD student',re.I)
        assert re.search(feat.EDUCATION,'teaching a school curriculum',re.I)
        identity={(row.split,row.id):i for i,row in ctx['data'].iterrows()}
        evidence=pd.read_csv(r.OUT/'rule_evidence.csv',dtype={'id':str},keep_default_na=False)
        for row in evidence.itertuples():
            text=ctx['data'].iloc[identity[(row.split,row.id)]][row.field]
            assert text[row.start:row.end]==row.evidence
            assert re.search(ctx['rules'][row.aspect][row.category],row.evidence,re.I)
        repaired=pd.read_csv(r.OUT/'repaired_content.csv',dtype={'id':str})
        pd.testing.assert_frame_equal(repaired[ctx['fixed_content'].columns],ctx['fixed_content'],check_dtype=False)
        control_count=0;lineage_count=0;views_checked=0;projection_checks=0
        audits=[]
        for name,fit,pred in r.jobs(ctx):
            old=r.parent_pack(name,fit,pred)
            for side,indices in [('a',fit),('b',pred)]:
                assert old[side].venue.tolist()==ctx['data'].venue.iloc[indices].tolist()
                np.testing.assert_array_equal(old[side].year,ctx['data'].year.iloc[indices])
            rich=joblib.load(r.OUT/(name+'_rich.joblib'))
            z=np.load(r.OUT/(name+'_predictions.npz'))
            assert str(z['protocol_sha256'])==feat.digest(r.OUT/'protocol.json')
            np.testing.assert_allclose(z['reference'],r.parent_prediction(ctx,name,fit,pred),atol=1e-10,rtol=1e-10)
            assert np.array_equal(rich['fit'],fit) and np.array_equal(rich['pred'],pred)
            seen=[]
            for split,history in zip(old['lineage'],rich['histories']):
                refs=np.asarray(split['refs']);queries=np.asarray(split['queries'])
                assert set(refs)|set(queries)==set(fit)
                assert set(ctx['groups'][refs]).isdisjoint(ctx['groups'][queries])
                assert not (set(refs)&set(pred))
                seen.extend(queries.tolist());lineage_count+=1
                expected=np.array([len(refs)-np.sum(ctx['groups'][refs]==group) for group in ctx['groups'][pred]])
                np.testing.assert_array_equal(history.neighbor_retrieval_all_references,expected)
                views_checked+=1
            assert sorted(seen)==sorted(fit.tolist())
            assert np.all(rich['abstract_a'][ctx['data'].abstract.iloc[fit].eq('')]==0)
            assert np.all(rich['abstract_b'][ctx['data'].abstract.iloc[pred].eq('')]==0)
            projection=rich['projection']
            for field,dimensions in [('title',128),('abstract',64)]:
                value=feat.transform_svd(projection[field+'_svd'],projection[field+'_vectorizer'].transform(ctx['data'][field].iloc[pred]),dimensions)
                np.testing.assert_allclose(value,rich[field+'_b'],atol=1e-12,rtol=1e-12)
                projection_checks+=1
            for arm in r.ARMS:
                a=feat.frame(ctx,old,rich,arm,'a')
                for k in range(3 if arm in ['matched','combined'] else 1):
                    b=feat.frame(ctx,old,rich,arm,'b',k)
                    assert a.columns.tolist()==b.columns.tolist()
                assert np.isfinite(z[arm]).all() and np.allclose(z[arm].sum(axis=1),1)
            audits.append(dict(partition=name,**rich['audit']));control_count+=1
        pd.DataFrame(audits).to_csv(r.OUT/'feature_audit.csv',index=False)
        frame=pd.read_csv(r.OUT/'fold_metrics.csv')
        oof=pd.read_csv(r.OUT/'oof.csv',dtype={'id':str})
        assert oof.id.tolist()==ctx['data'].id.iloc[:2494].tolist()
        checked=choices_checked=0
        for fold,fit,pred,nums in r.prior.outer_members(ctx['data']):
            choice=independent_choices(ctx,fit,nums,r.inner(ctx,fit,nums,f'outer{fold}'))
            check_choices(choice,r.OUT/f'outer{fold}_choices.json')
            choices_checked+=len(choice)
            z=np.load(r.OUT/f'outer{fold}_validation_predictions.npz')
            for policy,item in choice.items():
                probability=z[item['arm']];p=r.priority(probability,item['threshold'])
                measures=metric(ctx['y'][pred],p,ctx['data'].id.iloc[pred].to_numpy())
                measures.update(AUC=roc_auc_score(ctx['y'][pred]>1,1-probability[:,0]),
                    train_AUC=roc_auc_score(ctx['y'][fit]>1,1-z['train_'+item['arm']][:,0]),
                    gain_MAE=np.mean(np.abs(probability@(np.arange(5)/4)-(ctx['y'][pred]-1)/4)))
                saved=frame[frame.outer_fold.eq(fold)&frame.policy.eq(policy)].iloc[0]
                assert saved.selected_arm==item['arm']
                for key,value in measures.items():
                    assert abs(saved[key]-value)<1e-10,(fold,policy,key)
                np.testing.assert_allclose(oof.loc[pred,policy+'_priority'],p,atol=1e-12)
                checked+=1
        fit=np.arange(2494);pred=np.arange(2494,len(ctx['data']))
        members=pd.read_csv(r.prior.OUT/'final_inner_splits.csv',dtype={'id':str})
        choice=independent_choices(ctx,fit,members.inner_fold.to_numpy(),r.inner(ctx,fit,members.inner_fold.to_numpy(),'final'))
        check_choices(choice,r.OUT/'final_choices.json');choices_checked+=len(choice)
        bundle=joblib.load(r.OUT/'final_recipe.joblib')
        assert bundle['choice']==choice['selection_procedure']
        assert bundle['protocol_sha256']==feat.digest(r.OUT/'protocol.json')
        old,rich=feat.rebuild_rich(ctx,bundle,r.prior.feat)
        arm=bundle['choice']['arm'];view_count=3 if arm in ['matched','combined'] else 1
        model=CatBoostClassifier();model.load_model(str(r.OUT/bundle['model_file']))
        views=[feat.frame(ctx,old,rich,arm,'b',i) for i in range(view_count)]
        assert all(v.columns.tolist()==bundle['columns'] for v in views)
        rebuilt=np.mean([model.predict_proba(v) for v in views],axis=0)
        archive=np.load(r.OUT/'final_test_predictions.npz')
        np.testing.assert_allclose(rebuilt,archive[arm],atol=1e-10,rtol=1e-10)
        changed=dict(ctx);changed['y']=ctx['y'].copy();changed['y'][pred]=1+(np.arange(len(pred))%5)
        changed['data']=ctx['data'].copy();changed['data'].loc[pred,'Label']=changed['y'][pred].astype(int).astype(str)
        other_old,other_rich=feat.rebuild_rich(changed,bundle,r.prior.feat)
        for k,value in enumerate(views):
            other=feat.frame(changed,other_old,other_rich,arm,'b',k)
            pd.testing.assert_frame_equal(value,other,atol=1e-12,rtol=1e-12)
        metadata_guards=[]
        for field in ['venue','year','authors','doi','Label']:
            altered=ctx['data'].copy()
            altered.loc[0,field]=(float(altered.loc[0,field])+1 if field=='year' else str(altered.loc[0,field])+'__audit__')
            assert feat.input_digest(altered)!=locked['full_input_sha256']
            metadata_guards.append(field)
        p=r.priority(rebuilt,bundle['choice']['threshold'])
        expected=ctx['template'][['id']].merge(pd.DataFrame(dict(id=ctx['data'].id.iloc[pred].to_numpy(),priority=p)),on='id')
        submitted=pd.read_csv(r.OUT/'submission_candidate.csv',dtype={'id':str})
        assert submitted.columns.tolist()==['id','priority'] and len(submitted)==596 and submitted.id.is_unique
        assert submitted.id.tolist()==expected.id.tolist()
        assert np.isfinite(submitted.priority).all() and submitted.priority.ge(0).all()
        np.testing.assert_allclose(submitted.priority,expected.priority,atol=1e-10,rtol=1e-10)
        assert feat.digest(r.prior.OUT/'submission_candidate.csv')==locked['incumbent_sha256']
        result=dict(status='PASSED',control_partitions_replayed=control_count,
            grouped_reference_lineages_checked=lineage_count,matched_reference_views_checked=views_checked,
            projection_transforms_checked=projection_checks,evidence_spans_checked=len(evidence),
            semantic_wording_probes_checked=len(probes)+4,
            outer_policy_rows_recomputed=checked,inner_choices_recomputed=choices_checked,
            metadata_hash_guards=metadata_guards,original_labels_unchanged=True,
            missing_abstract_vectors_zero=True,heldout_label_inference_invariant=True,
            final_features_rebuilt=True,selected_model_reloaded=True,submission_rows=596,
            library_versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','catboost','joblib']},
            incumbent_unchanged=True,uploaded=False)
        (r.OUT/'selected_model_params.json').write_text(json.dumps(model.get_all_params(),indent=2)+'\n')
        (r.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
