#!/usr/bin/env python3
"""Recompute metrics, lineage, inner choices, evidence, and final inference."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

import run_augmentation as ex
import features as feat


def metric(y,p,ids):
    """Independent implementation of the detailed Evaluation, with text-ID ties."""
    selected=sorted((i for i in range(len(p)) if p[i]>0),key=lambda i:(-p[i],str(ids[i])))
    npositive=sum(v>1 for v in y)
    tp=sum(y[i]>1 for i in selected)
    weights=[1/np.log2(r+2) for r in range(len(selected))]
    q=sum(w*(y[i]-1)/4 for w,i in zip(weights,selected))/sum(weights) if weights else 0.
    recall=tp/npositive if npositive else 0.
    return dict(score=2*q*recall/(q+recall) if q+recall else 0.,relevance=q,coverage=recall,
                selected=len(selected),TP=tp,FP=len(selected)-tp,FN=npositive-tp)


def rank(p,ids):
    return [str(ids[i]) for i in sorted(np.flatnonzero(p>0),key=lambda i:(-p[i],str(ids[i])))]


def check_search(ctx,fit,nums,predictions,reported):
    data,_,y,*_=ctx
    best=[]
    for config in ex.CONFIGS:
        thresholds=ex.ab.THRESHOLDS if config['family']=='ridge' else np.arange(0,.901,.05)
        scores=[]
        for t in thresholds:
            p=ex.priorities(config,predictions[config['key']],t)
            scores.append(np.mean([metric(y[fit[nums==k]],p[nums==k],data.id.iloc[fit[nums==k]].to_numpy())['score'] for k in range(4)]))
        ix=int(np.argmax(scores))
        row=reported[reported.key.eq(config['key'])].iloc[0]
        assert abs(row.threshold-thresholds[ix])<1e-12
        assert abs(row.inner_score-scores[ix])<1e-12
        best.append(dict(config,threshold=float(thresholds[ix]),inner_score=float(scores[ix])))
    choices={}
    for row in best:
        if row['family'] not in choices or row['inner_score']>choices[row['family']]['inner_score']:
            choices[row['family']]=row
    choices['selection_procedure']=max(best,key=lambda r:r['inner_score'])
    return choices


def assemble_inner(fit,nums,prefix):
    result={c['key']:np.full((len(fit),5),np.nan) if c['family']!='ridge' else np.full(len(fit),np.nan) for c in ex.CONFIGS}
    for k in range(4):
        with np.load(ex.OUT/(prefix+f'_inner{k}.npz')) as z:
            mask=nums==k
            assert np.array_equal(z['fit'],fit[~mask]) and np.array_equal(z['pred'],fit[mask])
            for key in result:result[key][mask]=z[key]
    return result


def main():
    with threadpool_limits(4):
        ctx=ex.context();data,template,y,groups,title,retrieval,content,evidence=ctx
        ex.write_protocol(data)
        prior_path=ex.OUT/'protocol_before_entrypoint_rename.json'
        if prior_path.exists():
            prior=json.loads(prior_path.read_text())
            current=json.loads((ex.OUT/'protocol.json').read_text())
            for key in ['data_sha256','split_sha256','configs','seed','iterations','learning_rate',
                        'l2_leaf_reg','rules','crossfit_splits','neighbor_k','cat_thresholds','ridge_thresholds']:
                assert prior[key]==current[key]
            assert prior['source_sha256']['features.py']==current['source_sha256']['features.py']
        raw,_=ex.ab.baseline.load_data()
        assert np.array_equal(data.Label.iloc[:2494],raw.Label.iloc[:2494])
        exported=pd.read_csv(ex.OUT/'content_evidence.csv',dtype={'id':str},keep_default_na=False)
        assert len(exported)==len(evidence)
        lookup={(r.split,r.id):i for i,r in data.iterrows()}
        for r in exported.itertuples():
            text=data.iloc[lookup[(r.split,r.id)]][r.field]
            assert text[r.start:r.end]==r.evidence
            assert feat.RULES[r.aspect][r.category]
        lineage_count=0
        for path in sorted(ex.OUT.glob('*_lineage.json')):
            v=json.loads(path.read_text());f=np.asarray(v['fit']);p=np.asarray(v['pred'])
            if np.all(p<2494):assert set(groups[f]).isdisjoint(groups[p])
            seen=[]
            for row in v['crossfits']:
                refs=np.asarray(row['refs']);queries=np.asarray(row['queries'])
                assert set(refs)|set(queries)==set(f)
                assert not (set(refs)&set(queries))
                assert set(groups[refs]).isdisjoint(groups[queries])
                seen+=queries.tolist();lineage_count+=1
            assert sorted(seen)==sorted(f.tolist())
        m=pd.read_csv(ex.OUT/'fold_metrics.csv');s=pd.read_csv(ex.OUT/'inner_search.csv')
        oof=pd.read_csv(ex.OUT/'oof_predictions.csv',dtype={'id':str})
        assert np.array_equal(oof.id,data.id.iloc[:2494]) and np.array_equal(oof.Label.astype(int),y[:2494])
        count=0
        for fold,fit,val,nums in ex.outer_members(data):
            choices=check_search(ctx,fit,nums,assemble_inner(fit,nums,f'outer{fold}'),s[s.outer_fold.eq(fold)])
            with np.load(ex.OUT/f'outer{fold}_validation.npz') as z:
                for family,choice in choices.items():
                    row=m[m.outer_fold.eq(fold)&m.family.eq(family)].iloc[0]
                    assert row.key==choice['key'] and abs(row.threshold-choice['threshold'])<1e-12
                    p=ex.priorities(choice,z[choice['key']],choice['threshold'])
                    np.testing.assert_allclose(p,oof.loc[val,family+'_priority'],atol=1e-12)
                    result=metric(y[val],p,data.id.iloc[val].to_numpy())
                    for key,value in result.items():assert abs(value-row[key])<1e-10
                    assert row.missed_high==int(((y[val]>=4)&(p==0)).sum())
                    count+=1
        fit=np.arange(2494);pred=np.arange(2494,len(data))
        folds=pd.read_csv(ex.OUT/'final_inner_splits.csv',dtype={'id':str})
        assert np.array_equal(folds.id,data.id.iloc[:2494])
        nums=folds.inner_fold.to_numpy()
        choices=check_search(ctx,fit,nums,assemble_inner(fit,nums,'final'),pd.read_csv(ex.OUT/'final_search.csv'))
        choice=json.loads((ex.OUT/'final_choice.json').read_text())
        assert choice==choices['selection_procedure']
        bundle=joblib.load(ex.OUT/'final_recipe.joblib')
        rebuilt=feat.rebuild_prediction(data,y,groups,title,retrieval,content,fit,pred,bundle['recipe'])
        cached=joblib.load(ex.OUT/'final_test_features.joblib')
        pd.testing.assert_frame_equal(rebuilt,cached['b'],check_dtype=False,atol=1e-11,rtol=1e-11)
        if choice['family']=='ridge':
            a,b,_=feat.ridge_space(data,title,fit,pred)
            values=bundle['recipe']['ridge'].predict(b)
        else:
            model=CatBoostClassifier();model.load_model(str(ex.OUT/'final_classifier.cbm'))
            values=model.predict_proba(rebuilt[bundle['columns']])
        priorities=ex.priorities(choice,values,choice['threshold'])
        submission=pd.read_csv(ex.OUT/'submission_candidate.csv',dtype={'id':str})
        assert len(submission)==596 and submission.id.is_unique
        assert np.array_equal(submission.id,template.id)
        assert np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
        byid=dict(zip(data.id.iloc[pred],priorities));expected=np.array([byid[i] for i in submission.id])
        np.testing.assert_allclose(submission.priority,expected,atol=1e-11,rtol=1e-11)
        assert rank(submission.priority.to_numpy(),submission.id.to_numpy())==rank(expected,submission.id.to_numpy())
        # Labels of all held-out rows changed to invalid values: feature output must stay identical.
        pack=joblib.load(ex.OUT/'outer0_validation_features.joblib');f=pack['fit'];q=pack['pred'][:8]
        mutated=y.copy();mutated[pack['pred']]=999
        h=feat.history(data,mutated,groups,title,retrieval,f,q)
        np.testing.assert_allclose(h.ridge_rating,pack['b'].ridge_rating.iloc[:8],atol=1e-10)
        for col in h.drop(columns='ridge_rating'):
            np.testing.assert_allclose(h[col],pack['b']['neighbor_'+col].iloc[:8],atol=1e-10)
        result=dict(original_labels_unchanged=True,evidence_spans_verified=len(exported),
            grouped_crossfit_partitions_verified=lineage_count,outer_metrics_recomputed=count,
            all_inner_choices_recomputed=True,heldout_label_perturbation_invariant=True,
            final_features_rebuilt=True,model_reloaded=True,identical_full_ranking=True,
            n_submission=596,positive_priority=int((expected>0).sum()),not_submitted_to_kaggle=True)
        (ex.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
