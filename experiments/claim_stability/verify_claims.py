#!/usr/bin/env python3
"""Independent scores/choices, evidence offsets, projection lineage and final reload."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import run_claims as run
import claim_features as cf


def official(y,p,ids):
    order=sorted(np.flatnonzero(p>0),key=lambda i:(-p[i],str(ids[i])))
    w=1/np.log2(np.arange(len(order))+2)
    q=float(np.average((y[order]-1)/4,weights=w)) if len(order) else 0.
    tp=int((y[order]>1).sum());positive=int((y>1).sum());r=tp/positive if positive else 0.
    return dict(score=2*q*r/(q+r) if q+r else 0.,relevance=q,coverage=r,
                TP=tp,FP=len(order)-tp,FN=positive-tp,selected=len(order))


def choose(ctx,fit,nums,values):
    data,_,y,*_=ctx['prior'];choices={}
    for arm in run.ARMS:
        scores=[]
        for cutoff in run.THRESHOLDS:
            p=run.priority(values[arm],cutoff)
            scores.append(np.mean([official(y[fit[nums==k]],p[nums==k],data.id.iloc[fit[nums==k]].to_numpy())['score'] for k in range(4)]))
        j=int(np.argmax(scores))
        choices[arm]=dict(family=arm,threshold=float(run.THRESHOLDS[j]),inner_score=float(scores[j]))
    choices['selection_procedure']=max(choices.values(),key=lambda r:r['inner_score'])
    return choices


def inner_values(fit,nums,prefix):
    values={arm:np.full((len(fit),5),np.nan) for arm in run.ARMS}
    for k in range(4):
        mask=nums==k
        with np.load(run.OUT/(prefix+f'_inner{k}_predictions.npz')) as z:
            assert np.array_equal(z['fit'],fit[~mask]) and np.array_equal(z['pred'],fit[mask])
            for arm in run.ARMS:values[arm][mask]=z[arm]
    return values


def main():
    with threadpool_limits(4):
        ctx=run.context();proto=run.protocol(ctx);data,template,y,groups,title,retrieval,old_content,_=ctx['prior']
        raw,_=run.previous.ab.baseline.load_data()
        assert np.array_equal(data.Label.iloc[:2494],raw.Label.iloc[:2494])
        texts=pd.read_csv(run.OUT/'extracted_texts.csv',dtype=str,keep_default_na=False)
        pd.testing.assert_frame_equal(texts,ctx['texts'],check_dtype=False)
        exported=pd.read_csv(run.OUT/'sentence_evidence.csv',dtype={'id':str},keep_default_na=False)
        assert len(exported)==len(ctx['evidence'])
        lookup={(r.split,r.id):i for i,r in data.iterrows()}
        for row in exported.itertuples():
            text=data.iloc[lookup[(row.split,row.id)]][row.field]
            assert text[row.start:row.end]==row.evidence
        z=np.load(run.OUT/'sentence_embeddings.npz')
        assert np.array_equal(z['keys'],(data.split+':'+data.id).to_numpy(dtype=str))
        meta=json.loads((run.OUT/'embedding_manifest.json').read_text())
        assert meta['input_sha256']==hashlib.sha256(ctx['texts'].to_csv(index=False).encode()).hexdigest()
        assert meta['revision']==proto['encoder']['revision']
        ctx['vectors']={k:z[k] for k in ['prefix','claims']}
        for kind,v in ctx['vectors'].items():
            assert v.shape==(3090,3,384) and np.isfinite(v).all()
            for j,role in enumerate(cf.ASPECTS):
                available=ctx['texts'][kind+'_'+role].str.len().gt(0).to_numpy()
                norms=np.linalg.norm(v[:,j],axis=1)
                np.testing.assert_allclose(norms[available],1,atol=1e-5)
                assert (norms[~available]==0).all()
        lineage_count=0;projection_count=0
        for path in sorted(run.OUT.glob('*_features.joblib')):
            pack=joblib.load(path);fit=pack['fit'];pred=pack['pred']
            if (pred<2494).all():assert set(groups[fit]).isdisjoint(groups[pred])
            line=json.loads(path.with_name(path.name.replace('_features.joblib','_lineage.json')).read_text())
            assert line['projection_fit']==fit.tolist()
            seen=[]
            for row in line['crossfits']:
                r=np.asarray(row['refs']);q=np.asarray(row['queries'])
                assert set(r)|set(q)==set(fit) and not(set(r)&set(q))
                assert set(groups[r]).isdisjoint(groups[q])
                seen+=q.tolist();lineage_count+=1
            assert sorted(seen)==sorted(fit.tolist())
            for kind in ['prefix','claims']:
                models=pack['projections'][kind]['models']
                projected=cf.project_saved(ctx['vectors'][kind],pred,models)
                pd.testing.assert_frame_equal(projected,pack['projections'][kind]['b'],atol=1e-12,rtol=1e-12)
                assert all(m.n_features_in_==384 and m.n_components==16 for m in models)
                projection_count+=3
            for part,ix in [('a',fit),('b',pred)]:
                expected=run.repair_frame(pack['original_'+part],ctx['repaired'],ix)
                pd.testing.assert_frame_equal(expected,pack['repaired_'+part])
        m=pd.read_csv(run.OUT/'fold_metrics.csv');oof=pd.read_csv(run.OUT/'oof_predictions.csv',dtype={'id':str})
        search=pd.read_csv(run.OUT/'inner_search.csv');assert len(m)==90
        assert np.array_equal(oof.id,data.id.iloc[:2494]) and np.array_equal(oof.Label.astype(int),y[:2494])
        count=0
        for fold,fit,val,nums in run.previous.outer_members(data):
            choices=choose(ctx,fit,nums,inner_values(fit,nums,f'outer{fold}'))
            for family,choice in choices.items():
                if family=='selection_procedure':continue
                selected=search[search.outer_fold.eq(fold)&search.family.eq(family)].sort_values('threshold')
                assert len(selected)==len(run.THRESHOLDS)
                best=selected.iloc[int(np.argmax(selected.inner_score))]
                assert abs(best.threshold-choice['threshold'])<1e-12 and abs(best.inner_score-choice['inner_score'])<1e-12
            for seed in run.SEEDS:
                path=run.OUT/(f'outer{fold}_validation_predictions.npz' if seed==run.SEEDS[0] else f'outer{fold}_seed{seed}.npz')
                with np.load(path) as cache:
                    for family,choice in choices.items():
                        arm=choice['family'];prob=cache[arm];train=cache['train_'+arm]
                        np.testing.assert_allclose(prob.sum(axis=1),1,atol=1e-12)
                        assert np.isfinite(prob).all() and np.isfinite(train).all()
                        p=run.priority(prob,choice['threshold'])
                        np.testing.assert_allclose(oof.loc[val,f'{family}_seed{seed}_priority'],p,atol=1e-11)
                        row=m[m.outer_fold.eq(fold)&m.seed.eq(seed)&m.family.eq(family)].iloc[0]
                        assert row.chosen_arm==arm and abs(row.threshold-choice['threshold'])<1e-12
                        result=official(y[val],p,data.id.iloc[val].to_numpy())
                        va=roc_auc_score(y[val]>1,1-prob[:,0]);ta=roc_auc_score(y[fit]>1,1-train[:,0])
                        result.update(read_auc=va,train_auc=ta,auc_gap=ta-va,
                            validation_logloss=float(-np.log(prob[np.arange(len(val)),y[val].astype(int)-1]).mean()),
                            train_logloss=float(-np.log(train[np.arange(len(fit)),y[fit].astype(int)-1]).mean()),
                            missed_high=int(((y[val]>=4)&(p==0)).sum()))
                        for key,value in result.items():assert abs(value-row[key])<1e-10,(fold,seed,family,key)
                        count+=1
        old=pd.read_csv(run.previous.OUT/'oof_predictions.csv',dtype={'id':str})
        np.testing.assert_allclose(oof[f'reference_seed{run.SEEDS[0]}_priority'],old.both_priority,atol=1e-11)
        fit=np.arange(2494);pred=np.arange(2494,len(data))
        nums=pd.read_csv(run.previous.OUT/'final_inner_splits.csv').inner_fold.to_numpy()
        choices=choose(ctx,fit,nums,inner_values(fit,nums,'final'))
        final_choice=json.loads((run.OUT/'final_choice.json').read_text())
        assert choices['selection_procedure']==final_choice
        recipe=joblib.load(run.OUT/'final_recipe.joblib')
        assert recipe['protocol_sha256']==cf.digest(run.OUT/'protocol.json')
        base=run.previous.feat.rebuild_prediction(data,y,groups,title,retrieval,old_content,fit,pred,recipe['base_recipe'])
        arm=final_choice['family']
        rebuilt=base if arm in ['reference','regularized'] else run.repair_frame(base,ctx['repaired'],pred)
        if arm in ['prefix','claims']:
            text=cf.project_saved(ctx['vectors'][arm],pred,recipe['projections'][arm])
            rebuilt=pd.concat([rebuilt,text],axis=1)
        pack=joblib.load(run.OUT/'final_test_features.joblib')
        pd.testing.assert_frame_equal(base,pack['original_b'],check_dtype=False,atol=1e-11,rtol=1e-11)
        pd.testing.assert_frame_equal(rebuilt,run.frame(pack,arm,'b'),check_dtype=False,atol=1e-11,rtol=1e-11)
        model=CatBoostClassifier();model.load_model(str(run.OUT/'final_classifier.cbm'))
        assert np.array_equal(model.classes_,np.arange(5))
        probabilities=model.predict_proba(rebuilt[recipe['columns']]);priority=run.priority(probabilities,final_choice['threshold'])
        detailed=pd.read_csv(run.OUT/'test_predictions.csv',dtype={'id':str})
        np.testing.assert_allclose(probabilities,detailed[[f'p{k}' for k in range(1,6)]],atol=1e-11)
        submission=pd.read_csv(run.OUT/'submission_candidate.csv',dtype={'id':str})
        assert len(submission)==596 and submission.id.is_unique and np.array_equal(submission.id,template.id)
        mapping=dict(zip(data.id.iloc[pred],priority));expected=np.array([mapping[i] for i in submission.id])
        np.testing.assert_allclose(submission.priority,expected,atol=1e-11)
        assert np.isfinite(expected).all() and (expected>=0).all()
        def rank(p):return sorted(np.flatnonzero(p>0),key=lambda i:(-p[i],str(submission.id.iloc[i])))
        assert rank(expected)==rank(submission.priority.to_numpy())
        result=dict(original_labels_preserved=True,verbatim_evidence_spans_verified=len(exported),
            inherited_grouped_crossfits_verified=lineage_count,fit_only_svd_projections_checked=projection_count,
            outer_metrics_recomputed=count,reference_primary_matches_previous=True,
            secondary_cutoffs_and_arm_choices_held_fixed=True,inner_decisions_recomputed=True,
            final_features_rebuilt=True,final_model_reloaded=True,identical_full_ranking=True,
            n_submission=596,positive_priority=int((expected>0).sum()),no_kaggle_upload=True)
        (run.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
