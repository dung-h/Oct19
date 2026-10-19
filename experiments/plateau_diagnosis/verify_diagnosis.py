#!/usr/bin/env python3
"""Verify subset isolation, evidence and cached-prediction metrics."""
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import run_diagnosis as run


def official(y,p,ids):
    order=sorted(np.flatnonzero(p>0),key=lambda i:(-p[i],str(ids[i])))
    weight=1/np.log2(np.arange(len(order))+2)
    q=float(np.average((y[order]-1)/4,weights=weight)) if len(order) else 0.
    npositive=int((y>1).sum());tp=int((y[order]>1).sum())
    recall=tp/npositive if npositive else 0.
    return dict(score=2*q*recall/(q+recall) if q+recall else 0.,relevance=q,coverage=recall,
                selected=len(order),TP=tp,FP=len(order)-tp,FN=npositive-tp)


def main():
    with threadpool_limits(4):
        ctx=run.previous.context();run.protocol(ctx)
        data,_,y,groups,_,retrieval,_,_=ctx
        raw,_=run.previous.ab.baseline.load_data()
        assert np.array_equal(data.Label.iloc[:2494],raw.Label.iloc[:2494])
        specs=list(run.subsets(ctx));assert len(specs)==35
        metrics=pd.read_csv(run.OUT/'learning_curve_metrics.csv')
        membership=pd.read_csv(run.OUT/'subset_membership.csv',dtype={'id':str})
        assert len(metrics)==140 and len(metrics.name.unique())==35
        lineage_count=0;count=0
        full_by_fold={s['outer_fold']:s for s in specs if s['fraction']==1.}
        for spec in specs:
            name=spec['name'];fit=spec['fit'];val=spec['val']
            assert max(fit)<2494 and max(val)<2494
            assert set(fit).issubset(full_by_fold[spec['outer_fold']]['fit'])
            assert np.array_equal(val,full_by_fold[spec['outer_fold']]['val'])
            assert set(groups[fit]).isdisjoint(groups[val])
            mem=membership[membership.name.eq(name)]
            assert np.array_equal(mem['index'],fit) and np.array_equal(mem.id,data.id.iloc[fit])
            line=json.loads((run.OUT/(name+'_lineage.json')).read_text())
            assert line['fit']==fit.tolist() and line['val']==val.tolist()
            seen=[]
            for row in line['crossfits']:
                r=np.asarray(row['refs']);q=np.asarray(row['queries'])
                assert set(r)|set(q)==set(fit) and not(set(r)&set(q))
                assert set(groups[r]).isdisjoint(groups[q])
                seen+=q.tolist();lineage_count+=1
            assert sorted(seen)==sorted(fit.tolist())
            with np.load(run.OUT/(name+'.npz')) as z:
                assert np.array_equal(z['fit'],fit) and np.array_equal(z['val'],val)
                for model in ['ridge','catboost']:
                    for part,ix in [('train',fit),('validation',val)]:
                        values=z[model+'_'+part]
                        assert np.isfinite(values).all()
                        if model=='catboost':
                            np.testing.assert_allclose(values.sum(axis=1),1,atol=1e-12)
                            assert (values>=0).all()
                            rating=values@np.arange(1,6);read=1-values[:,0]
                            priority=np.where(read>.5,10*(values@(np.arange(5)/4)),0)
                        else:
                            rating=values;read=values
                            gain=(values-1)/4;priority=np.where(gain>.25,10*gain,0)
                        expected=official(y[ix],priority,data.id.iloc[ix].to_numpy())
                        expected.update(rating_mae=float(np.abs(y[ix]-rating).mean()),
                            rating_rmse=float(np.sqrt(((y[ix]-rating)**2).mean())),
                            read_auc=roc_auc_score(y[ix]>1,read),missed_high=int(((y[ix]>=4)&(priority==0)).sum()))
                        if model=='catboost':
                            correct=values[np.arange(len(ix)),y[ix].astype(int)-1]
                            prob=np.clip(read,1e-12,1-1e-12);positive=(y[ix]>1).astype(float)
                            expected.update(multiclass_logloss=float(-np.log(correct).mean()),
                                read_logloss=float(-(positive*np.log(prob)+(1-positive)*np.log(1-prob)).mean()))
                        row=metrics[metrics.name.eq(name)&metrics.model.eq(model)&metrics.partition.eq(part)]
                        assert len(row)==1;row=row.iloc[0]
                        assert row.n_fit==len(fit) and row.n_eval==len(ix)
                        for key,value in expected.items():assert abs(value-row[key])<1e-10,(name,key,value,row[key])
                        count+=1
                if spec['fraction']==1.:
                    with np.load(run.previous.OUT/f"outer{spec['outer_fold']}_validation.npz") as old:
                        np.testing.assert_allclose(z['ridge_validation'],old['ridge'],atol=1e-10)
                        np.testing.assert_allclose(z['catboost_validation'],old['both_d4'],atol=1e-10)
        for fold in range(5):
            for seed in run.SUBSET_SEEDS:
                seq=[s for s in specs if s['outer_fold']==fold and s['subset_seed']==seed]
                assert [s['fraction'] for s in seq]==[.25,.5,.75]
                for a,b in zip(seq[:-1],seq[1:]):assert set(a['fit']).issubset(b['fit'])
        pairs=pd.read_csv(run.OUT/'content_pairs.csv',dtype={'a_id':str,'b_id':str},keep_default_na=False)
        sample=pairs[pairs.selection.eq('sample')]
        assert len(sample)==20 and sample.groupby('venue').size().eq(4).all()
        assert len(set(sample.a_id)|set(sample.b_id))==40
        assert sample.a_label.eq(1).all() and sample.b_label.ge(4).all()
        expected=run.audit_content.select_pairs(data,y,groups,retrieval)
        assert sample.pair_id.tolist()==[r['pair_id'] for r in expected]
        assert sample.a_index.tolist()==[r['a_index'] for r in expected]
        assert sample.b_index.tolist()==[r['b_index'] for r in expected]
        lookup=dict(zip(data.id.iloc[:2494],range(2494)))
        for row in pairs.itertuples():
            assert data.id.iloc[row.a_index]==row.a_id and data.id.iloc[row.b_index]==row.b_id
            assert y[row.a_index]==row.a_label and y[row.b_index]==row.b_label
        claims=pd.read_csv(run.OUT/'claim_spans.csv',dtype={'id':str},keep_default_na=False)
        for row in claims.itertuples():
            text=data.abstract.iloc[lookup[row.id]]
            assert text[row.start:row.end]==row.evidence
        reviews=pd.read_csv(run.OUT/'reviewed_pairs.csv',dtype={'a_id':str,'b_id':str},keep_default_na=False)
        assert len(reviews)==20 and reviews.preference_rationale.eq('unknown').all()
        assert set(reviews.pair_id)==set(sample.pair_id)
        for row in reviews.to_dict('records'):
            for role in ['a','b']:
                text=data.iloc[lookup[row[role+'_id']]][row[role+'_field']]
                assert text[row[role+'_start']:row[role+'_end']]==row[role+'_evidence']
        result=dict(original_labels_preserved=True,subsets_verified=35,subset_nesting=True,
            grouped_crossfit_partitions_verified=lineage_count,metrics_recomputed=count,
            full_size_predictions_match_previous_at_1e_10=True,sampled_pairs=20,
            manually_reviewed_content_pairs=20,manual_evidence_spans_verified=40,
            candidate_claim_spans_verified=len(claims),no_preference_reannotation=True,
            no_new_submission=True,review_sha256=run.feat.digest_file(run.HERE/'reviews.json'))
        (run.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
