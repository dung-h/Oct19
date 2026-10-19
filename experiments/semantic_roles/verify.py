#!/usr/bin/env python3
"""Independent metric/threshold replay, evidence and model reload verification."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import experiment as e
import features as f
from roles import OUT,source,ROLES,spans,digest,materialize


def independent_metric(labels,priority,ids):
    labels=np.asarray(labels,int);priority=np.asarray(priority,float)
    selected=priority>0;positive=labels>=2
    tp=int(np.count_nonzero(positive&selected));fp=int(np.count_nonzero(~positive&selected))
    fn=int(np.count_nonzero(positive&~selected));tn=int(np.count_nonzero(~positive&~selected))
    table=pd.DataFrame(dict(id=np.asarray(ids,str),priority=priority,gain=(labels-1)/4))
    table=table[table.priority>0].sort_values(['priority','id'],ascending=[False,True],kind='stable')
    discount=1/np.log2(np.arange(2,len(table)+2))
    q=float((table.gain.to_numpy()*discount).sum()/discount.sum()) if len(table) else 0.
    r=tp/int(positive.sum()) if positive.any() else 0.
    return dict(total=len(labels),accepted=tp+fp,dropped=fn+tn,TP=tp,FP=fp,FN=fn,TN=tn,total_error=fp+fn,
        missed_high=int(((labels>=4)&~selected).sum()),H=2*q*r/(q+r) if q+r else 0.,Q=q,R=r)


def independent_choice(y,p,inc):
    y=np.asarray(y,int);old=inc>.55
    budget=int(((y>=2)&~old).sum());high_budget=int(((y>=4)&~old).sum())
    eligible=[]
    for cutoff in sorted(set([0.]+p.tolist())):
        chosen=p>cutoff
        fp=int(np.count_nonzero(chosen&(y==1)))
        fn=int(np.count_nonzero(~chosen&(y>=2)))
        high=int(np.count_nonzero(~chosen&(y>=4)))
        if fn<=budget and high<=high_budget:
            eligible.append((fp,fn,high,-cutoff))
    winner=min(eligible)
    return dict(FP=winner[0],FN=winner[1],missed_high=winner[2],threshold=-winner[3],reference_FN=budget,reference_high_FN=high_budget)


def main():
    ctx=e.context();protocol=e.protocol(ctx)
    original=ctx['data'];text=ctx['texts']
    evidence=pd.read_csv(OUT/'evidence.csv',dtype={'id':str},keep_default_na=False)
    lookup={(r.split,r.id):r for r in original.itertuples()}
    for row in evidence.itertuples():
        raw=getattr(lookup[(row.split,row.id)],row.field)
        assert raw[row.start:row.end]==row.text
    assert len(evidence)==sum(text[role].str.len().gt(0).sum() for role in ROLES)
    metrics=pd.read_csv(OUT/'fold_metrics.csv');oof=pd.read_csv(OUT/'outcomes_by_id.csv',dtype={'id':str})
    assert len(metrics)==25 and len(oof)==2494*5
    outer=source.outer_members(ctx);final_fit,final_pred,final_nums=source.final_partition(ctx)
    blocks=[(f'outer{fold}',fit,pred,nums) for fold,fit,pred,nums in outer]+[('final',final_fit,final_pred,final_nums)]
    choices_checked=reloads=rows_checked=0
    for prefix,fit,pred,nums in blocks:
        inner={k:np.full(len(fit),np.nan) for k in ['text','roles','inc']}
        for k in range(4):
            mask=nums==k;name=f'{prefix}_inner{k}';cache=np.load(OUT/(name+'_probabilities.npz'))
            np.testing.assert_array_equal(cache['fit'],fit[~mask]);np.testing.assert_array_equal(cache['pred'],fit[mask])
            assert set(ctx['groups'][fit[~mask]]).isdisjoint(ctx['groups'][fit[mask]])
            for arm in ['text','roles']:inner[arm][mask]=cache[arm]
            inner['inc'][mask]=1-e.incumbent(name,fit[~mask],fit[mask],ctx)[:,0]
        saved_choices=json.loads((OUT/(prefix+'_choices.json')).read_text())
        branch_inner=dict(text_raw=inner['text'],roles_raw=inner['roles'],
            text_blend=(inner['text']+inner['inc'])/2,roles_blend=(inner['roles']+inner['inc'])/2)
        for branch,p in branch_inner.items():
            choice=independent_choice(ctx['y'][fit],p,inner['inc'])
            for key,value in choice.items():assert abs(saved_choices[branch][key]-value)<1e-12,(prefix,branch,key)
            choices_checked+=1
        name=prefix+'_validation' if prefix!='final' else 'final_test'
        pack=joblib.load(OUT/(name+'_models.joblib'));cache=np.load(OUT/(name+'_probabilities.npz'))
        np.testing.assert_array_equal(pack['fit'],fit);np.testing.assert_array_equal(pack['pred'],pred)
        np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['pred'],pred)
        assert pack['protocol_sha256']==str(cache['protocol_sha256'])==digest(OUT/'protocol.json')
        spaces=pack['spaces']
        if prefix=='final':
            rebuilt=f.fit_spaces(text,fit)
            for (_,_,a),(_,_,b) in zip(spaces,rebuilt):
                assert (a is None)==(b is None)
                if a is not None:
                    assert a.vocabulary_==b.vocabulary_
                    np.testing.assert_allclose(a.idf_,b.idf_,atol=1e-14)
        query=f.transform(text,ctx['embedding'],pred,spaces);raw={}
        for arm in ['text','roles']:
            raw[arm]=pack['models'][arm].predict_proba(query[arm])[:,1]
            np.testing.assert_allclose(raw[arm],cache[arm],atol=1e-12);reloads+=1
        inc=e.incumbent(name,fit,pred,ctx)
        branch_outer=dict(incumbent=1-inc[:,0],text_raw=raw['text'],roles_raw=raw['roles'],
            text_blend=(raw['text']+1-inc[:,0])/2,roles_blend=(raw['roles']+1-inc[:,0])/2)
        ids=original.id.iloc[pred].to_numpy(str)
        for branch,p in branch_outer.items():
            cutoff=.55 if branch=='incumbent' else saved_choices[branch]['threshold']
            selected=p>cutoff;priority=np.where(selected,1+9*np.sum(inc*np.linspace(0,1,5),axis=1),0.)
            if prefix=='final':
                if branch=='incumbent':continue
                detail=pd.read_csv(OUT/('test_'+branch+'.csv'),dtype={'id':str})
                submission=pd.read_csv(OUT/('submission_'+branch+'.csv'),dtype={'id':str})
                assert detail.id.tolist()==ids.tolist() and submission.id.tolist()==ctx['template'].id.tolist()
                assert len(submission)==596 and submission.id.is_unique and np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
                np.testing.assert_array_equal(detail.selected,selected)
                np.testing.assert_allclose(detail.priority,priority,atol=1e-12)
                expected=dict(zip(ids,priority));np.testing.assert_allclose(submission.priority,[expected[k] for k in submission.id],atol=1e-12)
                actual_order=detail[detail.priority>0].sort_values(['priority','id'],ascending=[False,True]).id.tolist()
                expected_order=[str(ids[i]) for i in sorted(np.flatnonzero(selected),key=lambda i:(-priority[i],str(ids[i])))]
                assert actual_order==expected_order
            else:
                fold=int(prefix[5:]);part=oof[(oof.outer_fold==fold)&oof.branch.eq(branch)]
                assert part.id.tolist()==ids.tolist()
                np.testing.assert_array_equal(part.Label,ctx['y'][pred].astype(int))
                np.testing.assert_allclose(part.read_probability,p,atol=1e-12)
                np.testing.assert_allclose(part.priority,priority,atol=1e-12)
                expected=independent_metric(ctx['y'][pred],priority,ids)
                row=metrics[(metrics.outer_fold==fold)&metrics.branch.eq(branch)].iloc[0]
                for key,value in expected.items():assert abs(row[key]-value)<1e-12,(prefix,branch,key)
                rows_checked+=1
    baseline=metrics[metrics.branch.eq('incumbent')]
    assert int(baseline.FP.sum())==338 and int(baseline.FN.sum())==313 and int(baseline.missed_high.sum())==38
    assert abs(baseline.H.mean()-.6402759969)<1e-9
    expected_sha='8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf'
    assert digest(source.INC/'submission_candidate.csv')==expected_sha
    result=dict(status='PASSED',evidence_spans_checked=len(evidence),outer_metric_rows=rows_checked,
        thresholds_independently_recomputed=choices_checked,reloaded_classifiers=reloads,
        final_vocabularies_rebuilt=True,final_submissions=4,incumbent_unchanged=True,uploaded=False,
        extraction_role_correctness='Schema/source exactness checked; semantic correctness requires descriptive review, not guaranteed by schema.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(4):main()
