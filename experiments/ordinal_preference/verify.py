#!/usr/bin/env python3
"""Independent PAV/metric/selection checks and final feature rebuild/model reload."""
import json

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import isotonic_regression
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits

import run_ordinal as run


def view(raw,ordinal):
    if ordinal:
        q=np.array([isotonic_regression(row,increasing=False) for row in raw])
        p=-np.diff(np.column_stack([np.ones(len(raw)),q,np.zeros(len(raw))]),axis=1)
        return dict(probability=p,read=q[:,0],gain=q.sum(1)/4,raw=raw,tails=q)
    p=raw
    return dict(probability=p,read=1-p[:,0],gain=np.dot(p,[0,.25,.5,.75,1.]),raw=1-np.cumsum(p[:,:4],axis=1),tails=1-np.cumsum(p[:,:4],axis=1))


def metric(y,priority,ids):
    order=sorted(np.flatnonzero(priority>0),key=lambda i:(-priority[i],str(ids[i])))
    weight=np.array([1/np.log2(i+2) for i in range(len(order))])
    q=float(sum(w*(y[i]-1)/4 for w,i in zip(weight,order))/weight.sum()) if order else 0.
    npositive=int((y>1).sum());tp=sum(y[i]>1 for i in order);r=tp/npositive if npositive else 0.
    return dict(score=2*q*r/(q+r) if q+r else 0.,relevance=q,coverage=r,TP=int(tp),FP=len(order)-tp,
                FN=npositive-tp,selected=len(order),missed_high=int(((y>=4)&(priority==0)).sum()))


def priorities(v,t):return np.where(v['read']>t,10*v['gain'],0.)


def load_partition(ctx,fit,pred,name,train=False):
    z=np.load(run.OUT/(name+'.npz'))
    assert np.array_equal(z['fit'],fit) and np.array_equal(z['pred'],pred)
    info=json.loads((run.OUT/(name+'_training.json')).read_text())
    assert info['fit']==fit.tolist() and info['pred']==pred.tolist()
    assert info['feature_sha256']==run.digest(run.PRIOR/(name+'_features.joblib'))
    raw=z['raw'];v=view(raw,True)
    assert raw.shape==(len(pred),4) and np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all()
    np.testing.assert_allclose(v['tails'],z['tails'],atol=1e-14)
    np.testing.assert_allclose(v['probability'],z['probability'],atol=1e-14)
    np.testing.assert_allclose(v['tails'].sum(1),raw.sum(1),atol=1e-14)
    assert (v['probability']>=0).all()
    np.testing.assert_allclose(v['probability'].sum(1),1,atol=1e-14)
    assert info['crossing_rows']==int((np.diff(raw,axis=1)>0).any(1).sum())
    assert info['max_abs_gain_change']<1e-12
    for j,k in enumerate([2,3,4,5]):
        assert info['models'][j]['target']==k
        assert info['models'][j]['positive']==int((ctx['y'][fit]>=k).sum())
        assert info['models'][j]['negative']==int((ctx['y'][fit]<k).sum())
    return v,view(z['train_raw'],True) if train else None


def inner(ctx,fit,nums,prefix):
    arrays={'multiclass':np.full((len(fit),5),np.nan),'ordinal':np.full((len(fit),4),np.nan)}
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask]
        assert set(ctx['groups'][a]).isdisjoint(ctx['groups'][b])
        v,_=load_partition(ctx,a,b,f'{prefix}_inner{k}')
        arrays['ordinal'][mask]=v['raw']
        z=np.load(run.PRIOR/f'{prefix}_inner{k}.npz')
        assert np.array_equal(z['fit'],a) and np.array_equal(z['pred'],b)
        arrays['multiclass'][mask]=z['both_d4']
    return {f:view(a,f=='ordinal') for f,a in arrays.items()}


def choices(ctx,fit,nums,views):
    result={}
    for family in run.FAMILIES:
        rows=[]
        for t in run.CUTOFFS:
            p=priorities(views[family],t)
            h=np.mean([metric(ctx['y'][fit[nums==k]],p[nums==k],ctx['data'].id.iloc[fit[nums==k]].to_numpy())['score'] for k in range(4)])
            rows.append(dict(family=family,threshold=float(t),inner_score=float(h)))
        result[family]=max(rows,key=lambda x:x['inner_score'])
    result['selection_procedure']=max(result.values(),key=lambda x:x['inner_score']).copy()
    return result


def check_choices(a,b):
    assert a.keys()==b.keys()
    for k in a:
        assert a[k]['family']==b[k]['family']
        assert abs(a[k]['threshold']-b[k]['threshold'])<1e-12
        assert abs(a[k]['inner_score']-b[k]['inner_score'])<1e-12


def main():
    with threadpool_limits(4):
        ctx=run.context();run.protocol(ctx)
        data=ctx['data'];y=ctx['y'];groups=ctx['groups']
        original,_=run.ab.baseline.load_data()
        np.testing.assert_array_equal(data.id.iloc[:2494],original.id.iloc[:2494])
        np.testing.assert_array_equal(y[:2494],original.Label.iloc[:2494].astype(int))
        assert np.isnan(y[2494:]).all()
        lineages=0
        for name,sha in ctx['protocol']['feature_sha256'].items():
            assert run.digest(run.PRIOR/name)==sha
            source=joblib.load(run.PRIOR/name)
            f=source['fit'];p=source['pred']
            if (p<2494).all():assert set(groups[f]).isdisjoint(groups[p])
            seen=[]
            for row in source['lineage']:
                refs=np.asarray(row['refs']);query=np.asarray(row['queries'])
                assert set(refs)|set(query)==set(f) and not(set(refs)&set(query))
                assert set(groups[refs]).isdisjoint(groups[query])
                seen.extend(query.tolist());lineages+=1
            assert sorted(seen)==sorted(f.tolist())
        frame=pd.read_csv(run.OUT/'fold_metrics.csv')
        tails=pd.read_csv(run.OUT/'tail_metrics.csv')
        oof=pd.read_csv(run.OUT/'oof_predictions.csv',dtype={'id':str})
        assert oof.id.tolist()==data.id.iloc[:2494].tolist() and oof.outer_fold.notna().all()
        counts=0;tailcount=0;seen=[]
        for fold,fit,val,nums in run.previous.outer_members(data):
            seen.extend(val.tolist())
            expected=choices(ctx,fit,nums,inner(ctx,fit,nums,f'outer{fold}'))
            saved=json.loads((run.OUT/f'outer{fold}_choices.json').read_text());check_choices(expected,saved)
            ordinal,train_ordinal=load_partition(ctx,fit,val,f'outer{fold}_validation',True)
            inc,train_inc=run.incumbent(fit,val,f'outer{fold}_validation',True)
            views={'ordinal':ordinal,'multiclass':view(inc['probability'],False)}
            trains={'ordinal':train_ordinal,'multiclass':view(train_inc['probability'],False)}
            for family,choice in expected.items():
                v=views[choice['family']];t=trains[choice['family']]
                priority=priorities(v,choice['threshold'])
                actual=metric(y[val],priority,data.id.iloc[val].to_numpy())
                actual.update(AUC=roc_auc_score(y[val]>1,v['read']),logloss=log_loss(y[val]-1,v['probability'],labels=np.arange(5)),
                    gain_MAE=float(np.mean(np.abs(v['gain']-(y[val]-1)/4))),train_AUC=roc_auc_score(y[fit]>1,t['read']))
                actual['AUC_gap']=actual['train_AUC']-actual['AUC']
                row=frame[frame.outer_fold.eq(fold)&frame.family.eq(family)].iloc[0]
                assert row.selected_family==choice['family'] and abs(row.threshold-choice['threshold'])<1e-12
                for k,value in actual.items():assert abs(row[k]-value)<1e-10,(fold,family,k)
                np.testing.assert_allclose(oof.loc[val,family+'_priority'],priority,atol=1e-12)
                for k in range(5):np.testing.assert_allclose(oof.loc[val,family+f'_p{k+1}'],v['probability'][:,k],atol=1e-12)
                counts+=1
            for family,v in views.items():
                for j,k in enumerate([2,3,4,5]):
                    for stage in ['raw','tails']:
                        row=tails[tails.outer_fold.eq(fold)&tails.family.eq(family)&tails.target.eq(k)&tails.stage.eq(stage)].iloc[0]
                        assert abs(row.AUC-roc_auc_score(y[val]>=k,v[stage][:,j]))<1e-10
                        assert abs(row.Brier-np.mean((v[stage][:,j]-(y[val]>=k))**2))<1e-10
                        tailcount+=1
        assert sorted(seen)==list(range(2494))
        old=pd.read_csv(run.PRIOR/'oof_predictions.csv',dtype={'id':str})
        np.testing.assert_allclose(oof.multiclass_priority,old.both_priority,atol=1e-12)
        fit=np.arange(2494);test=np.arange(2494,len(data))
        nums=pd.read_csv(run.OUT/'final_inner_splits.csv',dtype={'id':str}).inner_fold.to_numpy()
        choice=choices(ctx,fit,nums,inner(ctx,fit,nums,'final'))
        check_choices(choice,json.loads((run.OUT/'final_choices.json').read_text()))
        saved,_=load_partition(ctx,fit,test,'final_test')
        # Independently rebuild features from the inherited fold-local recipe.
        oldctx=run.previous.context()
        d,_,labels,g,title,retrieval,content,_=oldctx
        recipe=joblib.load(run.OUT/'final_recipe.joblib')
        rebuilt=run.previous.feat.rebuild_prediction(d,labels,g,title,retrieval,content,fit,test,recipe['feature_recipe'])
        cached=joblib.load(run.PRIOR/'final_test_features.joblib')
        pd.testing.assert_frame_equal(rebuilt,cached['b'],check_dtype=False,atol=1e-11,rtol=1e-11)
        raw=np.zeros((596,4))
        for j,k in enumerate([2,3,4,5]):
            model=CatBoostClassifier();model.load_model(str(run.OUT/f'ordinal_ge{k}.cbm'))
            assert model.get_all_params()['depth']==4 and model.tree_count_==300
            assert model.get_all_params()['random_seed']==run.SEED
            raw[:,j]=model.predict_proba(rebuilt[recipe['columns']])[:,1]
        np.testing.assert_allclose(raw,saved['raw'],atol=1e-12)
        reloaded=view(raw,True);p=priorities(reloaded,choice['ordinal']['threshold'])
        submission=pd.read_csv(run.OUT/'submission_ordinal.csv',dtype={'id':str})
        byid=dict(zip(data.id.iloc[test],p));expected=np.array([byid[i] for i in submission.id])
        np.testing.assert_allclose(submission.priority,expected,atol=1e-11)
        order=lambda a:sorted(np.flatnonzero(a>0),key=lambda i:(-a[i],submission.id.iloc[i]))
        assert order(submission.priority.to_numpy())==order(expected)
        candidate=pd.read_csv(run.OUT/'submission_candidate.csv',dtype={'id':str})
        for f in [submission,candidate]:
            assert list(f.columns)==['id','priority'] and len(f)==596 and f.id.is_unique
            assert f.id.tolist()==ctx['template'].id.tolist() and np.isfinite(f.priority).all() and (f.priority>=0).all()
        expected_path=run.PRIOR/'submission_candidate.csv' if choice['selection_procedure']['family']=='multiclass' else run.OUT/'submission_ordinal.csv'
        assert run.digest(expected_path)==run.digest(run.OUT/'submission_candidate.csv')
        assert run.digest(run.PRIOR/'submission_candidate.csv')=='8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf'
        result=dict(status='PASSED',original_labels_preserved=True,inherited_group_crossfits_checked=lineages,
            binary_fits_audited=120,inner_choices_recomputed=18,outer_metric_rows_recomputed=counts,tail_metric_rows_recomputed=tailcount,
            pav_independently_recomputed=True,gain_preserved_by_projection=True,incumbent_oof_exact=True,
            final_features_rebuilt=True,four_models_reloaded=True,final_predictions_identical=True,full_ranking_identical=True,
            n_test=596,ordinal_selected=int(submission.priority.gt(0).sum()),candidate_selected=int(candidate.priority.gt(0).sum()),
            incumbent_csv_unchanged=True,uploaded=False)
        (run.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
