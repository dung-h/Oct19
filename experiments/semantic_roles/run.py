#!/usr/bin/env python3
"""Fixed semantic-role ablation with grouped inner operating constraints."""
import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits

import experiment as e
import features as f
from roles import OUT,source,digest


def partition(ctx,name,fit,pred):
    target=OUT/(name+'_probabilities.npz');lock=digest(OUT/'protocol.json')
    inc=e.incumbent(name,fit,pred,ctx)
    keep=name.endswith('_validation') or name=='final_test'
    recipe_path=OUT/(name+'_models.joblib')
    if target.exists():
        z=np.load(target);np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['pred'],pred)
        assert str(z['protocol_sha256'])==lock
        if keep:assert recipe_path.exists()
        return {k:z[k].copy() for k in ['text','roles']}
    started=time.monotonic();spaces=f.fit_spaces(ctx['texts'],fit)
    a=f.transform(ctx['texts'],ctx['embedding'],fit,spaces)
    b=f.transform(ctx['texts'],ctx['embedding'],pred,spaces)
    result={};models={};logs=[]
    for arm in ['text','roles']:
        clf=LogisticRegression(C=4,penalty='l2',solver='liblinear',max_iter=2000,tol=1e-5,
            random_state=20261018,class_weight=None)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always');clf.fit(a[arm],(ctx['y'][fit]>1).astype(int))
        convergence=[str(w.message) for w in recorded if issubclass(w.category,ConvergenceWarning)]
        assert not convergence,convergence
        result[arm]=clf.predict_proba(b[arm])[:,1];models[arm]=clf
        logs.append(dict(arm=arm,n_fit=len(fit),n_pred=len(pred),features=a[arm].shape[1],iterations=clf.n_iter_.tolist()))
    if keep:joblib.dump(dict(spaces=spaces,models=models,fit=fit,pred=pred,protocol_sha256=lock),recipe_path,compress=3)
    np.savez_compressed(target,fit=fit,pred=pred,text=result['text'],roles=result['roles'],protocol_sha256=lock)
    (OUT/(name+'_training.json')).write_text(json.dumps(dict(partition=name,seconds=time.monotonic()-started,models=logs),indent=2)+'\n')
    print(name,'done',round(time.monotonic()-started,1),'seconds',flush=True)
    return result


def inner(ctx,fit,nums,prefix):
    values={k:np.full(len(fit),np.nan) for k in ['text','roles','incumbent']}
    for k in range(4):
        mask=nums==k;name=f'{prefix}_inner{k}';a,b=fit[~mask],fit[mask]
        assert set(ctx['groups'][a]).isdisjoint(ctx['groups'][b])
        pred=partition(ctx,name,a,b)
        for arm in ['text','roles']:values[arm][mask]=pred[arm]
        values['incumbent'][mask]=1-e.incumbent(name,a,b,ctx)[:,0]
    assert all(np.isfinite(x).all() for x in values.values())
    branches=e.branch_values(values['text'],values['roles'],values['incumbent'])
    choices={branch:e.choose(ctx['y'][fit],p,values['incumbent']) for branch,p in branches.items()}
    (OUT/(prefix+'_choices.json')).write_text(json.dumps(choices,indent=2)+'\n')
    return choices


def audit(ctx):
    counts=[]
    for name,fit,pred in e.jobs(ctx):
        # Inner/outer groups must be disjoint. Final test duplicates are known;
        # the new text learners use no label-derived history features.
        if name!='final_test':assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][pred])
        e.incumbent(name,fit,pred,ctx)
        counts.append(dict(partition=name,n_fit=len(fit),n_pred=len(pred)))
    pd.DataFrame(counts).to_csv(OUT/'partition_audit.csv',index=False)
    assert len(counts)==30


def run_outer(ctx):
    all_rows=[];metrics=[]
    for fold,fit,pred,nums in source.outer_members(ctx):
        choices=inner(ctx,fit,nums,f'outer{fold}')
        raw=partition(ctx,f'outer{fold}_validation',fit,pred)
        inc=e.incumbent(f'outer{fold}_validation',fit,pred,ctx)
        values=e.branch_values(raw['text'],raw['roles'],1-inc[:,0])
        values={'incumbent':1-inc[:,0],**values}
        ids=ctx['data'].id.iloc[pred].to_numpy(str);y=ctx['y'][pred].astype(int)
        for branch,p in values.items():
            cutoff=.55 if branch=='incumbent' else choices[branch]['threshold']
            selected=p>cutoff;priority=e.priorities(selected,inc)
            row=dict(outer_fold=fold,branch=branch,threshold=cutoff,**e.metrics(y,priority,ids),
                AUC=float(roc_auc_score(y>1,p)),Brier=float(np.mean((p-(y>1))**2)),
                logloss=float(log_loss(y>1,p,labels=[False,True])))
            metrics.append(row)
            outcome=np.select([selected&(y>1),selected&(y==1),(~selected)&(y>1)],['TP','FP','FN'],default='TN')
            all_rows.append(pd.DataFrame(dict(id=ids,outer_fold=fold,branch=branch,Label=y,
                title=ctx['data'].title.iloc[pred].to_numpy(),venue=ctx['data'].venue.iloc[pred].to_numpy(),
                read_probability=p,threshold=cutoff,selected=selected,priority=priority,outcome=outcome)))
        pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
        pd.concat(all_rows,ignore_index=True).to_csv(OUT/'outcomes_by_id.csv',index=False)
        print('OUTER',fold,[(r['branch'],r['FP'],r['FN'],r['missed_high']) for r in metrics if r['outer_fold']==fold],flush=True)


def run_final(ctx):
    fit,pred,nums=source.final_partition(ctx);choices=inner(ctx,fit,nums,'final')
    raw=partition(ctx,'final_test',fit,pred);inc=e.incumbent('final_test',fit,pred,ctx)
    values=e.branch_values(raw['text'],raw['roles'],1-inc[:,0]);rows=[]
    ids=ctx['data'].id.iloc[pred].to_numpy(str);old=1-inc[:,0]>.55
    for branch,p in values.items():
        selected=p>choices[branch]['threshold'];priority=e.priorities(selected,inc)
        details=pd.DataFrame(dict(id=ids,read_probability=p,selected=selected,priority=priority))
        details.to_csv(OUT/('test_'+branch+'.csv'),index=False)
        submission=ctx['template'][['id']].merge(details[['id','priority']],on='id',validate='one_to_one')
        assert len(submission)==596 and submission.id.is_unique
        submission.to_csv(OUT/('submission_'+branch+'.csv'),index=False)
        rows.append(dict(branch=branch,accepted=int(selected.sum()),dropped=int((~selected).sum()),
            newly_selected=int((~old&selected).sum()),newly_dropped=int((old&~selected).sum())))
    pd.DataFrame(rows).to_csv(OUT/'test_counts.csv',index=False)
    (OUT/'final_recipe.json').write_text(json.dumps(dict(primary=e.PRIMARY,choices=choices,
        protocol_sha256=digest(OUT/'protocol.json'),uploaded=False,incumbent_preserved=True),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','outer','final','all'],nargs='?',default='all');stage=parser.parse_args().stage
    with threadpool_limits(4):
        ctx=e.context();e.protocol(ctx);audit(ctx)
        if stage in ['outer','all']:run_outer(ctx)
        if stage in ['final','all']:run_final(ctx)


if __name__=='__main__':main()
