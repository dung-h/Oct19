#!/usr/bin/env python3
"""Run the pre-registered binary read-gate repair experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

import author_features as authors
import gate_lib as g


def model():
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=.05, l2_leaf_reg=8,
        loss_function='Logloss', random_seed=g.SEED, thread_count=4, bootstrap_type='Bayesian',
        bagging_temperature=1, leaf_estimation_iterations=1, verbose=False, allow_writing_files=False)


def protocol(ctx):
    g.OUT.mkdir(parents=True, exist_ok=True)
    source = [g.HERE/'PLAN.md', g.HERE/'experiment_spec.json', g.HERE/'author_features.py',
              g.HERE/'gate_lib.py', g.HERE/'run_gate.py', g.HERE/'verify_gate.py',
              g.HERE/'write_report.py', g.HERE/'tests'/'test_gate.py']
    caches = list(g.INC.glob('*_features.joblib'))
    used = [p for p in caches if p.stem.replace('_features','').startswith(('outer','final'))]
    record = dict(status='locked_before_new_results', source_sha256={str(p.relative_to(g.ROOT.parent)):g.digest(p) for p in source},
        cache_sha256={str(p.relative_to(g.ROOT.parent)):g.digest(p) for p in used},
        metadata_sha256=g.digest(g.META), raw_zip_sha256=g.digest(g.ROOT.parent/'assignment-data-mining-261.zip'),
        input_sha256=g.title_run.ab.baseline.input_digest(ctx['data']),
        labels_sha256=hashlib.sha256(ctx['y'][:2494].astype(np.int64).tobytes()).hexdigest(),
        groups_sha256=hashlib.sha256(ctx['groups'].astype(np.int64).tobytes()).hexdigest(),
        config=json.loads((g.HERE/'experiment_spec.json').read_text()), source_incumbent_sha256=g.digest(g.INC/'submission_candidate.csv'))
    target=g.OUT/'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == record, 'Locked source/cache changed; use a new experiment directory.'
    else:
        target.write_text(json.dumps(record,indent=2)+'\n')
    return record


def load_ledger(ctx):
    audit = pd.read_csv(g.META, dtype={'id':str})
    ledger=authors.source_ledger(ctx['data'], audit)
    serial=ledger.drop(columns='author_tokens').copy(); serial['author_tokens']=ledger.author_tokens.map(json.dumps)
    serial.to_csv(g.OUT/'metadata_ledger.csv',index=False)
    rows=[]
    for i,row in ledger.iterrows():
        for token in row.author_tokens:
            rows.append(dict(split=row['split'],id=row['id'],author=token,source_verified=row.author_source_verified))
    pd.DataFrame(rows,columns=['split','id','author','source_verified']).to_csv(g.OUT/'author_tokens.csv',index=False)
    return ledger


def audit(ctx, ledger):
    rows=[]
    for fold,fit,pred,nums in g.outer_members(ctx):
        assert not(set(ctx['groups'][fit]) & set(ctx['groups'][pred]))
        for k in range(4):
            mask=nums==k; p,a,b=g.load_pack(f'outer{fold}_inner{k}',fit[~mask],fit[mask])
            assert not(set(ctx['groups'][fit[~mask]]) & set(ctx['groups'][fit[mask]]))
            rows.append(dict(partition=f'outer{fold}_inner{k}',n_fit=len(p['fit']),n_pred=len(p['pred']),features=a.shape[1],finite=True))
        p,a,b=g.load_pack(f'outer{fold}_validation',fit,pred)
        rows.append(dict(partition=f'outer{fold}_validation',n_fit=len(fit),n_pred=len(pred),features=a.shape[1],finite=True))
    fit,pred,nums=g.final_partition(ctx)
    for k in range(4):
        mask=nums==k;p,a,b=g.load_pack(f'final_inner{k}',fit[~mask],fit[mask])
        rows.append(dict(partition=f'final_inner{k}',n_fit=len(p['fit']),n_pred=len(p['pred']),features=a.shape[1],finite=True))
    p,a,b=g.load_pack('final_test',fit,pred)
    rows.append(dict(partition='final_test',n_fit=len(fit),n_pred=len(pred),features=a.shape[1],finite=True))
    pd.DataFrame(rows).to_csv(g.OUT/'feature_catalog.csv',index=False)
    audit=dict(status='PASSED', partitions=len(rows), base_features=277, rows=len(ledger),
               missing_authors=int(ledger.author_missing.sum()), verified_authors=int(ledger.author_source_verified.sum()),
               uncertain_author_format=int(ledger.author_format_uncertain.sum()))
    (g.OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')


def pair(ctx, ledger, name, fit, pred):
    pack,base_a,base_b=g.load_pack(name,fit,pred)
    author_a,author_b=authors.pair_features(ledger,ctx['y'],ctx['groups'],fit,pred,pack['lineage'])
    assert list(author_a.columns)==authors.AUTHOR_COLUMNS and np.isfinite(author_a.to_numpy()).all()
    return base_a,base_b,author_a,author_b,pack


def fit_both(base_a, base_b, author_a, author_b, target, save_dir=None):
    base_cols=base_a.columns.tolist(); author_cols=base_cols+authors.AUTHOR_COLUMNS
    a_author=pd.concat([base_a.reset_index(drop=True),author_a.reset_index(drop=True)],axis=1)
    b_author=pd.concat([base_b.reset_index(drop=True),author_b.reset_index(drop=True)],axis=1)
    result={}; models={}
    for key,a,b,cols in [('binary',base_a,base_b,base_cols),('author',a_author,b_author,author_cols)]:
        clf=model();clf.fit(a[cols],target,cat_features=['venue']); result[key]=clf.predict_proba(b[cols])[:,1]; models[key]=clf
        if save_dir is not None: clf.save_model(str(Path(save_dir)/f'{key}_model.cbm'))
    return result,models,base_cols,author_cols


def branch_probability(raw, calibrators, venue):
    return dict(raw=raw, global_=g.apply_calibrator(calibrators['global'],raw,venue), venue_=g.apply_calibrator(calibrators['venue'],raw,venue))


def title_epoch(prefix):
    choice=json.loads((g.TITLE/(f'{prefix}_choices.json' if prefix.startswith('outer') else 'final_choices.json')).read_text())
    choice=choice['primary'] if prefix.startswith('outer') else choice
    return int(choice['adapted']['epoch'])


def save_outer_rows(ctx, ledger, fold, pred, incumbent, title, values, probabilities):
    ids=ctx['data'].id.iloc[pred].to_numpy(str); y=ctx['y'][pred].astype(int)
    base=pd.DataFrame(dict(id=ids,outer_fold=fold,Label=y,venue=ctx['data'].venue.iloc[pred].to_numpy(str),
                           author_missing=ledger.author_missing.iloc[pred].to_numpy(int),
                           author_verified=ledger.author_source_verified.iloc[pred].to_numpy(int)))
    frames=[]; metrics=[]
    controls={'incumbent_deployed':1-incumbent[:,0], 'incumbent_at_050':1-incumbent[:,0]}
    cutoffs={'incumbent_deployed':.55,'incumbent_at_050':.50}
    for branch,p in {**controls, **probabilities}.items():
        selected=p>cutoffs.get(branch,.5); pri=g.priority(selected,incumbent,title)
        row=base.copy();row['branch']=branch;row['read_probability']=p;row['selected']=selected;row['priority']=pri
        row['outcome']=np.select([selected&(y>1),selected&(y==1),(~selected)&(y>1)],['TP','FP','FN'],default='TN')
        frames.append(row)
        metrics.append(dict(outer_fold=fold,branch=branch,**g.selection_metrics(y,selected),**g.official(y,pri,ids)))
    return pd.concat(frames,ignore_index=True),metrics


def run_outer(ctx, ledger):
    all_rows=[]; all_metrics=[]
    for fold,fit,pred,nums in g.outer_members(ctx):
        inner={k:np.full(len(fit),np.nan) for k in ['binary','author']}
        for k in range(4):
            mask=nums==k; name=f'outer{fold}_inner{k}'
            ba,bb,aa,ab,_=pair(ctx,ledger,name,fit[~mask],fit[mask])
            values,_,_,_=fit_both(ba,bb,aa,ab,(ctx['y'][fit[~mask]]>1).astype(int))
            for key in inner: inner[key][mask]=values[key]
        assert all(np.isfinite(x).all() for x in inner.values())
        np.savez_compressed(g.OUT/f'outer{fold}_inner_base_probabilities.npz',fit=fit,prob_binary=inner['binary'],prob_author=inner['author'])
        target=(ctx['y'][fit]>1).astype(int);venue_fit=ctx['data'].venue.iloc[fit].fillna('unknown').astype(str).to_numpy()
        calibrators={family:{kind:g.fit_calibrator(inner[family],venue_fit,target,kind) for kind in ['global','venue']} for family in inner}
        joblib.dump(calibrators,g.OUT/f'outer{fold}_calibrators.joblib',compress=3)
        ba,bb,aa,ab,_=pair(ctx,ledger,f'outer{fold}_validation',fit,pred)
        raw,_,_,_=fit_both(ba,bb,aa,ab,target)
        venue=ctx['data'].venue.iloc[pred].fillna('unknown').astype(str).to_numpy()
        probs={}
        for family in ['binary','author']:
            result=branch_probability(raw[family],calibrators[family],venue)
            probs[f'{family}_raw']=result['raw'];probs[f'{family}_global']=result['global_'];probs[f'{family}_venue']=result['venue_']
        np.savez_compressed(g.OUT/f'outer{fold}_validation_probabilities.npz',fit=fit,pred=pred,raw_binary=raw['binary'],raw_author=raw['author'],**probs)
        inc=g.load_incumbent_outer(fold,fit,pred); epoch=title_epoch(f'outer{fold}'); title=g.title_probability(f'outer{fold}',fit,pred,epoch)
        rows,metrics=save_outer_rows(ctx,ledger,fold,pred,inc,title,raw,probs);all_rows.append(rows);all_metrics.extend(metrics)
        print(f'outer {fold}: '+', '.join(f"{x['branch']} E={x['total_error']}" for x in metrics),flush=True)
    frame=pd.concat(all_rows,ignore_index=True);frame.to_csv(g.OUT/'outcomes_by_id.csv',index=False)
    pd.DataFrame(all_metrics).to_csv(g.OUT/'fold_metrics.csv',index=False)
    summarize_outer(frame)


def summarize_outer(frame):
    summary=[]; labels=[]; transitions=[]
    for branch,part in frame.groupby('branch',sort=False):
        result=g.selection_metrics(part.Label,part.selected);priority=part.priority.to_numpy();result.update(g.official(part.Label.to_numpy(),priority,part.id.to_numpy()))
        summary.append(dict(branch=branch,**result))
        for label,section in part.groupby('Label'):
            labels.append(dict(branch=branch,Label=int(label),total=len(section),accepted=int(section.selected.sum()),dropped=int((~section.selected).sum())))
        if branch != 'incumbent_deployed':
            base=frame[frame.branch.eq('incumbent_deployed')].set_index('id').loc[part.id]
            now=part.set_index('id').loc[base.index]; pos=base.Label.to_numpy()>1; old=base.selected.to_numpy(bool);new=now.selected.to_numpy(bool)
            values=dict(branch=branch,FN_rescued=int((~old&new&pos).sum()),FP_removed=int((old&~new&~pos).sum()),new_FP=int((~old&new&~pos).sum()),lost_TP=int((old&~new&pos).sum()))
            values['delta_error']=values['new_FP']+values['lost_TP']-values['FN_rescued']-values['FP_removed']
            assert values['delta_error']==result['total_error']-summary[0]['total_error']
            transitions.append(values)
    pd.DataFrame(summary).to_csv(g.OUT/'summary.csv',index=False)
    pd.DataFrame(labels).to_csv(g.OUT/'counts_by_label.csv',index=False)
    pd.DataFrame(transitions).to_csv(g.OUT/'transition_summary.csv',index=False)
    reliability_rows=[]
    for branch,part in frame.groupby('branch',sort=False):
        bins,brier,ll=g.reliability(part.Label,part.read_probability)
        for row in bins:reliability_rows.append(dict(branch=branch,brier=brier,logloss=ll,**row))
    pd.DataFrame(reliability_rows).to_csv(g.OUT/'reliability.csv',index=False)


def run_final(ctx, ledger):
    fit,pred,nums=g.final_partition(ctx); inner={k:np.full(len(fit),np.nan) for k in ['binary','author']}
    for k in range(4):
        mask=nums==k;ba,bb,aa,ab,_=pair(ctx,ledger,f'final_inner{k}',fit[~mask],fit[mask])
        values,_,_,_=fit_both(ba,bb,aa,ab,(ctx['y'][fit[~mask]]>1).astype(int))
        for key in inner:inner[key][mask]=values[key]
    target=(ctx['y'][fit]>1).astype(int);venue_fit=ctx['data'].venue.iloc[fit].fillna('unknown').astype(str).to_numpy()
    calibrators={family:{kind:g.fit_calibrator(inner[family],venue_fit,target,kind) for kind in ['global','venue']} for family in inner}
    np.savez_compressed(g.OUT/'final_inner_base_probabilities.npz',fit=fit,prob_binary=inner['binary'],prob_author=inner['author'])
    joblib.dump(calibrators,g.OUT/'final_calibrators.joblib',compress=3)
    ba,bb,aa,ab,_=pair(ctx,ledger,'final_test',fit,pred)
    (g.OUT/'final_models').mkdir(exist_ok=True)
    raw,_,base_cols,author_cols=fit_both(ba,bb,aa,ab,target,save_dir=g.OUT/'final_models')
    joblib.dump(dict(base_columns=base_cols,author_columns=author_cols),g.OUT/'final_models'/'columns.joblib',compress=3)
    venue=ctx['data'].venue.iloc[pred].fillna('unknown').astype(str).to_numpy(); probs={}
    for family in ['binary','author']:
        result=branch_probability(raw[family],calibrators[family],venue)
        probs[f'{family}_raw']=result['raw'];probs[f'{family}_global']=result['global_'];probs[f'{family}_venue']=result['venue_']
    inc=g.load_incumbent_final(ctx,fit,pred);epoch=title_epoch('final');title=g.title_probability('final',fit,pred,epoch)
    ids=ctx['data'].id.iloc[pred].to_numpy(str);records=[]
    values={'incumbent_deployed':1-inc[:,0],'incumbent_at_050':1-inc[:,0],**probs};cutoffs={'incumbent_deployed':.55,'incumbent_at_050':.5}
    for branch,p in values.items():
        selected=p>cutoffs.get(branch,.5);priority=g.priority(selected,inc,title)
        detail=pd.DataFrame(dict(id=ids,read_probability=p,selected=selected,priority=priority))
        detail.to_csv(g.OUT/f'test_predictions_{branch}.csv',index=False)
        submission=ctx['template'][['id']].merge(detail[['id','priority']],on='id',validate='one_to_one')
        submission.to_csv(g.OUT/f'submission_{branch}.csv',index=False)
        records.append(dict(branch=branch,total=len(detail),accepted=int(selected.sum()),dropped=int((~selected).sum())))
    pd.DataFrame(records).to_csv(g.OUT/'test_counts.csv',index=False)
    recipe=dict(primary='author_venue',cutoff=.5,comparison='strict greater than',ranking='0.5 incumbent gain + 0.5 adapted-title gain',title_epoch=epoch,uploaded=False,protocol_sha256=g.digest(g.OUT/'protocol.json'))
    (g.OUT/'final_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','outer','final','all'],nargs='?',default='all');args=parser.parse_args()
    with threadpool_limits(4):
        ctx=g.context();protocol(ctx);ledger=load_ledger(ctx)
        if args.stage in ['audit','all']:audit(ctx,ledger)
        if args.stage in ['outer','all']:run_outer(ctx,ledger)
        if args.stage in ['final','all']:run_final(ctx,ledger)


if __name__=='__main__': main()
