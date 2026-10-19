#!/usr/bin/env python3
"""Post-hoc accounting and feature audit, never fed into training/selection."""
import json
from pathlib import Path

from catboost import CatBoostClassifier,Pool
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_controls as diag
base=diag.base
om=base.om
OUT=diag.OUT/'mechanisms'


def main():
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(4):
        ctx=base.context();base.protocol(ctx);data=ctx['data'];y=ctx['y']
        old=pd.read_csv(base.OUT/'fold_metrics.csv')
        oof=pd.read_csv(base.OUT/'oof_predictions.csv',dtype={'id':str})
        crossed=[];decomp=[];checks=[];featureaudit=[];drift=[];lossrows=[];gainrows=[]
        for fold,fit,val,nums in base.previous.outer_members(data):
            pack=base.feature_pack(ctx,f'outer{fold}_validation',fit,val)
            z=np.load(base.OUT/f'outer{fold}_validation.npz')
            inc,_=base.incumbent(fit,val,f'outer{fold}_validation')
            ordinal=om.ordinal_view(z['raw']);views=dict(multiclass=inc,ordinal=ordinal)
            choices=json.loads((base.OUT/f'outer{fold}_choices.json').read_text())
            gates={f:views[f]['read']>choices[f]['threshold'] for f in views}
            h={}
            for gate in views:
                for ranker in views:
                    p=np.where(gates[gate],10*views[ranker]['gain'],0.)
                    result=base.ab.evaluate(y[val],p,data.id.iloc[val]);h[(gate,ranker)]=result['score']
                    crossed.append(dict(fold=fold,gate=gate,ranker=ranker,**result))
            gate_effect=.5*((h['ordinal','multiclass']-h['multiclass','multiclass'])+(h['ordinal','ordinal']-h['multiclass','ordinal']))
            rank_effect=.5*((h['multiclass','ordinal']-h['multiclass','multiclass'])+(h['ordinal','ordinal']-h['ordinal','multiclass']))
            assert abs(gate_effect+rank_effect-(h['ordinal','ordinal']-h['multiclass','multiclass']))<1e-12
            decomp.append(dict(fold=fold,delta_H=h['ordinal','ordinal']-h['multiclass','multiclass'],gate_effect=gate_effect,rank_effect=rank_effect))
            threshold=choices['ordinal']['threshold']
            raw_gate=z['raw'][:,0]>threshold
            no_projection=base.ab.evaluate(y[val],np.where(raw_gate,10*ordinal['gain'],0.),data.id.iloc[val])['score']
            same_cutoff=base.ab.evaluate(y[val],om.priorities(ordinal,choices['multiclass']['threshold']),data.id.iloc[val])['score']
            # Oracle uses true validation labels; diagnosis only, never selected policy.
            oracle=max(base.ab.evaluate(y[val],om.priorities(ordinal,t),data.id.iloc[val])['score'] for t in np.r_[-1,np.unique(ordinal['read'])])
            checks.append(dict(fold=fold,ordinal_H=h['ordinal','ordinal'],without_projection_read_gate_H=no_projection,
                projection_changed_selected=int(np.sum(raw_gate!=gates['ordinal'])),same_multiclass_cutoff_H=same_cutoff,
                oracle_cutoff_H=oracle))
            for f,v in views.items():
                trueprob=v['probability'][np.arange(len(val)),y[val].astype(int)-1]
                individual=-np.log(np.clip(trueprob,np.finfo(float).eps,1))
                for offset,index in enumerate(val):
                    lossrows.append(dict(fold=fold,id=data.id.iloc[index],Label=int(y[index]),family=f,
                        true_probability=float(trueprob[offset]),logloss=float(individual[offset]),zero_mass=bool(trueprob[offset]<=0)))
                for label in [1,2,3,4,5]:
                    mask=y[val]==label
                    gainrows.append(dict(fold=fold,family=f,Label=label,n=int(mask.sum()),
                        absolute_error_sum=float(np.abs(v['gain'][mask]-(label-1)/4).sum()),
                        bias_sum=float((v['gain'][mask]-(label-1)/4).sum())))
            a=pack['a'];b=pack['b'];numeric=[c for c in ctx['columns'] if c!='venue']
            featureaudit.append(dict(fold=fold,n_features=len(ctx['columns']),
                numeric_nonfinite_train=int((~np.isfinite(a[numeric].to_numpy(dtype=float))).sum()),
                numeric_nonfinite_val=int((~np.isfinite(b[numeric].to_numpy(dtype=float))).sum()),
                constant_train=int(sum(a[c].nunique()==1 for c in ctx['columns'])),
                title_svd_variance=float(pack['recipe']['svd'].explained_variance_ratio_.sum()),
                empty_title_val=int(data.title.iloc[val].str.len().eq(0).sum()),
                usable_abstract_val=int(data.abstract.iloc[val].str.len().gt(0).sum())))
            for col in numeric:
                av=a[col].to_numpy(dtype=float);bv=b[col].to_numpy(dtype=float)
                if any(s in col for s in ['references','support']):
                    drift.append(dict(fold=fold,column=col,train_min=float(av.min()),train_max=float(av.max()),
                        val_min=float(bv.min()),val_max=float(bv.max()),
                        train_mean=float(av.mean()),val_mean=float(bv.mean()),
                        val_outside_train_range=float(((bv<av.min())|(bv>av.max())).mean())))
        for name,rows in [('crossed_gate_rank',crossed),('H_decomposition',decomp),('postprocessing_controls',checks),
                          ('feature_audit',featureaudit),('history_size_drift',drift),('individual_logloss',lossrows),('gain_error_by_label',gainrows)]:
            pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
        oldm=CatBoostClassifier();oldm.load_model(str(base.PRIOR/'final_classifier.cbm'))
        ordm=CatBoostClassifier();ordm.load_model(str(base.OUT/'ordinal_ge2.cbm'))
        p0=oldm.get_all_params();p1=ordm.get_all_params()
        differences={k:dict(multiclass=p0.get(k),ordinal=p1.get(k)) for k in sorted(set(p0)|set(p1)) if p0.get(k)!=p1.get(k)}
        (OUT/'default_parameter_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
        importance=[]
        final=base.feature_pack(ctx,'final_test',np.arange(2494),np.arange(2494,len(data)))
        pool=Pool(final['a'][ctx['columns']],cat_features=['venue'])
        for family,model in [('multiclass',oldm),('ordinal_read',ordm)]:
            values=model.get_feature_importance(data=pool,type='PredictionValuesChange',thread_count=4)
            for col,value in zip(ctx['columns'],values):
                group='neighbor' if col.startswith('neighbor_') else 'content' if col.startswith('content_') else 'base'
                importance.append(dict(family=family,feature=col,group=group,importance=float(value)))
        pd.DataFrame(importance).to_csv(OUT/'training_feature_importance.csv',index=False)
        loss=pd.DataFrame(lossrows);pav_zero=loss[loss.family.eq('ordinal')&loss.zero_mass]
        summary=dict(
            mean_H_decomposition=pd.DataFrame(decomp).drop(columns='fold').mean().to_dict(),
            crossed_H=pd.DataFrame(crossed).groupby(['gate','ranker']).score.mean().to_dict(),
            postprocessing=pd.DataFrame(checks).drop(columns='fold').mean().to_dict(),
            pav_changed_selection_count=int(pd.DataFrame(checks).projection_changed_selected.sum()),
            ordinal_zero_true_mass_rows=len(pav_zero),zero_true_mass_labels=pav_zero.Label.value_counts().to_dict(),
            mean_title_svd_TFIDF_variance=float(pd.DataFrame(featureaudit).title_svd_variance.mean()),
            numeric_nonfinite_count=int(pd.DataFrame(featureaudit)[['numeric_nonfinite_train','numeric_nonfinite_val']].to_numpy().sum()),
            features=dict(base=35,content=92,neighbors=150),
            limitations=['Crossed-gate Shapley values account for H difference, not root-cause attribution.',
                         'Oracle cutoff uses validation truth; not deployable or expected improvement.',
                         'TFIDF variance retained is not the fraction of predictive information retained.',
                         'Training feature importance or feature drift alone does not prove causal harm.'])
        summary['crossed_H']={str(k):v for k,v in summary['crossed_H'].items()}
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
