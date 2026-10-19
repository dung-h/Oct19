#!/usr/bin/env python3
"""Verify nested decisions and summarize input effects without retraining."""
import json

import numpy as np
import pandas as pd

import run_ablation as ab


def main():
    data,_=ab.load_clean_data();y=data.Label.iloc[:2494].astype(int).to_numpy()
    oof=pd.read_csv(ab.OUT/'oof_predictions.csv',dtype={'id':str})
    m=pd.read_csv(ab.OUT/'fold_metrics.csv');splits=pd.read_csv(ab.V2/'nested_splits.csv',dtype={'id':str})
    assert len(oof)==2494 and oof.id.is_unique and np.array_equal(oof.id,data.id.iloc[:2494])
    assert np.array_equal(oof.Label,y)
    groups=ab.baseline.grouping(data.iloc[:2494].reset_index(drop=True));lookup=dict(zip(oof.id,oof.index))
    configs=ab.configurations();count=0
    for fold in range(5):
        rows=splits[splits.outer_fold.eq(fold)];fr=rows[rows.partition.eq('outer_train')];vr=rows[rows.partition.eq('outer_validation')]
        fit=np.array([lookup[x] for x in fr.id]);val=np.array([lookup[x] for x in vr.id]);nums=fr.inner_fold.to_numpy()
        assert set(groups[fit]).isdisjoint(groups[val])
        for k in range(4):assert set(groups[fit[nums!=k]]).isdisjoint(groups[fit[nums==k]])
        cached=np.load(ab.OUT/f'outer_{fold}.npz')
        inner={c['key']:cached['inner_'+c['key']] for c in configs}
        choices,_=ab.select(data,y,fit,nums,inner,configs)
        for row in m[m.outer_fold.eq(fold)].to_dict('records'):
            selected=choices[row['family']]
            assert selected['key']==row['config'] and abs(selected['threshold']-row['threshold'])<1e-12
            p=ab.priority(cached['outer_'+row['config']],row['threshold'])
            np.testing.assert_allclose(oof.loc[val,row['family']+'_priority'],p,atol=1e-12)
            result=ab.evaluate(y[val],p,data.id.iloc[val])
            for key,value in result.items():assert abs(value-row[key])<1e-9
            count+=1
    # Anchor must reproduce the previous controlled experiment before interpreting an input effect.
    old=pd.read_csv(ab.HERE.parent/'diagnose_bottleneck/outputs/oof_predictions.csv',dtype={'id':str})
    np.testing.assert_allclose(oof.specter_joint_rating,old.regression_ridge_rating,atol=1e-10)
    np.testing.assert_allclose(oof.specter_joint_priority,old.regression_ridge_priority,atol=1e-10)
    deltas=[];changes=[]
    for family in sorted(m.family.unique()):
        a=m[m.family.eq('specter_joint')].set_index('outer_fold');b=m[m.family.eq(family)].set_index('outer_fold')
        h=b.score-a.score
        deltas.append(dict(family=family,delta_H=h.mean(),folds_improved=int((h>0).sum()),
            delta_AUC=(b.read_auc-a.read_auc).mean(),delta_MAE=(b.validation_mae-a.validation_mae).mean(),
            **{f'fold_{k}_delta':v for k,v in h.items()}))
        baseline_error=oof.specter_joint_priority.gt(0).ne(oof.Label.gt(1))
        new_error=oof[family+'_priority'].gt(0).ne(oof.Label.gt(1))
        changes.append(dict(family=family,fixed=int((baseline_error&~new_error).sum()),
            introduced=int((~baseline_error&new_error).sum()),persisting=int((baseline_error&new_error).sum())))
    pd.DataFrame(deltas).to_csv(ab.OUT/'paired_deltas.csv',index=False)
    pd.DataFrame(changes).to_csv(ab.OUT/'error_changes.csv',index=False)
    # Fixed-alpha control separates the input effect from regularization selection.
    searches=pd.read_csv(ab.OUT/'inner_search.csv');fixed_rows=[]
    fixed_configs=[c for c in configs if c['representation']=='specter' and c['mode'] in ['title','joint'] and c['alpha']==.1]
    for fold in range(5):
        cache=np.load(ab.OUT/f'outer_{fold}.npz');val=cache['val']
        for config in fixed_configs:
            row=searches[searches.outer_fold.eq(fold)&searches.key.eq(config['key'])].iloc[0]
            p=ab.priority(cache['outer_'+config['key']],row.threshold)
            fixed_rows.append(dict(outer_fold=fold,mode=config['mode'],**ab.evaluate(y[val],p,data.id.iloc[val])))
    pd.DataFrame(fixed_rows).to_csv(ab.OUT/'fixed_alpha_0_1_input_ablation.csv',index=False)
    result=dict(n=2494,outer_metrics_recomputed=count,inner_choices_recomputed=True,
        grouped_splits_disjoint=True,original_labels_preserved=True,
        joint_baseline_reproduced_at_1e_10=True)
    (ab.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print(pd.DataFrame(deltas).to_string(index=False))


if __name__=='__main__':main()
