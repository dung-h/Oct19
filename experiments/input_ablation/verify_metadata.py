#!/usr/bin/env python3
"""Recompute source coverage, stage choices, split isolation and metadata scores."""
import json
import numpy as np
import pandas as pd

import run_ablation as ab
import run_metadata as meta


def main():
    data,_=ab.load_clean_data();y=data.Label.iloc[:2494].astype(int).to_numpy()
    source=pd.read_csv(ab.OUT/'metadata_audit.csv',dtype={'id':str},keep_default_na=False)
    assert len(source)==3090 and np.array_equal(source.id,data.id) and np.array_equal(source.split,data.split)
    assert source.loc[source.authors_changed,'author_verified'].all()
    assert (source.loc[source.year_changed,'year_clean']-source.loc[source.year_changed,'year_original']).abs().le(1).all()
    m=pd.read_csv(meta.OUT/'fold_metrics.csv');oof=pd.read_csv(meta.OUT/'oof_predictions.csv',dtype={'id':str})
    previous=pd.read_csv(ab.OUT/'oof_predictions.csv',dtype={'id':str})
    assert np.array_equal(oof.id,previous.id) and np.array_equal(oof.Label,y)
    np.testing.assert_allclose(oof.input_policy_rating,previous.selection_procedure_rating,atol=1e-10)
    np.testing.assert_allclose(oof.input_policy_priority,previous.selection_procedure_priority,atol=1e-10)
    splits=pd.read_csv(ab.V2/'nested_splits.csv',dtype={'id':str});lookup=dict(zip(oof.id,oof.index))
    groups=ab.baseline.grouping(data.iloc[:2494].reset_index(drop=True));configs=meta.configurations();count=0
    for fold in range(5):
        s=splits[splits.outer_fold.eq(fold)];fr=s[s.partition.eq('outer_train')];vr=s[s.partition.eq('outer_validation')]
        fit=np.array([lookup[x] for x in fr.id]);val=np.array([lookup[x] for x in vr.id]);nums=fr.inner_fold.to_numpy()
        assert set(groups[fit]).isdisjoint(groups[val])
        for k in range(4):assert set(groups[fit[nums!=k]]).isdisjoint(groups[fit[nums==k]])
        cache=np.load(meta.OUT/f'outer_{fold}.npz')
        choices,_=ab.select(data,y,fit,nums,{c['key']:cache['inner_'+c['key']] for c in configs},configs)
        for row in m[m.outer_fold.eq(fold)].to_dict('records'):
            choice=choices[row['family']]
            assert row['config']==choice['key'] and abs(row['threshold']-choice['threshold'])<1e-12
            p=ab.priority(cache['outer_'+row['config']],row['threshold'])
            np.testing.assert_allclose(oof.loc[val,row['family']+'_priority'],p,atol=1e-10)
            r=ab.evaluate(y[val],p,data.id.iloc[val])
            for key,value in r.items():assert abs(value-row[key])<1e-9
            count+=1
    original=m[m.family.eq('original_all')].set_index('outer_fold');clean=m[m.family.eq('clean_all')].set_index('outer_fold')
    delta=clean.score-original.score
    pd.DataFrame(dict(outer_fold=delta.index,delta_H=delta,delta_AUC=clean.AUC-original.AUC)).to_csv(meta.OUT/'clean_vs_original_metadata.csv',index=False)
    result=dict(outer_metrics_recomputed=count,all_inner_choices_recomputed=True,
        source_keys_original_labels_and_grouping_validated=True,stage1_reference_identical=True,
        clean_vs_original_delta_H=float(delta.mean()),clean_vs_original_folds_improved=int((delta>0).sum()))
    (meta.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
