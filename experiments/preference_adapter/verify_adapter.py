#!/usr/bin/env python3
"""Verify fit/calibration partitions, fixed-cutoff seed checks and every neural OOF score."""
import hashlib
import json

import numpy as np
import pandas as pd

import run_adapter as n


def main():
    data,_=n.ab.load_clean_data();y=data.Label.iloc[:2494].astype(int).to_numpy()
    groups=n.ab.baseline.grouping(data.iloc[:2494].reset_index(drop=True))
    split=pd.read_csv(n.ab.V2/'nested_splits.csv',dtype={'id':str});lookup=dict(zip(data.id.iloc[:2494],range(2494)))
    oof=pd.read_csv(n.OUT/'oof_predictions.csv',dtype={'id':str});m=pd.read_csv(n.OUT/'fold_metrics.csv')
    stability=pd.read_csv(n.OUT/'stability_metrics.csv');assert len(oof)==2494 and np.array_equal(oof.id,data.id.iloc[:2494])
    assert np.array_equal(oof.Label,y)
    count=0;training_logs=0
    for fold in range(5):
        s=split[split.outer_fold.eq(fold)];fr=s[s.partition.eq('outer_train')];vr=s[s.partition.eq('outer_validation')]
        fit=np.array([lookup[x] for x in fr.id]);val=np.array([lookup[x] for x in vr.id]);nums=fr.inner_fold.to_numpy()
        fi,ci=fit[nums!=0],fit[nums==0]
        assert set(groups[fit]).isdisjoint(groups[val]) and set(groups[fi]).isdisjoint(groups[ci])
        calibration={}
        for arm in n.ARMS:
            path=n.OUT/f'outer{fold}_inner0_{arm}.npz';cache=np.load(path)
            np.testing.assert_array_equal(cache['fit'],fi);np.testing.assert_array_equal(cache['val'],ci)
            calibration[arm]=cache['values']
            info=json.loads(path.with_suffix('.json').read_text())
            assert info['fit_ids_sha256']==hashlib.sha256('\n'.join(data.id.iloc[fi]).encode()).hexdigest()
            assert len(info['loss_by_epoch'])==3 and np.isfinite(info['loss_by_epoch']).all()
            training_logs+=1
        choices=n.choose(data,y,ci,np.zeros(len(ci),dtype=int),calibration)
        for arm,choice in choices.items():
            row=m[m.outer_fold.eq(fold)&m.family.eq(arm)].iloc[0]
            assert abs(row.threshold-choice['threshold'])<1e-12 and abs(row.inner_score-choice['inner_score'])<1e-10
            cache=np.load(n.OUT/f'outer{fold}_{arm}.npz')
            np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['val'],val)
            values=cache['values'];p=n.make_priority(values,row.threshold)
            np.testing.assert_allclose(oof.loc[val,arm+'_priority'],p,atol=1e-10)
            result=n.ab.evaluate(y[val],p,data.id.iloc[val])
            for k,v in result.items():assert abs(v-row[k])<1e-9
            row2=stability[stability.outer_fold.eq(fold)&stability.family.eq(arm)].iloc[0]
            assert abs(row2.threshold-row.threshold)<1e-12
            cache=np.load(n.OUT/f'stability_outer{fold}_{arm}.npz');values=cache['values']
            np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['val'],val)
            result=n.ab.evaluate(y[val],n.make_priority(values,row.threshold),data.id.iloc[val])
            for k,v in result.items():assert abs(v-row2[k])<1e-9
            count+=2
            for file in [n.OUT/f'outer{fold}_{arm}.json',n.OUT/f'stability_outer{fold}_{arm}.json']:
                info=json.loads(file.read_text());assert len(info['loss_by_epoch'])==3
                expected=0 if arm=='frozen' else 113040
                assert info['trained_encoder_parameters']==expected
                training_logs+=1
    result=dict(outer_scores_recomputed=count,training_logs_checked=training_logs,original_labels_preserved=True,
        calibration_and_outer_groups_disjoint=True,cutoffs_selected_from_training_calibration_only=True,
        secondary_seed_keeps_primary_cutoffs=True,no_encoder_base_weights_trained=True)
    (n.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
