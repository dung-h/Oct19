#!/usr/bin/env python3
"""Independent metric replay and TabPFN outer-context replay."""
from __future__ import annotations
import importlib.util, json
import numpy as np
import pandas as pd
import tabpfn_lib as x
from metric_reference import official

def runner():
    spec=importlib.util.spec_from_file_location("tabpfn_runner",x.HERE/"run.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    ctx=x.context();runner().protocol(ctx);sha=x.digest(x.OUT/"protocol.json")
    assert (x.OUT/"outer_run.json").is_file()
    metrics=pd.read_csv(x.OUT/"fold_metrics.csv");oof=pd.read_csv(x.OUT/"outer_predictions.csv",dtype={"id":str})
    checked=[]
    for fold,fit,pred,nums in x.jobs(ctx):
        for k in range(4):
            name=f"outer{fold}_inner{k}";mask=nums==k
            p,_=x.load_prediction(x.OUT/f"{name}_tabpfn.npz",fit[~mask],fit[mask],sha)
            assert p.shape==(mask.sum(),5);checked.append(name)
        name=f"outer{fold}_validation";saved,cats=x.load_prediction(x.OUT/f"{name}_tabpfn.npz",fit,pred,sha); _,a,b=x.pack(name,fit,pred);xf,xp,replayed_cats=x.matrix(a,b)
        assert cats==replayed_cats
        replay=x.probability(x.tabpfn(),xf,ctx["labels"][fit]-1,xp)
        np.testing.assert_allclose(replay,saved,atol=1e-5,rtol=1e-5)
        for arm in metrics.arm.unique():
            d=oof[(oof.outer_fold==fold)&(oof.arm==arm)]
            assert d.id.tolist()==ctx["ids"][pred].astype(str).tolist()
            independently=official(ctx["labels"][pred],d.priority.to_numpy(float),ctx["ids"][pred])
            row=metrics[(metrics.outer_fold==fold)&(metrics.arm==arm)].iloc[0]
            for key,value in independently.items():assert np.isclose(row[key],value,atol=1e-12),(fold,arm,key,row[key],value)
        checked.append(name)
    result=dict(status="PASSED",partitions_checked=len(checked),tabpfn_outer_replay=True,independent_metric=True,
                heldout_labels_not_used_for_feature_matrix=True,protocol_sha256=sha,uploaded=False)
    (x.OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
