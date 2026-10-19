#!/usr/bin/env python3
"""Write the preregistered report and decision for TabPFN A."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import tabpfn_lib as x

def main():
    v=json.loads((x.OUT/"verification.json").read_text());assert v["status"]=="PASSED"
    fold=pd.read_csv(x.OUT/"fold_metrics.csv");oof=pd.read_csv(x.OUT/"outer_predictions.csv",dtype={"id":str})
    summary=fold.groupby("arm").agg(mean_H=("H","mean"),mean_Q=("Q","mean"),mean_R=("R","mean"),accepted=("accepted","sum"),TP=("TP","sum"),FP=("FP","sum"),FN=("FN","sum"),total_error=("total_error","sum"),missed_label4=("missed_label4","sum"),missed_label5=("missed_label5","sum"),mean_AUC=("read_AUC","mean"),mean_Brier=("Brier","mean")).reset_index();summary.to_csv(x.OUT/"summary.csv",index=False)
    c0=oof[oof.arm=="C0_deployed"].set_index("id").sort_index();t1=oof[oof.arm=="T1_primary"].set_index("id").sort_index();assert c0.index.equals(t1.index)
    y=c0.Label.to_numpy();change=np.select([c0.selected&~t1.selected&(y==1),~c0.selected&t1.selected&(y>1),~c0.selected&t1.selected&(y==1),c0.selected&~t1.selected&(y>1)],["FP_removed","FN_rescued","new_FP","lost_TP"],default="unchanged")
    trans=pd.DataFrame(dict(id=c0.index,Label=y,outer_fold=c0.outer_fold,transition=change,c0_selected=c0.selected,t1_selected=t1.selected));trans.to_csv(x.OUT/"transitions.csv",index=False)
    get=lambda arm:summary[summary.arm==arm].iloc[0]
    c0s,c1,t1=get("C0_deployed"),get("C1_matched"),get("T1_primary")
    win_c0=int((fold[fold.arm=="T1_primary"].set_index("outer_fold").H>fold[fold.arm=="C0_deployed"].set_index("outer_fold").H).sum());win_c1=int((fold[fold.arm=="T1_primary"].set_index("outer_fold").H>fold[fold.arm=="C1_matched"].set_index("outer_fold").H).sum())
    promising=bool(t1.mean_H>=c0s.mean_H+.005 and t1.mean_H>=c1.mean_H+.005 and win_c0>=3 and win_c1>=3)
    decision=dict(status="PROMISING" if promising else "NO_PROMOTION",promising=promising,delta_H_vs_C0=float(t1.mean_H-c0s.mean_H),delta_H_vs_C1=float(t1.mean_H-c1.mean_H),wins_vs_C0=win_c0,wins_vs_C1=win_c1,transitions={k:int(v) for k,v in pd.Series(change).value_counts().items()},uploaded=False)
    (x.OUT/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    table="\n".join(f"| {r.arm} | {r.mean_H:.6f} | {r.mean_Q:.6f} | {r.mean_R:.6f} | {int(r.FP)} | {int(r.FN)} | {int(r.missed_label4)} | {int(r.missed_label5)} | {r.mean_AUC:.6f} |" for r in summary.itertuples())
    (x.HERE/"RESULTS.md").write_text("# TabPFN preference — verified grouped outer OOF\n\n| arm | H | Q | R | FP | FN | missed L4 | missed L5 | read AUC |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"+table+f"\n\nT1 thắng C0 ở {win_c0}/5 folds và C1 ở {win_c1}/5 folds. Verifier passed.\n")
    (x.HERE/"DECISION.md").write_text(f"# Quyết định TabPFN A\n\n**{decision['status']}**. Delta mean H T1−C0={decision['delta_H_vs_C0']:.6f}; T1−C1={decision['delta_H_vs_C1']:.6f}. Không upload Kaggle.\n")
    print(json.dumps(decision,indent=2))
if __name__=="__main__":main()
