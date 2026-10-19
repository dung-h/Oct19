#!/usr/bin/env python3
"""Count-first report with matched text/roles controls and exact error transitions."""
import json

import numpy as np
import pandas as pd

import experiment as e
from roles import HERE,OUT,ROLES


def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='PASSED'
    folds=pd.read_csv(OUT/'fold_metrics.csv');oof=pd.read_csv(OUT/'outcomes_by_id.csv',dtype={'id':str})
    summary=folds.groupby('branch',sort=False).agg(accepted=('accepted','sum'),dropped=('dropped','sum'),
        TP=('TP','sum'),FP=('FP','sum'),FN=('FN','sum'),TN=('TN','sum'),total_error=('total_error','sum'),
        missed_high=('missed_high','sum'),H=('H','mean'),Q=('Q','mean'),R=('R','mean'),AUC=('AUC','mean'))
    summary.to_csv(OUT/'summary.csv')
    base=oof[oof.branch.eq('incumbent')].set_index('id');transitions=[];detailed=[]
    for branch in e.BRANCHES:
        current=oof[oof.branch.eq(branch)].set_index('id').loc[base.index]
        old=base.selected.to_numpy(bool);new=current.selected.to_numpy(bool);positive=base.Label.to_numpy()>1
        kind=np.select([~old&new&positive,old&~new&~positive,~old&new&~positive,old&~new&positive],
            ['FN_rescued','FP_removed','new_FP','lost_TP'],default='unchanged')
        counts={key:int((kind==key).sum()) for key in ['FN_rescued','FP_removed','new_FP','lost_TP']}
        counts['delta_error']=counts['new_FP']+counts['lost_TP']-counts['FN_rescued']-counts['FP_removed']
        assert counts['delta_error']==int(summary.loc[branch,'total_error']-summary.loc['incumbent','total_error'])
        transitions.append(dict(branch=branch,**counts))
        detailed.append(pd.DataFrame(dict(id=base.index,branch=branch,Label=base.Label.to_numpy(),
            title=base.title.to_numpy(),old_selected=old,new_selected=new,transition=kind)))
    transition=pd.DataFrame(transitions);transition.to_csv(OUT/'transition_summary.csv',index=False)
    pd.concat(detailed,ignore_index=True).to_csv(OUT/'transitions_by_id.csv',index=False)
    for dimension in ['Label','venue']:
        rows=[]
        for (branch,value),part in oof.groupby(['branch',dimension]):
            selected=part.selected.to_numpy(bool);y=part.Label.to_numpy(int)
            rows.append(dict(branch=branch,group=value,total=len(part),accepted=int(selected.sum()),dropped=int((~selected).sum()),
                FP=int((selected&(y==1)).sum()),FN=int((~selected&(y>1)).sum()),missed_high=int((~selected&(y>=4)).sum())))
        pd.DataFrame(rows).to_csv(OUT/('counts_by_'+dimension+'.csv'),index=False)
    text=pd.read_csv(OUT/'texts.csv',dtype={'id':str},keep_default_na=False)
    coverage=[]
    for split,part in text.groupby('split'):
        for role in ROLES:coverage.append(dict(split=split,role=role,total=len(part),present=int(part[role].str.len().gt(0).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'role_coverage.csv',index=False)
    a=summary.loc[e.PRIMARY];b=summary.loc['incumbent']
    paired=folds.pivot(index='outer_fold',columns='branch',values='total_error')
    wins=int((paired[e.PRIMARY]<paired.incumbent).sum())
    promote=bool(a.FP<=b.FP and a.FN<=b.FN and (a.FP<b.FP or a.FN<b.FN) and a.missed_high<=b.missed_high and wins>=4)
    if promote:status='promising';decision='Primary cải thiện FP/FN và đạt rule đã khóa; lưu candidate để cân nhắc thay incumbent.'
    elif a.total_error<b.total_error:status='mixed';decision='Primary giảm tổng lỗi nhưng chưa đạt đầy đủ rule FP/FN/high-label/folds; giữ incumbent.'
    else:status='negative';decision='Primary chưa giảm tổng lỗi; giữ incumbent. Không mở thêm search trong lượt này.'
    dec=dict(status=status,decision=decision,primary=e.PRIMARY,folds_with_lower_error=wins,
        incumbent_preserved=True,uploaded=False)
    (OUT/'decision.json').write_text(json.dumps(dec,indent=2)+'\n')
    display=summary[['accepted','dropped','FP','FN','total_error','missed_high','H','Q','R']]
    result='# Semantic roles: controlled classification experiment\n\n'+decision+'\n\n'
    formatted=display.copy()
    for col in list(display.columns)[:6]:formatted[col]=display[col].map(lambda x:str(int(x)))
    for col in ['H','Q','R']:formatted[col]=display[col].map(lambda x:f'{x:.5f}')
    result+='## Counts and assignment metric\n\n'+formatted.to_markdown(disable_numparse=True)+'\n\n'
    result+='Counts sum2494 OOF rows; H/Q/R are means of five fold metrics. Ranking always uses incumbent expected gain. Cutoffs selected using four-inner OOF only, subject to incumbent FN/high-FN budgets; constraints are not guaranteed on outer/test.\n\n'
    result+='## Error transitions vs incumbent\n\n'+transition.to_markdown(index=False)+'\n\n'
    result+='Primary reduces total error on '+str(wins)+'/5 outer folds. All four branches were declared in advance; primary is roles_blend, not the best outer branch.\n\n'
    contrasts=[]
    for arm,reference in [('roles_raw','text_raw'),('roles_blend','text_blend'),('roles_blend','incumbent')]:
        contrasts.append(dict(arm=arm,reference=reference,delta_FP=int(summary.loc[arm,'FP']-summary.loc[reference,'FP']),
            delta_FN=int(summary.loc[arm,'FN']-summary.loc[reference,'FN']),delta_error=int(summary.loc[arm,'total_error']-summary.loc[reference,'total_error']),
            delta_H=float(summary.loc[arm,'H']-summary.loc[reference,'H'])))
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    result+='## Matched representation contrasts\n\n'+pd.DataFrame(contrasts).to_markdown(index=False)+'\n\n'
    result+='## Evidence extraction\n\n'+pd.DataFrame(coverage).to_markdown(index=False)+'\n\n'
    result+='Invalid extractions: '+str(int(text.invalid.sum()))+'. Each present role is an exact source sentence, not a generated paraphrase. This guarantees provenance, not semantic role correctness. Qwen receives title/abstract only; no ratings/IDs/metadata/predictions. Original full text retained in both feature arms. No SVD; all384 dimensions retained per dense block.\n\n'
    test=pd.read_csv(OUT/'test_counts.csv');result+='## Final artifacts\n\n'+test.to_markdown(index=False)+'\n\n'
    result+='Test has no known ratings: only selected/dropped/changed-ID counts are reported. All596 IDs exported for every branch; no Kaggle upload or incumbent replacement.\n\n'
    result+='Verifier PASSED: independent threshold and metric replay, exact evidence spans, final vocab refit,12 classifier reloads, and submission-order checks. Existing outer folds were repeatedly inspected; results are exploratory. Selected diagnostic pairs are not an unseen semantic benchmark.\n'
    (HERE/'RESULTS.md').write_text(result)
    (HERE/'DECISION.md').write_text('# Decision\n\n'+decision+'\n\nPrimary error-improved folds: '+str(wins)+'/5. See [RESULTS.md](RESULTS.md).\n')
    (HERE/'README.md').write_text('# Semantic roles\n\nUse `/tmp/assignment-specter2-venv/bin/python`:\n\n1. `roles.py preflight`\n2. `roles.py extract`\n3. `roles.py encode`\n4. `test_roles.py`\n5. `run.py all`\n6. `verify.py`\n7. `report.py`\n\nPLAN.md locks the representation, primary branch, operating constraints and stop rule. Hashes prevent accidental reuse after source changes. Old artifacts are preserved.\n')
    print(display.to_string());print(transition.to_string(index=False));print(json.dumps(dec,indent=2))


if __name__=='__main__':main()
