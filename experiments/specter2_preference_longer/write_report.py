#!/usr/bin/env python3
"""Summarize the locked 8-vs-16 epoch experiment after independent verification."""
import json
import pandas as pd
from pathlib import Path

HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'


def main():
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['status']=='PASSED'
    frame=pd.read_csv(OUT/'fold_metrics.csv')
    summary=frame.groupby('policy',sort=False).agg(H=('score','mean'),Q=('relevance','mean'),
        R=('coverage','mean'),AUC=('AUC','mean'),FP=('FP','sum'),FN=('FN','sum'),
        missed_high=('missed_high','sum')).reindex(['adapted_8','adapted_16','incumbent','neural_selection','selection_procedure'])
    summary.to_csv(OUT/'summary.csv')
    a=frame[frame.policy.eq('adapted_16')].set_index('outer_fold')
    b=frame[frame.policy.eq('adapted_8')].set_index('outer_fold')
    delta=a.score-b.score
    comparison=dict(delta_H=float(delta.mean()),improved_folds=int((delta>0).sum()),
        delta_R=float((a.coverage-b.coverage).mean()),delta_Q=float((a.relevance-b.relevance).mean()),
        delta_FN=int(a.FN.sum()-b.FN.sum()),delta_missed_high=int(a.missed_high.sum()-b.missed_high.sum()))
    pd.DataFrame([dict(outer_fold=int(f),delta_H=float(delta.loc[f]),delta_R=float(a.loc[f,'coverage']-b.loc[f,'coverage']),
        delta_Q=float(a.loc[f,'relevance']-b.loc[f,'relevance']),delta_FN=int(a.loc[f,'FN']-b.loc[f,'FN']),
        delta_missed_high=int(a.loc[f,'missed_high']-b.loc[f,'missed_high'])) for f in delta.index]).to_csv(OUT/'epoch16_vs8_by_fold.csv',index=False)
    (OUT/'epoch16_vs8_summary.json').write_text(json.dumps(comparison,indent=2)+'\n')
    choice=json.loads((OUT/'final_choices.json').read_text())
    recipe=json.loads((OUT/'final_recipe.json').read_text())
    pass_go=(comparison['delta_H']>0 and comparison['improved_folds']>=3 and
        comparison['delta_missed_high']<=0 and comparison['delta_R']>=-.01)
    decision='PROCEED to a separate title+abstract matched-input experiment' if pass_go else 'STOP this extension; retain the existing incumbent'
    body=f'''# SPECTER2 title-only: epoch 8 vs 16

The same title-only input, pretrained adapter, head, loss, optimizer, folds and seed were used. Each partition trained once to epoch16; epoch8 was read from that run. The archived epoch8 predictions were checked to match. Thresholds and the global family were selected by four inner folds. The outer folds are previously studied and this remains exploratory.

| Policy | H | Q | Recall | AUC | FP | FN | Missed labels 4/5 |
|---|---:|---:|---:|---:|---:|---:|---:|
{summary.to_markdown(index=True,floatfmt='.5f')}

## Epoch 16 versus epoch 8

- Mean outer H change: **{comparison['delta_H']:+.5f}**; improved in **{comparison['improved_folds']}/5** folds.
- Mean recall change: **{comparison['delta_R']:+.5f}**; mean Q change: **{comparison['delta_Q']:+.5f}**.
- Total FN change: **{comparison['delta_FN']:+d}**; missed labels4/5 change: **{comparison['delta_missed_high']:+d}**.

## Next step from the locked rule

**{decision}.** The gate was predeclared as higher mean H, improvement on at least3/5 folds, no increase in missed high-rated papers, and mean recall loss no worse than.01. This is a conservative research stop/go rule, not part of the assignment metric. It does not claim the epoch is converged or guarantee a Kaggle gain.

Final inner choices:

```json
{json.dumps(choice,indent=2)}
```

Global final family: `{recipe['family']}`, checkpoint `{recipe['epoch']}`. `outputs/submission_candidate.csv` is the selected candidate; the existing incumbent is preserved. No upload was made.

Independent verification: `{verification['status']}`; {verification['outer_policy_metric_rows_recomputed']} outer policy rows and {verification['final_inner_choices_recomputed']} final choices recomputed; archived checkpoint8 replay, final-test epoch8 replay ({verification['final_test_epoch8_replay']}), labels, candidate format, and selected model reload checked. The original run had no saved final-test cache, so that one comparison used a fresh replay of the identical epoch8 recipe. Full per-fold deltas are in `outputs/epoch16_vs8_by_fold.csv`.
'''
    (HERE/'RESULTS.md').write_text(body)
    (HERE/'DECISION.md').write_text('# Quyết định\n\n'+decision+'.\n\n'+
        f"Chênh lệch H trung bình epoch16−epoch8={comparison['delta_H']:+.5f} ({comparison['improved_folds']}/5 folds tăng), recall={comparison['delta_R']:+.5f}, bỏ sót nhãn4/5={comparison['delta_missed_high']:+d}.\n\n"+
        ('Nếu đạt tiêu chí đã khóa, bước tiếp là so title-only với title+abstract bằng cùng adapter/head/fold và inner-selection. Nếu không đạt, dừng nhánh này và giữ incumbent. ' if pass_go else 'Không mở rộng sang title+abstract trong lượt này; dữ liệu không cho thấy kéo dài training đáng để nhân rộng thí nghiệm. ')+
        'Đây là CV thăm dò trên fold cũ, không phải điểm leaderboard mới. Chi tiết và giới hạn trong [RESULTS.md](RESULTS.md).\n')
    print(json.dumps(dict(comparison=comparison,pass_go=pass_go,selected=recipe),indent=2))


if __name__=='__main__':main()
