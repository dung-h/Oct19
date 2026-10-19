#!/usr/bin/env python3
"""Summarize the completed fixed ordinal experiment without choosing on outer scores."""
import json
import numpy as np
import pandas as pd

import run_ordinal as run


def main():
    out=run.OUT
    folds=pd.read_csv(out/'fold_metrics.csv')
    assert len(folds)==15 and folds.outer_fold.nunique()==5
    summary=folds.groupby('family',sort=False).agg(H=('score','mean'),H_std=('score','std'),Q=('relevance','mean'),
        R=('coverage','mean'),FP=('FP','sum'),FN=('FN','sum'),missed_high=('missed_high','sum'),AUC=('AUC','mean'),
        logloss=('logloss','mean'),gain_MAE=('gain_MAE','mean'),train_AUC=('train_AUC','mean'),AUC_gap=('AUC_gap','mean'))
    summary.to_csv(out/'summary.csv')
    tail=pd.read_csv(out/'tail_metrics.csv')
    ts=tail.groupby(['family','target','stage']).agg(AUC=('AUC','mean'),Brier=('Brier','mean')).reset_index()
    ts.to_csv(out/'tail_summary.csv',index=False)
    comparisons=[]
    reference=folds[folds.family.eq('multiclass')].set_index('outer_fold')
    for family in ['ordinal','selection_procedure']:
        part=folds[folds.family.eq(family)].set_index('outer_fold')
        for k,row in part.iterrows():
            comparisons.append(dict(outer_fold=k,family=family,
                **{'delta_'+col:float(row[col]-reference.loc[k,col]) for col in ['score','relevance','coverage','FP','FN','missed_high','AUC','gain_MAE']}))
    paired=pd.DataFrame(comparisons);paired.to_csv(out/'paired_effects.csv',index=False)
    info=[]
    for path in sorted(out.glob('*_training.json')):
        d=json.loads(path.read_text())
        info.append(dict(partition=path.name.removesuffix('_training.json'),n_pred=d['n_pred'],crossing_rows=d['crossing_rows'],
                         crossing_rate=d['crossing_rows']/d['n_pred'],mean_abs_read_change=d['mean_abs_read_change'],
                         max_abs_gain_change=d['max_abs_gain_change']))
    diagnostics=pd.DataFrame(info);diagnostics.to_csv(out/'projection_diagnostics.csv',index=False)
    outer=diagnostics[diagnostics.partition.str.contains('_validation')]
    d=pd.read_csv(out/'oof_predictions.csv',dtype={'id':str});truth=d.Label.gt(1)
    old=d.multiclass_priority.gt(0);new=d.ordinal_priority.gt(0)
    transitions=d[['id','title','venue','Label','outer_fold']].copy()
    transitions['old_status']=np.where(old,np.where(truth,'TP','FP'),np.where(truth,'FN','TN'))
    transitions['new_status']=np.where(new,np.where(truth,'TP','FP'),np.where(truth,'FN','TN'))
    transitions['old_priority']=d.multiclass_priority;transitions['new_priority']=d.ordinal_priority
    transitions['fixed']=old.ne(truth)&new.eq(truth);transitions['introduced']=old.eq(truth)&new.ne(truth)
    transitions.to_csv(out/'failure_transitions.csv',index=False)
    breakdown=[]
    for family in run.FAMILIES:
        selected=d[family+'_priority'].gt(0)
        for grouping in ['venue','Label']:
            for name,ix in d.groupby(grouping).groups.items():
                s=selected.loc[ix];p=truth.loc[ix]
                breakdown.append(dict(family=family,grouping=grouping,group=str(name),n=len(ix),
                    FP=int((s&~p).sum()),FN=int((~s&p).sum()),TP=int((s&p).sum()),TN=int((~s&~p).sum())))
    pd.DataFrame(breakdown).to_csv(out/'failure_breakdown.csv',index=False)
    choices=json.loads((out/'final_choices.json').read_text())
    verify=json.loads((out/'verification.json').read_text())
    delta=float(summary.loc['ordinal','H']-summary.loc['multiclass','H'])
    wins=int((paired[paired.family.eq('ordinal')].delta_score>0).sum())
    decision=dict(ordinal_delta_H=delta,ordinal_improved_folds=wins,
        policy_delta_H=float(summary.loc['selection_procedure','H']-summary.loc['multiclass','H']),
        final_inner_choice=choices['selection_procedure'],
        incumbent_artifacts_unchanged=True,
        caveat='Exploratory outer reuse; this descriptive recommendation is not independent evidence or a statistical significance test',
        uploaded=False)
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    recommendation='Đối chứng hoàn tất; giữ nguyên artifact incumbent, lưu ứng viên ordinal riêng để đối chiếu.'
    text=f'''# Cumulative ordinal classification — kết quả

**{recommendation}**

Giữ đúng277features,5outer/4inner groups,seed20261018 và policy chọn/xếp hạng.
Thay một CatBoost5lớp bằng4binary CatBoost cho y>=2/3/4/5, mỗi model300trees,
depth4,lr0.05,L2=8. Total capacity/compute khác với multiclass; đây là đối chứng
toàn formulation. Chưa thêm calibration hoặc đổi encoder.

{summary.reset_index().to_markdown(index=False,floatfmt='.5f')}

Ordinal − multiclass: **{delta:+.5f} H**, tốt hơn **{wins}/5folds**.
Điểm toàn quy trình chọn family bằng inner báo riêng ở `selection_procedure`.
H tính từng fold rồi lấy trung bình; không tính harmonic mean của hai cột Q/R đã
được trung bình. Các outer folds đã được xem ở nhiều nghiên cứu trước.

## Paired effects

{paired.to_markdown(index=False,floatfmt='.5f')}

## Tính nhất quán xác suất

q2=P(y>=2),q3=P(y>=3),q4=P(y>=4),q5=P(y>=5). P(read)=q2;
expected gain=(q2+q3+q4+q5)/4. PAV sửa4giá trị mỗi bài thành dãy không tăng,
không dùng nhãn, giữ tổng/gain. Khi pooled tails bằng nhau, một số xác suất lớp
giữa có thể bằng0; multiclass logloss có thể phạt mạnh dù H/gain vẫn hợp lệ.

Trên2494outer predictions: **{int(outer.crossing_rows.sum())}** dòng có crossing
trước projection. Toàn bộ corrections ở `outputs/projection_diagnostics.csv`.

{ts.to_markdown(index=False,floatfmt='.5f')}

Các AUC/Brier này dùng để chẩn đoán từng ngưỡng; không thay H để chọn model.

## Failure và chọn cuối

So với multiclass, ordinal sửa **{int(transitions.fixed.sum())}** quyết định nhị phân,
phát sinh **{int(transitions.introduced.sum())}** lỗi mới. Chi tiết từng ID ở
`outputs/failure_transitions.csv`, theo venue/label ở `failure_breakdown.csv`.
Các thay đổi không chứng minh nguyên nhân sở thích của người gán nhãn.

```json
{json.dumps(choices,indent=2)}
```

`outputs/submission_ordinal.csv`: đủ596ID, **{verify['ordinal_selected']}** priority
dương. `outputs/submission_candidate.csv`: chọn theo final inner policy, **{verify['candidate_selected']}**
priority dương. Bản cũ không bị ghi đè; chưa gửi Kaggle. Bốn model lưu ở
`outputs/ordinal_ge2.cbm` tới `ordinal_ge5.cbm`, recipe ở `final_recipe.joblib`.

## Xác minh và giới hạn

4unit tests,120binary fits được audit,90inherited group crossfits,15outer metric
rows và80tail metric rows tính lại,18inner choices xác minh. PAV được tính lại
bằng solver isotonic độc lập. Rebuild test features từ recipe, reload4models cho
cùng xác suất/priority/toàn ranking; nhãn gốc và CSV incumbent giữ nguyên.
Trạng thái: **{verify['status']}**, chi tiết `outputs/verification.json`.

Đây là cumulative classification +PAV, không phải neural CORAL. Không có bảo đảm
thắng CE5lớp hoặc đạt0.7; xác suất được chiếu nhất quán chưa đồng nghĩa calibrated.
Dùng H từ phần Evaluation đã chép; mâu thuẫn với Overview60/40 chưa được xác nhận
bên ngoài. Chưa áp dụng thay đổi metric hoặc chọn family bằng nhãn test.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_preference/run_ordinal.py all
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_preference/verify.py
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_preference/write_report.py
```

Tham khảo về ordinal binary subtasks: [CORAL](https://arxiv.org/abs/1901.07884).
'''
    (run.HERE/'RESULTS.md').write_text(text)
    print(summary.to_string());print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
