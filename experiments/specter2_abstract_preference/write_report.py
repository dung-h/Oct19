#!/usr/bin/env python3
"""Report the complete pilot and paired contrasts only after verification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
ORDER=['incumbent','title_adapted','joint_frozen','joint_adapted','neural_selection','selection_procedure']


def diagnostic_h(y,p,ids):
    selected=sorted(np.flatnonzero(p>0),key=lambda i:(-p[i],str(ids[i])))
    if not selected:return 0.
    w=1/np.log2(np.arange(len(selected))+2)
    q=float(np.average((y[selected]-1)/4,weights=w))
    recall=float(sum(y[selected]>1)/sum(y>1))
    return 2*q*recall/(q+recall) if q+recall else 0.


def main():
    verified=json.loads((OUT/'verification.json').read_text());assert verified['status']=='PASSED'
    frame=pd.read_csv(OUT/'fold_metrics.csv')
    summary=frame.groupby('policy',sort=False).agg(H=('score','mean'),H_std=('score','std'),
        Q=('relevance','mean'),R=('coverage','mean'),AUC=('AUC','mean'),train_AUC=('train_AUC','mean'),
        logloss=('logloss','mean'),gain_MAE=('gain_MAE','mean'),FP=('FP','sum'),FN=('FN','sum'),
        missed_high=('missed_high','sum')).reindex(ORDER)
    summary['AUC_gap']=summary.train_AUC-summary.AUC
    summary.to_csv(OUT/'summary.csv')
    pairs=[('joint_adapted','title_adapted'),('joint_adapted','joint_frozen'),
        ('joint_adapted','incumbent'),('selection_procedure','incumbent')]
    effects=[];fold_effects=[]
    for arm,reference in pairs:
        a=frame[frame.policy.eq(arm)].set_index('outer_fold')
        b=frame[frame.policy.eq(reference)].set_index('outer_fold')
        delta=a.score-b.score
        effects.append(dict(arm=arm,reference=reference,delta_H=float(delta.mean()),improved_folds=int((delta>0).sum()),
            delta_Q=float((a.relevance-b.relevance).mean()),delta_R=float((a.coverage-b.coverage).mean()),
            delta_AUC=float((a.AUC-b.AUC).mean()),delta_FP=int(a.FP.sum()-b.FP.sum()),
            delta_FN=int(a.FN.sum()-b.FN.sum()),delta_missed_high=int(a.missed_high.sum()-b.missed_high.sum())))
        for f in delta.index:
            fold_effects.append(dict(outer_fold=int(f),arm=arm,reference=reference,delta_H=float(delta.loc[f])))
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'paired_effects.csv',index=False)
    pd.DataFrame(fold_effects).to_csv(OUT/'paired_effects_by_fold.csv',index=False)
    input_audit=pd.read_csv(OUT/'input_audit.csv',dtype={'id':str})
    oof=pd.read_csv(OUT/'oof.csv',dtype={'id':str})
    y=oof.Label.to_numpy();positive=y>1
    accounting=[]
    for arm,reference in pairs:
        for fold,part in oof.groupby('outer_fold'):
            a=part[[arm+f'_p{k}' for k in range(1,6)]].to_numpy()
            b=part[[reference+f'_p{k}' for k in range(1,6)]].to_numpy()
            threshold_a=frame[frame.outer_fold.eq(fold)&frame.policy.eq(arm)].threshold.iloc[0]
            threshold_b=frame[frame.outer_fold.eq(fold)&frame.policy.eq(reference)].threshold.iloc[0]
            def crossed(gate,cutoff,ranker):
                priority=np.where(1-gate[:,0]>cutoff,10*(ranker@(np.arange(5)/4)),0.)
                return diagnostic_h(part.Label.to_numpy(),priority,part.id.to_numpy())
            aa=crossed(a,threshold_a,a);bb=crossed(b,threshold_b,b)
            ab=crossed(a,threshold_a,b);ba=crossed(b,threshold_b,a)
            gate=.5*((ab-bb)+(aa-ba));rank=.5*((ba-bb)+(aa-ab))
            assert abs(gate+rank-(aa-bb))<1e-12
            accounting.append(dict(arm=arm,reference=reference,outer_fold=int(fold),
                delta_H=aa-bb,gate_contribution=gate,rank_contribution=rank))
    pd.DataFrame(accounting).to_csv(OUT/'gate_rank_accounting_by_fold.csv',index=False)
    accounting_summary=pd.DataFrame(accounting).groupby(['arm','reference'],sort=False)[['delta_H','gate_contribution','rank_contribution']].mean()
    accounting_summary.to_csv(OUT/'gate_rank_accounting.csv')
    lookup=input_audit[input_audit['mode'].eq('joint')&input_audit.split.eq('train')].set_index('id')
    transitions=[];groups=[]
    base=oof.title_adapted_priority.to_numpy()>0
    for arm in ORDER:
        selected=oof[arm+'_priority'].to_numpy()>0
        for i in range(len(oof)):
            transitions.append(dict(id=oof.id.iloc[i],title=oof.title.iloc[i],Label=int(y[i]),policy=arm,
                before=('TP' if positive[i] else 'FP') if base[i] else ('FN' if positive[i] else 'TN'),
                after=('TP' if positive[i] else 'FP') if selected[i] else ('FN' if positive[i] else 'TN'),
                fixed=int(base[i]!=positive[i] and selected[i]==positive[i]),
                introduced=int(base[i]==positive[i] and selected[i]!=positive[i]),
                abstract_available=bool(lookup.loc[oof.id.iloc[i],'abstract_available']),
                truncated=bool(lookup.loc[oof.id.iloc[i],'truncated'])))
        for field,values in [('venue',oof.venue),('Label',oof.Label),
            ('abstract_available',oof.id.map(lookup.abstract_available)),('truncated',oof.id.map(lookup.truncated))]:
            for value in pd.unique(values):
                mask=np.asarray(values==value)
                groups.append(dict(policy=arm,field=field,value=value,n=int(mask.sum()),
                    FP=int((selected&~positive&mask).sum()),FN=int((~selected&positive&mask).sum()),
                    missed_high=int((~selected&(y>=4)&mask).sum())))
    pd.DataFrame(transitions).to_csv(OUT/'failure_transitions.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'failure_breakdown.csv',index=False)
    traces=pd.read_csv(OUT/'learning_traces.csv')
    calibration=traces[traces.phase.eq('calibration')]
    calibration.groupby(['arm','epoch']).agg(mean_loss=('loss','mean'),mean_elapsed=('seconds','mean')).to_csv(OUT/'calibration_learning_summary.csv')
    selected=[]
    for f in range(5):
        choices=json.loads((OUT/f'outer{f}_choices.json').read_text())
        for policy,item in choices.items():selected.append(dict(outer_fold=f,policy=policy,**item))
    pd.DataFrame(selected).to_csv(OUT/'selected_epochs.csv',index=False)
    recipe=json.loads((OUT/'final_recipe.json').read_text())
    stats=json.loads((OUT/'input_stats.json').read_text())
    abstract_effect=effects[(effects.arm=='joint_adapted')&(effects.reference=='title_adapted')].iloc[0]
    adaptation_effect=effects[(effects.arm=='joint_adapted')&(effects.reference=='joint_frozen')].iloc[0]
    policy_effect=effects[(effects.arm=='selection_procedure')&(effects.reference=='incumbent')].iloc[0]
    title_row=summary.loc['title_adapted'];joint_row=summary.loc['joint_adapted']
    adaptation_row=summary.loc['joint_frozen']
    diagnosis=(
        f"Joint adapted so với title adapted: Q {abstract_effect.delta_Q:+.5f}, "
        f"recall {abstract_effect.delta_R:+.5f}, read AUC {abstract_effect.delta_AUC:+.5f}; "
        f"FP {int(abstract_effect.delta_FP):+d}, FN {int(abstract_effect.delta_FN):+d}, "
        f"bỏ sót nhãn4/5 {int(abstract_effect.delta_missed_high):+d}. "
        f"AUC train/outer: title {title_row.train_AUC:.5f}/{title_row.AUC:.5f}, "
        f"joint {joint_row.train_AUC:.5f}/{joint_row.AUC:.5f}, "
        f"joint frozen {adaptation_row.train_AUC:.5f}/{adaptation_row.AUC:.5f}."
    )
    if abstract_effect.delta_H >= .005 and abstract_effect.improved_folds >= 4:
        conclusion='Joint adapted input showed a material pilot H improvement over title adapted; fuller validation is still required.'
    elif abstract_effect.delta_H > 0:
        conclusion='Joint adapted was numerically higher than title adapted, but the +H difference is too small and inconsistent to treat as a material improvement.'
    else:
        conclusion='Joint adapted input did not improve mean pilot H over title adapted in this recipe.'
    decision=dict(conclusion=conclusion,abstract_delta_H=float(abstract_effect.delta_H),
        abstract_improved_folds=int(abstract_effect.improved_folds),
        adaptation_delta_H=float(adaptation_effect.delta_H),policy_delta_H=float(policy_effect.delta_H),
        final_choice=recipe['choice'],incumbent_preserved=True,uploaded=False,
        limitation='5 outer folds repeatedly inspected; one calibration group, no claim of independent significance or Kaggle performance')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    text=f'''# SPECTER2 title + abstract học theo preference: pilot

**{conclusion}**

Đã chạy đủ5outer folds. Mỗi outer train dùng inherited inner_fold0 làm calibration
để chọn checkpoint1/2/4/8 và cutoff0..0.90 bước.05; refit toàn outer train đúng
checkpoint rồi chấm outer. Đây là pilot với một calibration group, không là
mean4inner nested evaluation. Outer folds đã được quan sát trong các nghiên cứu
trước nên kết quả exploratory; không gọi là Private Kaggle score/independent holdout.

## Các model thực sự nhận gì?

`title_adapted` nhận title+SEP. `joint_frozen` và `joint_adapted` nhận cùng
title+SEP+usable abstract. Tất cả max512, CLS768 L2, cùng MLP64/dropout.1->5,
CE5+.5BCE(read), headLR1e-3, effective batch32/microbatch8, seed20261021.
Base transformer cố định; adapted cập nhật regression adapter894528 params bằng
LR5e-5, frozen chỉ cập nhật head. Title control được fit lại bằng cùng recipe mới;
không ghép raw prediction từ title run cũ rồi gọi là matched comparison.

Thiếu abstract: fallback title+SEP. Snapshot có2974usable abstract/3090 records.
Title truncated: **{stats['title']['truncated']}**; joint truncated:
**{stats['joint']['truncated']}**, trần512 tokens. Không dùng ID/Label/metadata trong
token input. Không thêm synthetic preference labels.

## Kết quả

{summary.to_markdown(floatfmt='.5f')}

Mean H được tính từ từng outer fold, không harmonic mean của Q/R đã trung bình.
FP/FN/highFN là tổng2494OOF. Incumbent cutoff cũng chọn lại trên chính calibration
của pilot; vì thế incumbent pilot score khác score4inner trước và không được
trộn hai validation designs để báo một delta model thuần túy.

{effects.to_markdown(index=False,floatfmt='.5f')}

{diagnosis}

Train AUC và outer AUC được đo sau refit tại checkpoint đã chọn. Gap lớn cho thấy
recipe khái quát hóa kém hơn trên các bài held out; riêng gap này không xác định
nguyên nhân là embedding, độ nhiễu nhãn hay thiếu dữ liệu. Số FP/FN và phần
gate/rank bên dưới giúp phân biệt đổi cutoff với cải thiện thứ tự xếp hạng.

`joint_adapted-title_adapted` đo thay input với cùng cách cập nhật adapter/head;
`joint_adapted-joint_frozen` đo cập nhật adapter trên cùng joint text/head. Training
epoch được chọn riêng bằng calibration của mỗi arm, do đó contrasts đánh giá cả
quy trình chọn checkpoint, không chỉ raw encoder ở một epoch chung. Hai đầu vào
có lượng text/padding khác nhau; không suy ra người gán nhãn đọc abstract hay không.
Kết quả chỉ áp dụng regression adapter/LR/loss/head và lịch1/2/4/8 ở đây. Nếu
checkpoint8 được chọn thường xuyên, pilot chưa xác định hội tụ hoặc ngân sách
train tối ưu cho mỗi arm. Không kết luận abstract nói chung vô ích, hoặc mọi
full fine-tuning, pretrained adapter khác và cross-encoder đều sẽ giống kết quả này.

## Checkpoint, learning traces và failures

Đối chiếu gate chọn/bỏ và gain ranker giữa mỗi cặp, giữ nguyên cutoff calibration:

{accounting_summary.to_markdown(floatfmt='.5f')}

Hai contributions là trung bình hai thứ tự thay component và cộng đúng deltaH.
Đây là phân rã metric hậu nghiệm để khoanh failure, không chứng minh nguyên nhân
gốc hoặc là candidate hybrid được tune/triển khai bằng outer labels.

{pd.DataFrame(selected).query("policy in ['title_adapted','joint_frozen','joint_adapted']")[['outer_fold','policy','epoch','threshold','calibration_H']].to_markdown(index=False,floatfmt='.5f')}

Raw loss/time/epoch ở `outputs/learning_traces.csv`, input text/token count ở
`input_audit.csv`, failures từng ID ở `failure_transitions.csv`, theo venue/label/
abstract available/truncation ở `failure_breakdown.csv`. Các ca đã xem chỉ là
diagnostic probes; không sửa prediction/weight theo ID hoặc outer labels.

## Final và kiểm chứng

Final inherited inner_fold0 chọn:

```json
{json.dumps(recipe,indent=2)}
```

Refit toàn2494, các model cuối riêng được lưu cho cả ba nhánh, CSV theo template596
IDs. Candidate ở `outputs/submission_candidate.csv`; incumbent gốc giữ nguyên.
Chưa upload Kaggle. Không có lời hứa H>0.7 từ kiến trúc hoặc embedding dimension.

Independent verifier **{verified['status']}**: {verified['prediction_caches_checked']}
prediction caches, {verified['outer_metric_rows_recomputed']} outer metric rows,
{verified['calibration_choices_recomputed']} calibration choices và
{verified['gradient_and_drift_records_checked']} gradient/drift records. Encode lại
{verified['frozen_vector_samples_reencoded']} frozen vector samples; initial head
đồng nhất, nhãn nguồn giữ nguyên, tokenizer không đọc Label/ID; reload cả
{verified['final_models_reloaded']} final models cho cùng probabilities. Có
{verified['final_train_test_overlapping_groups']} publication groups giao train/test
trong source dataset; outer fit/validation groups tách biệt. Test labels không được dùng.

Xem [PLAN.md](PLAN.md), [verification.json](outputs/verification.json),
[paired effects](outputs/paired_effects.csv), [decision.json](outputs/decision.json).
'''
    (HERE/'RESULTS.md').write_text(text)
    (HERE/'DECISION.md').write_text('# Quyết định pilot\n\n'+conclusion+'\n\n'+
        f"Joint adapted-title adapted delta H={abstract_effect.delta_H:+.5f}, wins={int(abstract_effect.improved_folds)}/5; joint adapted-frozen={adaptation_effect.delta_H:+.5f}. Final family={recipe['choice']['family']}.\n\n"+
        'Giữ incumbent gốc. Pilot inner0, outer folds cũ: exploratory, không có kiểm định độc lập hoặc upload Kaggle. Chi tiết: [RESULTS.md](RESULTS.md).\n')
    print(summary.to_string());print(effects.to_string(index=False));print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
