#!/usr/bin/env python3
"""Paired contrasts and honest seed/selection sensitivity, with no best-seed picking."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_claims as run


def table(frame):return frame.to_markdown(index=False,floatfmt='.5f')


def main():
    out=run.OUT;check=json.loads((out/'verification.json').read_text())
    m=pd.read_csv(out/'fold_metrics.csv');s=pd.read_csv(out/'summary.csv')
    oof=pd.read_csv(out/'oof_predictions.csv',dtype={'id':str})
    primary=run.SEEDS[0]
    old=pd.read_csv(run.previous.OUT/'oof_predictions.csv',dtype={'id':str})
    assert np.array_equal(old.id,oof.id)
    pairs=[('regularized','reference'),('repaired','regularized'),('prefix','repaired'),
           ('claims','repaired'),('claims','prefix'),('selection_procedure','reference')]
    effects=[]
    for seed in run.SEEDS:
        for arm,reference in pairs:
            a=m[m.seed.eq(seed)&m.family.eq(arm)].set_index('outer_fold')
            b=m[m.seed.eq(seed)&m.family.eq(reference)].set_index('outer_fold')
            d=a.score-b.score
            effects.append(dict(seed=seed,arm=arm,reference=reference,delta_H=d.mean(),
                improved_folds=int((d>0).sum()),delta_AUC=(a.read_auc-b.read_auc).mean(),
                delta_auc_gap=(a.auc_gap-b.auc_gap).mean(),delta_R=(a.coverage-b.coverage).mean(),
                delta_FP=int(a.FP.sum()-b.FP.sum()),delta_FN=int(a.FN.sum()-b.FN.sum()),
                delta_high_FN=int(a.missed_high.sum()-b.missed_high.sum()),
                **{f'fold{k}_delta':float(v) for k,v in d.items()}))
    effects=pd.DataFrame(effects);effects.to_csv(out/'paired_effects.csv',index=False)
    stability=[]
    for family in run.ARMS+['selection_procedure']:
        selected=s[s.family.eq(family)].set_index('seed')
        first=oof[f'{family}_seed{primary}_priority'].to_numpy()
        row=dict(family=family,H_primary=selected.loc[primary,'H'],H_mean=selected.H.mean(),
                 H_min=selected.H.min(),H_max=selected.H.max(),H_range=selected.H.max()-selected.H.min(),
                 mean_AUC=selected.AUC.mean(),mean_auc_gap=selected.AUC_gap.mean())
        for seed in run.SEEDS[1:]:
            other=oof[f'{family}_seed{seed}_priority'].to_numpy()
            row[f'selection_flips_seed{seed}']=int(((other>0)!=(first>0)).sum())
        stability.append(row)
    stability=pd.DataFrame(stability);stability.to_csv(out/'seed_stability.csv',index=False)
    y=oof.Label.to_numpy();prior_sel=old.selection_procedure_priority.to_numpy()>0
    current=oof[f'selection_procedure_seed{primary}_priority'].to_numpy()>0
    def status(sel):return np.where(y>1,np.where(sel,'TP','FN'),np.where(sel,'FP','TN'))
    failures=oof[['id','title','venue','Label','outer_fold']].copy()
    failures['old_status']=status(prior_sel);failures['new_status']=status(current)
    before=prior_sel!=(y>1);after=current!=(y>1)
    failures['change']=np.select([before&~after,~before&after,before&after],['fixed','introduced','persisting'],default='correct')
    failures.to_csv(out/'failure_transitions.csv',index=False)
    known=failures[failures.id.isin(['217','355','1045','1184','1000','1001','2383','206'])].merge(
        oof[['id']+[f'{arm}_seed{primary}_priority' for arm in run.ARMS]],on='id',validate='one_to_one')
    extracted=pd.read_csv(out/'extracted_texts.csv',dtype={'id':str},keep_default_na=False)
    tr=extracted[extracted.split.eq('train')].copy()
    for kind in ['prefix','claims']:
        tr[kind+'_fields_present']=tr[[kind+'_'+r for r in run.cf.ASPECTS]].ne('').sum(axis=1)
    known=known.merge(tr[['id','prefix_fields_present','claims_fields_present']],on='id',validate='one_to_one')
    known.to_csv(out/'known_cases.csv',index=False)
    coverage_labels=tr[['id','claims_fields_present','prefix_fields_present']].merge(oof[['id','Label']],on='id',validate='one_to_one')
    coverage_labels=coverage_labels.groupby('Label',as_index=False).agg(n=('id','size'),
        no_claim_fields=('claims_fields_present',lambda x:int(x.eq(0).sum())),
        no_prefix_fields=('prefix_fields_present',lambda x:int(x.eq(0).sum())))
    coverage_labels.to_csv(out/'input_coverage_by_label.csv',index=False)
    broken=[]
    for field in ['venue','Label']:
        for value,group in failures.groupby(field):
            broken.append(dict(field=field,value=value,n=len(group),FP=int(group.new_status.eq('FP').sum()),
                FN=int(group.new_status.eq('FN').sum()),fixed=int(group.change.eq('fixed').sum()),
                introduced=int(group.change.eq('introduced').sum())))
    pd.DataFrame(broken).to_csv(out/'failure_breakdown.csv',index=False)
    chosen=json.loads((out/'final_choice.json').read_text())
    reg=effects[effects.seed.eq(primary)&effects.arm.eq('regularized')&effects.reference.eq('reference')].iloc[0]
    claims=effects[effects.arm.eq('claims')&effects.reference.eq('prefix')]
    policy=effects[effects.arm.eq('selection_procedure')]
    policy_primary=policy[policy.seed.eq(primary)].iloc[0]
    favorable=bool(policy_primary.delta_H>0 and policy.delta_H.ge(0).all())
    same_candidate=run.cf.digest(out/'submission_candidate.csv')==run.cf.digest(run.previous.OUT/'submission_candidate.csv')
    interpretation_vi=('Ứng viên đáng được kiểm tra độc lập thêm; mức tăng hiện vẫn là kết quả exploratory.' if favorable else
                       'Chưa có bằng chứng nhất quán để thay model đối chứng; giữ các nhánh mới ở dạng thí nghiệm.')
    decision=dict(primary_seed=primary,final_choice=chosen,
        delta_H_policy_vs_reference_by_seed={str(int(r.seed)):float(r.delta_H) for r in policy.itertuples()},
        delta_H_claims_vs_prefix_by_seed={str(int(r.seed)):float(r.delta_H) for r in claims.itertuples()},
        primary_and_all_sensitivity_seeds_nonnegative_policy_gain=favorable,
        interpretation='candidate merits limited independent validation; gain is exploratory' if favorable else 'no consistent support for replacing reference; retain candidate as experimental',
        interpretation_vi=interpretation_vi,candidate_identical_to_previous=same_candidate,
        no_best_seed_selection=True,no_previous_artifact_replacement=True,no_kaggle_upload=True)
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    labels=run.ARMS+['selection_procedure'];positions=np.arange(len(labels))
    for seed,color in zip(run.SEEDS,['#2563eb','#c2410c','#15803d']):
        selected=s[s.seed.eq(seed)].set_index('family').loc[labels]
        axes[0].plot(positions,selected.H,marker='o',label=str(seed),color=color)
        axes[1].plot(positions,selected.AUC_gap,marker='o',label=str(seed),color=color)
    for ax in axes:
        ax.set_xticks(positions,labels=['reference','regularized','repaired','prefix','claims','selection'],rotation=20)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean outer H');axes[1].set_ylabel('Mean train - validation AUC')
    axes[0].legend(title='Head seed',fontsize=8)
    fig.suptitle('Primary inner choices held fixed for secondary seeds; same five outer folds',fontsize=11)
    fig.tight_layout();fig.savefig(out/'stability_comparison.png',dpi=180);fig.savefig(out/'stability_comparison.pdf');plt.close(fig)
    meta=json.loads((out/'embedding_manifest.json').read_text())
    coverage=pd.read_csv(out/'coverage.csv');coverage=coverage[coverage.split.eq('train')]
    probes=pd.read_csv(out/'prior_probe_audit.csv',keep_default_na=False)
    special=probes[probes.pair_id.isin(['P05','P16'])][['pair_id','a_id','b_id','category_flag_differences','different_claim_blocks']]
    p=s[s.seed.eq(primary)]
    stability_index=stability.set_index('family')
    claim_primary=claims[claims.seed.eq(primary)].iloc[0]
    old_m=pd.read_csv(run.previous.OUT/'fold_metrics.csv')
    old_policy=old_m[old_m.family.eq('selection_procedure')].score.mean()
    new_policy=float(p.loc[p.family.eq('selection_procedure'),'H'].iloc[0])
    text=f'''# Kiểm soát độ nhạy và biểu diễn mệnh đề đóng góp — kết quả

## Kết quả chính

Quy trình chọn arm/ngưỡng bằng inner CV đạt **H={new_policy:.5f}** ở seed chính; mốc reference cố định đạt **{float(p.loc[p.family.eq('reference'),'H'].iloc[0]):.5f}**. Quy trình cũ tự chọn nhiều nhánh đạt {old_policy:.5f}; đây là một mốc lịch sử khác với reference cố định. Không dùng điểm của arm thắng outer để thay điểm toàn quy trình.

{table(p[['family','H','Q','R','FP','FN','missed_high','AUC','AUC_gap','logloss']])}

- Gói regularization depth3/L2=32 thay cho depth4/L2=8: delta H **{reg.delta_H:+.5f}**, delta AUC gap **{reg.delta_auc_gap:+.5f}**, delta validation AUC **{reg.delta_AUC:+.5f}**. Giảm gap chỉ có ý nghĩa khi xem cùng hiệu năng validation; không tự coi gap nhỏ là model tốt hơn.
- Câu mệnh đề so với câu đầu abstract, cùng encoder/max budget/head: delta H seed chính **{claim_primary.delta_H:+.5f}**, tốt hơn ở **{int(claim_primary.improved_folds)}/5 folds**. Các seed phụ được báo đầy đủ phía dưới; không chọn seed cao điểm nhất.
- Diễn giải quyết định: **{interpretation_vi}** Không áp dụng lại ngưỡng tăng 0.02 H tự đặt như điều kiện bắt buộc của assignment.

## Đối chứng và độ nhạy

{table(effects[['seed','arm','reference','delta_H','improved_folds','delta_AUC','delta_auc_gap','delta_FP','delta_FN','delta_high_FN']])}

Seed20261018 là seed chính: chọn cutoff0..0.90 bước0.05 bằng bốn inner folds cho từng arm và chọn arm chung. Seeds20261019/20261020 chỉ fit lại outer heads, giữ nguyên cutoff và lựa chọn arm của seed chính. Chúng đo độ nhạy khởi tạo/head, không phải ba CV độc lập hoặc ba lần tune. Input/history/projection giữ nguyên. Không chọn seed tốt nhất để fit model cuối.

{table(stability)}

Biên độ H qua ba head seeds: reference **{stability_index.loc['reference','H_range']:.5f}**, regularized **{stability_index.loc['regularized','H_range']:.5f}**. Độ lệch chuẩn H giữa năm folds ở seed chính lần lượt **{float(p.loc[p.family.eq('reference'),'H_fold_sd'].iloc[0]):.5f}** và **{float(p.loc[p.family.eq('regularized'),'H_fold_sd'].iloc[0]):.5f}**. Đây là hai mô tả độ nhạy khác nhau, không phải một phép phân rã phương sai. Không được coi giảm train–validation gap là đã giảm mọi loại độ nhạy, hoặc chọn seed mới để chữa lỗi.

![Độ nhạy qua các seed](outputs/stability_comparison.png)

[PDF để xuất](outputs/stability_comparison.pdf). Fold H thay đổi giữa các seed và các folds là mô tả, không phải khoảng tin cậy độc lập. Dữ liệu outer đã được xem trong các vòng nghiên cứu trước; kết quả exploratory chưa chứng minh tăng điểm Private Kaggle.

## Đã thay đổi gì trong biểu diễn

Năm arm cố định: reference, regularized cùng feature cũ; repaired sửa education và tutorial; prefix thêm ba khối câu đầu abstract; claims thêm ba khối ứng viên problem/method/result. Cả hai text arm cùng frozen MiniLM revision `{meta['revision']}`, mean pooling/L2, max128 tokens mỗi khối, SVD16 fit riêng trong training partition và một presence indicator mỗi khối. Tổng51 feature mới ở cả hai arm. Không fine-tune encoder và không sinh văn bản/nhãn sở thích.

SVD không học từ validation/test. Các features lịch sử cũ vẫn được cross-fit trong fit partition tương ứng, giữ nguyên thứ tự cột cũ. Câu được chọn bằng cues tổng quát; vai trò là ứng viên, không phải phân tích đóng góp ngữ nghĩa đã được xác minh. Không fallback sang toàn abstract khi không có cue; trường thiếu giữ vector0 và presence0.

{table(coverage[['kind','role','n','available']])}

{table(pd.DataFrame(meta['statistics']))}

Prefix và claims có cùng số khối và trần token, nhưng không có cùng số từ thực tế. Claims có nhiều khối unknown; prefix lấy sáu câu đầu chia ba khối. Vì vậy chênh lệch không chỉ đo “hiểu vai trò ngữ nghĩa”, mà có thể gồm ảnh hưởng của lượng/vị trí văn bản và coverage. Đây là đối chứng mạnh hơn việc chỉ thêm một encoder mới, nhưng chưa tách hết các yếu tố.

Các câu và offsets ở [sentence_evidence.csv](outputs/sentence_evidence.csv); văn bản model nhận ở [extracted_texts.csv](outputs/extracted_texts.csv). Sửa education chỉ đòi hỏi từ ngữ giáo dục rõ thay standalone student; tutorial là literal document-type flag. Các sửa chữa là label-independent, không bảo đảm làm tăng H. [repair_ledger.csv](outputs/repair_ledger.csv) lưu thay đổi.

## Những failure còn lại

So với quy trình tự chọn cũ: sửa **{int((before&~after).sum())}** quyết định, phát sinh **{int((~before&after).sum())}** lỗi mới; tổng lỗi nhị phân **{int(before.sum())} → {int(after.sum())}**. H còn phụ thuộc mức nhãn/thứ tự nên tổng lỗi không thay thế metric chính.

{table(known[['id','title','Label','old_status','new_status','claims_fields_present']])}

Trong input hiện có, ID1045,1000,2383 không có mệnh đề nào khớp cues nên cả ba claim fields đều unknown; prefix vẫn có thể nhận phần văn bản đầu abstract. Đây là giới hạn cụ thể của extractor theo cách diễn đạt, không phải bằng chứng rằng các bài thiếu abstract hoặc không có đóng góp. Các số này phải được xem cùng đối chứng prefix trước khi quy kết hiệu quả cho semantic role extraction.

{table(coverage_labels)}

Hai cặp từng có cùng category flags giờ có thể nhận các đoạn câu khác nhau:

{table(special)}

Đây là kiểm tra input representation, chưa chứng minh model hiểu được hay dùng đúng chi tiết đó. Các cặp20 bài đã giúp thiết kế extractor nên là diagnostic probes, không phải tập kiểm tra chất lượng extraction chưa từng thấy. Không gán lại preference hay ghép quyết định theo ID. Mỗi arm/seed có priority riêng trong `oof_predictions.csv`; chuyển đổi từng lỗi trong `failure_transitions.csv`.

## Model cuối và tái lập

Chọn lại arm/cutoff trên bốn inner folds của toàn train, cùng membership đã dùng trước, rồi fit toàn2494 labels bằng seed chính:

```json
{json.dumps(chosen,indent=2)}
```

Ứng viên [submission_candidate.csv](outputs/submission_candidate.csv) đủ596 ID, {check['positive_priority']} priority dương. Classifier và recipe ở `final_classifier.cbm`, `final_recipe.joblib`; bản trước không bị thay thế, chưa gửi Kaggle. Lựa chọn toàn train có thể khác arm có outer H cao nhất; không chọn cuối bằng outer labels.

Kiểm tra SHA-256 cho thấy CSV cuối **{'trùng hoàn toàn với ứng viên trước' if same_candidate else 'khác ứng viên trước'}**. Đây là kết quả chọn lại bằng inner CV, không phải loại model chỉ vì không vượt một ngưỡng tăng điểm tự đặt.

Đã qua5 unit tests, kiểm tra {check['verbatim_evidence_spans_verified']} evidence spans, {check['inherited_grouped_crossfits_verified']} inherited crossfits và {check['fit_only_svd_projections_checked']} projection blocks; tính lại {check['outer_metrics_recomputed']} outer metric rows và tất cả inner choices. Dựng lại feature sau khi tải model cho cùng selection và toàn bộ thứ tự submission.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/claim_stability/run_plan.py
```

Xem [PLAN.md](PLAN.md), [outputs/verification.json](outputs/verification.json), [outputs/paired_effects.csv](outputs/paired_effects.csv), [outputs/decision.json](outputs/decision.json). Classification là phương pháp môn học; các embedding chỉ là biểu diễn đầu vào.
'''
    (run.HERE/'RESULTS.md').write_text(text)
    print(table(p[['family','H','R','FP','FN','missed_high','AUC_gap']]))
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
