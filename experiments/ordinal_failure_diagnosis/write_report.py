#!/usr/bin/env python3
"""Evidence-ranked explanation; keep accounting distinct from intervention effects."""
import json
import numpy as np
import pandas as pd

import run_controls as diag
import defaults_control as dc
base=diag.base


def summarize(frame):
    return frame.groupby('family',sort=False).agg(H=('score','mean'),Q=('relevance','mean'),R=('coverage','mean'),
        AUC=('AUC','mean'),train_AUC=('train_AUC','mean'),AUC_gap=('AUC_gap','mean'),
        FP=('FP','sum'),FN=('FN','sum'),high_FN=('missed_high','sum'))


def main():
    out=diag.OUT
    controls=pd.read_csv(out/'control_fold_metrics.csv');default=pd.read_csv(dc.OUT/'fold_metrics.csv')
    summary=summarize(controls);defaults=summarize(default)
    summary.to_csv(out/'control_summary.csv');defaults.to_csv(dc.OUT/'summary.csv')
    mechanisms=json.loads((out/'mechanisms/summary.json').read_text())
    verification=json.loads((out/'verification.json').read_text())
    diffs=json.loads((out/'mechanisms/default_parameter_differences.json').read_text())
    simple=[]
    for key in ['bootstrap_type','subsample','bagging_temperature','leaf_estimation_iterations']:
        simple.append(dict(parameter=key,**diffs[key]))
    simple=pd.DataFrame(simple)
    raw=summary.loc['ORD_full','H'];mc=summary.loc['MC_full','H']
    effects=[]
    for family,reference in [('ORD_regularized','ORD_full'),('MC_regularized','MC_full'),
                             ('ORD_without_neighbors','ORD_full'),('MC_without_neighbors','MC_full')]:
        a=controls[controls.family.eq(family)].set_index('outer_fold');b=controls[controls.family.eq(reference)].set_index('outer_fold')
        for k in a.index:
            effects.append(dict(fold=int(k),family=family,reference=reference,
                delta_H=float(a.loc[k,'score']-b.loc[k,'score']),delta_AUC=float(a.loc[k,'AUC']-b.loc[k,'AUC']),
                delta_FP=int(a.loc[k,'FP']-b.loc[k,'FP']),delta_FN=int(a.loc[k,'FN']-b.loc[k,'FN'])))
    pd.DataFrame(effects).to_csv(out/'intervention_paired_effects.csv',index=False)
    loss=pd.read_csv(out/'mechanisms/individual_logloss.csv',dtype={'id':str})
    a=loss[loss.family.eq('ordinal')].set_index('id');b=loss[loss.family.eq('multiclass')].set_index('id').loc[a.index]
    zero=a.zero_mass.to_numpy(dtype=bool);lossdelta=a.logloss.to_numpy()-b.logloss.to_numpy()
    nll=dict(pooled_delta=float(lossdelta.mean()),zero_mass_difference_contribution=float(lossdelta[zero].sum()/len(a)),
             other_rows_difference_contribution=float(lossdelta[~zero].sum()/len(a)))
    (out/'mechanisms/logloss_attribution.json').write_text(json.dumps(nll,indent=2)+'\n')
    group_interaction=(summary.loc['ORD_without_neighbors','H']-raw)-(summary.loc['MC_without_neighbors','H']-mc)
    reg_interaction=(summary.loc['ORD_regularized','H']-raw)-(summary.loc['MC_regularized','H']-mc)
    facts=dict(gate_H_contribution=mechanisms['mean_H_decomposition']['gate_effect'],
        ranking_H_contribution=mechanisms['mean_H_decomposition']['rank_effect'],projection_H_effect=0.,
        ordinal_H=raw,incumbent_H=mc,ordinal_regularized_H=summary.loc['ORD_regularized','H'],
        ordinal_without_neighbors_H=summary.loc['ORD_without_neighbors','H'],ordinal_matched_defaults_H=defaults.loc['ordinal','H'],
        feature_group_H_interaction=float(group_interaction),regularization_H_interaction=float(reg_interaction),
        gains_are_not_additive=True,preprocessing_corruption_demonstrated=False,
        semantic_feature_bottleneck='plausible, not proven by these controls',
        retain_incumbent_main=True,uploaded=False)
    (out/'conclusions.json').write_text(json.dumps(facts,indent=2)+'\n')
    fchoices=json.loads((out/'final_choices.json').read_text())
    dchoices=json.loads((dc.OUT/'final_choices.json').read_text())
    text=f'''# Vì sao ordinal kém hơn — chẩn đoán bằng đối chứng

**Phần tụt H tập trung ở chọn/bỏ bài; có ảnh hưởng thực nghiệm từ training defaults
và tương tác giữa ordinal với nhóm feature láng giềng. Chưa có bằng chứng cleaning
text/đọc dữ liệu bị lỗi. PAV không làm tụt H trong lượt này.** Giữ incumbent làm
bản chính; các sửa đổi mới vẫn là thí nghiệm exploratory.

## 1. Khoanh chính xác phần H bị mất

Đổi chéo read gate và gain ranker từ hai model, giữ nguyên cutoff đã chọn bằng
inner CV. Shapley accounting trung bình5folds:

- Read gate ordinal đóng góp **{mechanisms['mean_H_decomposition']['gate_effect']:+.5f} H**.
- Gain ranker ordinal đóng góp **{mechanisms['mean_H_decomposition']['rank_effect']:+.5f} H**.
- Tổng: **{raw-mc:+.5f} H**.

Đây là phân rã kết quả, chưa phải nguyên nhân gốc. Nếu giữ read gate multiclass
và thay bằng ordinal ranker thì H**0.64118**, recall giữ nguyên; chỉ là diagnostic
crossed-component trên outer đã xem, không được xem là điểm độc lập hoặc chọn model
cuối bằng outer labels.

PAV có164dòng crossing, nhưng **0/2494 quyết định chọn bài thay đổi**; gain giữ
nguyên. Bỏ PAV khỏi read gate giữ H**0.63535**. Giữ cutoff multiclass ở mọi fold
đưa ordinal lên**0.63628**. Oracle cutoff (dùng nhãn thật chỉ để chẩn đoán) đạt
**0.64228**, không là policy triển khai hoặc mức tăng có thể kỳ vọng.

## 2. Điểm bỏ sót trong thiết kế đối chứng gốc: defaults theo loss

Các tham số explicit giống nhau chưa khóa mọi mặc định của CatBoost. Lấy trực
tiếp `get_all_params()` từ saved models:

{simple.to_markdown(index=False)}

CTR target border count cũng đổi4→1 theo5-class/binary target; không ép4borders
cho binary. Cấu trúc cây cũng được học riêng cho4tasks thay vì chia sẻ splits giữa
5classes. Vì vậy không thể quy mọi chênh lệch cho tên loss hoặc chất lượng feature.

Đối chứng mới giữ277features/depth4/L2=8, chỉ khóa Bayesian/bagging1 và leaf updates1:

{defaults.reset_index().to_markdown(index=False,floatfmt='.5f')}

Ordinal H **{raw:.5f}→{defaults.loc['ordinal','H']:.5f}**; train AUC
**{summary.loc['ORD_full','train_AUC']:.5f}→{defaults.loc['ordinal','train_AUC']:.5f}**,
validation AUC **{summary.loc['ORD_full','AUC']:.5f}→{defaults.loc['ordinal','AUC']:.5f}**.
Gói mặc định ảnh hưởng rõ độ fit trên train và sửa một phần H, không tạo ra cải
thiện tương ứng về khả năng xếp hạng đọc/không đọc trên validation. Chưa tách riêng
bootstrap với leaf updates, cũng chưa loại hết khác biệt giữa hai formulation.

## 3. Feature và độ phức tạp: hai can thiệp giữ các phần khác cố định

`regularized`:depth3/L2=32 thaydepth4/L2=8; `without_neighbors`:bỏ150neighbor
fields, giữ35base +92content. `ridge_rating` vẫn còn và vẫn cross-fit. Mọi threshold
chọn trên4inner; score dưới là mean5outer. MC control dùng raw caches matched.

{summary.reset_index().to_markdown(index=False,floatfmt='.5f')}

- Regularization sửa ordinal **{summary.loc['ORD_regularized','H']-raw:+.5f} H**,
  train AUC giảm{summary.loc['ORD_full','train_AUC']:.5f}→{summary.loc['ORD_regularized','train_AUC']:.5f}.
  Giảm train–validation gap chưa tự chứng minh nguyên nhân duy nhất hoặc model tốt.
- Bỏ neighbor sửa ordinal **{summary.loc['ORD_without_neighbors','H']-raw:+.5f} H**,
  trong khi multiclass đổi **{summary.loc['MC_without_neighbors','H']-mc:+.5f} H**.
  Interaction của phép bỏ nhóm feature giữa hai formulation là **{group_interaction:+.5f} H**.
  Đây là bằng chứng về tương tác feature×model ở cấu hình này, không phải nhóm
  feature đó vô ích cho mọi model hoặc toàn bộ preprocessing sai.
- Các mức tăng thuộc những can thiệp riêng, **không cộng chúng lại** để dự đoán
  hiệu quả của cấu hình kết hợp chưa được chạy.

## 4. Preprocessing: điều gì đã kiểm tra, điều gì chưa biết

Hai model gốc nhận **cùng các file feature đã hash**, cùng column order và cùng
folds. Audit không có numericNaN/inf ({mechanisms['numeric_nonfinite_count']} entries).
Nhãn nguồn, grouping, inherited crossfits, rebuild test features và model reload
đều được kiểm tra. Điều này chưa chứng minh representation đủ thông tin ngữ nghĩa.

Classifier nhận title TF-IDF SVD32 +venue/year +Ridge score,92lexical content
flags và150neighbor statistics. Abstract đi vào qua flags/similarity-derived
statistics; đây không phải một encoder abstract được học trực tiếp theo sở thích.
SVD giữ khoảng **{100*mechanisms['mean_title_svd_TFIDF_variance']:.1f}% TF-IDF variance**;
con số đó **không** là phần trăm thông tin sở thích được giữ hoặc bị mất.

History train rows dùng reference subsets của crossfit, validation rows dùng toàn
outer train. Có drift về references/support trong `history_size_drift.csv`; training
importance của nhóm count không chứng minh nó là cột gây hại. Phép bỏ toàn bộ
neighbor mới kiểm tra hiệu ứng cả nhóm, chưa tách reference-size drift khỏi noise,
redundancy hay tác dụng regularization do bỏ150fields.

## 5. Vì sao logloss tăng mạnh mà PAV không gây mất H?

PAV tạo xác suất lớp thật bằng0 ở **16**bài (8label2,5label3,3label4).
Những dòng này đóng góp khoảng **{nll['zero_mass_difference_contribution']:.5f}** vào
pooled Δlogloss **{nll['pooled_delta']:.5f}**. Đây là phân rã pooled per-item,
khác mean5folds một chút. Logloss phạt rất mạnh mass0; H chỉ dùng selection/order,
mà hai phần đó không đổi do PAV ở lượt này. Không dùng logloss để quy kết lỗi H.

## 6. Hướng sửa có cơ sở và giới hạn

Khóa rõ training defaults khi so objective; giữ riêng read gate và gain head trong
diagnostics. Với ordinal, dùng strong regularization/feature masks là các ứng viên
có cơ sở thử tiếp, cần kiểm tra cấu hình kết hợp riêng thay vì cộng các gains.
Nếu xử lý history, thử bỏ/chuẩn hóa reference-count fields hoặc làm reference sizes
train/predict tương đương để tách drift khỏi giá trị neighbor labels. Đây là giả
thuyết tiếp theo, chưa được chứng minh bởi current group ablation.

Muốn kiểm tra thiếu ngữ nghĩa, cần đối chứng title-only với title+abstract trong
supervised encoder; không thể kết luận từ việc ordinal đổi head là feature đã đủ
hoặc abstract không hữu ích. Các sửa hiện tại vẫn quanh0.64, chưa có bằng chứng
đạt0.7. Chưa biết cơ chế thật của người gán nhãn và không tự quy label conflicts
thành label noise.

## Final artifacts và xác minh

Control policy final:
```json
{json.dumps(fchoices['selection_procedure'],indent=2)}
```
Default-matched policy final:
```json
{json.dumps(dchoices['selection_procedure'],indent=2)}
```

Mỗi candidate theo inner selection của chính policy, không lấy family thắng outer
để tạo model cuối. Policy scores báo riêng trong hai bảng trên. Caches/models/CSV
đều ở `outputs`, default-matched ở `outputs/defaults`. Bản incumbent giữ nguyên;
chưa gửi Kaggle. Dataset outer được nghiên cứu lặp lại nên đây là bằng chứng
exploratory ở mức can thiệp cấu hình, không là kiểm định nhân quả độc lập.

Verify **{verification['status']}**:360new binary fits,50outer metric rows,60inner
choices,15final model reloads, rebuilt features, cùng priority/rank và nhãn nguồn.
Xem `outputs/verification.json`, `control_summary.csv`, `intervention_paired_effects.csv`,
`mechanisms/default_parameter_differences.json`, `mechanisms/H_decomposition.csv`.

Tham khảo: [CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning).
'''
    (diag.HERE/'RESULTS.md').write_text(text)
    old_report=base.HERE/'RESULTS.md';marker='\n## Follow-up: loss-dependent defaults và feature interaction\n'
    previous=old_report.read_text().split(marker)[0]
    old_report.write_text(previous+marker+'\nĐối chứng gốc giữ cùng explicit hyperparameters, nhưng saved models cho thấy\n'
        'CatBoost thay bootstrap và leaf_estimation_iterations theo loss. Kết quả\n'
        'gốc vẫn là so sánh hai pipeline cụ thể; không isolate riêng label formulation.\n'
        'Các can thiệp và nguyên nhân đã được kiểm tra trong\n'
        '[báo cáo chẩn đoán](../ordinal_failure_diagnosis/RESULTS.md).\n')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':main()
