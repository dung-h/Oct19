# SPECTER2 title + abstract học theo preference: pilot

**Joint adapted was numerically higher than title adapted, but the +H difference is too small and inconsistent to treat as a material improvement.**

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
Title truncated: **0**; joint truncated:
**9**, trần512 tokens. Không dùng ID/Label/metadata trong
token input. Không thêm synthetic preference labels.

## Kết quả

| policy              |       H |   H_std |       Q |       R |     AUC |   train_AUC |   logloss |   gain_MAE |        FP |        FN |   missed_high |   AUC_gap |
|:--------------------|--------:|--------:|--------:|--------:|--------:|------------:|----------:|-----------:|----------:|----------:|--------------:|----------:|
| incumbent           | 0.63768 | 0.01071 | 0.51792 | 0.83147 | 0.79946 |     0.87416 |   1.28296 |    0.21904 | 389.00000 | 269.00000 |      33.00000 |   0.07470 |
| title_adapted       | 0.63322 | 0.01648 | 0.54225 | 0.76793 | 0.80802 |     0.89291 |   1.32902 |    0.20936 | 311.00000 | 371.00000 |      54.00000 |   0.08489 |
| joint_frozen        | 0.57174 | 0.03907 | 0.52025 | 0.67373 | 0.74010 |     0.75856 |   1.41200 |    0.25762 | 325.00000 | 524.00000 |     119.00000 |   0.01846 |
| joint_adapted       | 0.63375 | 0.01593 | 0.52259 | 0.81142 | 0.79891 |     0.90938 |   1.33024 |    0.21142 | 379.00000 | 302.00000 |      42.00000 |   0.11047 |
| neural_selection    | 0.63335 | 0.01973 | 0.52834 | 0.79807 | 0.80320 |     0.90622 |   1.32682 |    0.20872 | 362.00000 | 323.00000 |      45.00000 |   0.10302 |
| selection_procedure | 0.63344 | 0.01777 | 0.52443 | 0.80586 | 0.79826 |     0.89400 |   1.31385 |    0.21355 | 368.00000 | 311.00000 |      45.00000 |   0.09575 |

Mean H được tính từ từng outer fold, không harmonic mean của Q/R đã trung bình.
FP/FN/highFN là tổng2494OOF. Incumbent cutoff cũng chọn lại trên chính calibration
của pilot; vì thế incumbent pilot score khác score4inner trước và không được
trộn hai validation designs để báo một delta model thuần túy.

| arm                 | reference     |   delta_H |   improved_folds |   delta_Q |   delta_R |   delta_AUC |   delta_FP |   delta_FN |   delta_missed_high |
|:--------------------|:--------------|----------:|-----------------:|----------:|----------:|------------:|-----------:|-----------:|--------------------:|
| joint_adapted       | title_adapted |   0.00054 |                3 |  -0.01966 |   0.04349 |    -0.00911 |         68 |        -69 |                 -12 |
| joint_adapted       | joint_frozen  |   0.06201 |                5 |   0.00235 |   0.13769 |     0.05881 |         54 |       -222 |                 -77 |
| joint_adapted       | incumbent     |  -0.00393 |                2 |   0.00467 |  -0.02006 |    -0.00055 |        -10 |         33 |                   9 |
| selection_procedure | incumbent     |  -0.00424 |                2 |   0.00650 |  -0.02561 |    -0.00120 |        -21 |         42 |                  12 |

Joint adapted so với title adapted: Q -0.01966, recall +0.04349, read AUC -0.00911; FP +68, FN -69, bỏ sót nhãn4/5 -12. AUC train/outer: title 0.89291/0.80802, joint 0.90938/0.79891, joint frozen 0.75856/0.74010.

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

|                                      |   delta_H |   gate_contribution |   rank_contribution |
|:-------------------------------------|----------:|--------------------:|--------------------:|
| ('joint_adapted', 'title_adapted')   |   0.00054 |            -0.00094 |             0.00147 |
| ('joint_adapted', 'joint_frozen')    |   0.06201 |             0.04730 |             0.01471 |
| ('joint_adapted', 'incumbent')       |  -0.00393 |            -0.00520 |             0.00127 |
| ('selection_procedure', 'incumbent') |  -0.00424 |            -0.00442 |             0.00018 |

Hai contributions là trung bình hai thứ tự thay component và cộng đúng deltaH.
Đây là phân rã metric hậu nghiệm để khoanh failure, không chứng minh nguyên nhân
gốc hoặc là candidate hybrid được tune/triển khai bằng outer labels.

|   outer_fold | policy        |   epoch |   threshold |   calibration_H |
|-------------:|:--------------|--------:|------------:|----------------:|
|            0 | title_adapted |       8 |     0.65000 |         0.61750 |
|            0 | joint_frozen  |       8 |     0.60000 |         0.58547 |
|            0 | joint_adapted |       8 |     0.45000 |         0.61279 |
|            1 | title_adapted |       8 |     0.55000 |         0.65042 |
|            1 | joint_frozen  |       8 |     0.55000 |         0.60314 |
|            1 | joint_adapted |       8 |     0.45000 |         0.64639 |
|            2 | title_adapted |       4 |     0.55000 |         0.63626 |
|            2 | joint_frozen  |       8 |     0.50000 |         0.60559 |
|            2 | joint_adapted |       8 |     0.25000 |         0.63628 |
|            3 | title_adapted |       8 |     0.50000 |         0.62117 |
|            3 | joint_frozen  |       8 |     0.60000 |         0.60772 |
|            3 | joint_adapted |       8 |     0.45000 |         0.62546 |
|            4 | title_adapted |       8 |     0.35000 |         0.65338 |
|            4 | joint_frozen  |       8 |     0.50000 |         0.61961 |
|            4 | joint_adapted |       8 |     0.40000 |         0.64265 |

Raw loss/time/epoch ở `outputs/learning_traces.csv`, input text/token count ở
`input_audit.csv`, failures từng ID ở `failure_transitions.csv`, theo venue/label/
abstract available/truncation ở `failure_breakdown.csv`. Các ca đã xem chỉ là
diagnostic probes; không sửa prediction/weight theo ID hoặc outer labels.

## Final và kiểm chứng

Final inherited inner_fold0 chọn:

```json
{
  "choice": {
    "family": "title_adapted",
    "epoch": 8,
    "threshold": 0.5,
    "calibration_H": 0.6616144143554086,
    "calibration_R": 0.8258706467661692
  },
  "model_dir": "final_title_adapted_model",
  "protocol_sha256": "a98967fa181ec362c5d89eba4c7a44d74e40b5b572bc4ee3cbb4336a10585efe",
  "evaluation": "pilot inner0 calibration; exploratory"
}
```

Refit toàn2494, các model cuối riêng được lưu cho cả ba nhánh, CSV theo template596
IDs. Candidate ở `outputs/submission_candidate.csv`; incumbent gốc giữ nguyên.
Chưa upload Kaggle. Không có lời hứa H>0.7 từ kiến trúc hoặc embedding dimension.

Independent verifier **PASSED**: 36
prediction caches, 30 outer metric rows,
36 calibration choices và
36 gradient/drift records. Encode lại
128 frozen vector samples; initial head
đồng nhất, nhãn nguồn giữ nguyên, tokenizer không đọc Label/ID; reload cả
3 final models cho cùng probabilities. Có
2 publication groups giao train/test
trong source dataset; outer fit/validation groups tách biệt. Test labels không được dùng.

Xem [PLAN.md](PLAN.md), [verification.json](outputs/verification.json),
[paired effects](outputs/paired_effects.csv), [decision.json](outputs/decision.json).
