# Cumulative ordinal classification — kết quả

**Đối chứng hoàn tất; giữ nguyên artifact incumbent, lưu ứng viên ordinal riêng để đối chiếu.**

Giữ đúng277features,5outer/4inner groups,seed20261018 và policy chọn/xếp hạng.
Thay một CatBoost5lớp bằng4binary CatBoost cho y>=2/3/4/5, mỗi model300trees,
depth4,lr0.05,L2=8. Total capacity/compute khác với multiclass; đây là đối chứng
toàn formulation. Chưa thêm calibration hoặc đổi encoder.

| family              |       H |   H_std |       Q |       R |   FP |   FN |   missed_high |     AUC |   logloss |   gain_MAE |   train_AUC |   AUC_gap |
|:--------------------|--------:|--------:|--------:|--------:|-----:|-----:|--------------:|--------:|----------:|-----------:|------------:|----------:|
| multiclass          | 0.64028 | 0.01120 | 0.53250 | 0.80324 |  338 |  313 |            38 | 0.79946 |   1.28296 |    0.21904 |     0.87416 |   0.07470 |
| ordinal             | 0.63535 | 0.01547 | 0.53260 | 0.78886 |  340 |  337 |            49 | 0.79874 |   1.54373 |    0.21447 |     0.96438 |   0.16565 |
| selection_procedure | 0.64028 | 0.01120 | 0.53250 | 0.80324 |  338 |  313 |            38 | 0.79946 |   1.28296 |    0.21904 |     0.87416 |   0.07470 |

Ordinal − multiclass: **-0.00493 H**, tốt hơn **1/5folds**.
Điểm toàn quy trình chọn family bằng inner báo riêng ở `selection_procedure`.
H tính từng fold rồi lấy trung bình; không tính harmonic mean của hai cột Q/R đã
được trung bình. Các outer folds đã được xem ở nhiều nghiên cứu trước.

## Paired effects

|   outer_fold | family              |   delta_score |   delta_relevance |   delta_coverage |   delta_FP |   delta_FN |   delta_missed_high |   delta_AUC |   delta_gain_MAE |
|-------------:|:--------------------|--------------:|------------------:|-----------------:|-----------:|-----------:|--------------------:|------------:|-----------------:|
|            0 | ordinal             |       0.00561 |           0.00114 |          0.01623 |    1.00000 |   -5.00000 |             1.00000 |     0.00508 |         -0.00573 |
|            1 | ordinal             |      -0.01664 |          -0.00472 |         -0.04012 |    0.00000 |   13.00000 |             6.00000 |    -0.00131 |         -0.00246 |
|            2 | ordinal             |      -0.00262 |          -0.00795 |          0.00974 |    6.00000 |   -3.00000 |            -1.00000 |    -0.00717 |         -0.00504 |
|            3 | ordinal             |      -0.00285 |           0.01772 |         -0.04518 |   -8.00000 |   15.00000 |             3.00000 |     0.00860 |         -0.00479 |
|            4 | ordinal             |      -0.00814 |          -0.00568 |         -0.01254 |    3.00000 |    4.00000 |             2.00000 |    -0.00880 |         -0.00486 |
|            0 | selection_procedure |       0.00000 |           0.00000 |          0.00000 |    0.00000 |    0.00000 |             0.00000 |     0.00000 |          0.00000 |
|            1 | selection_procedure |       0.00000 |           0.00000 |          0.00000 |    0.00000 |    0.00000 |             0.00000 |     0.00000 |          0.00000 |
|            2 | selection_procedure |       0.00000 |           0.00000 |          0.00000 |    0.00000 |    0.00000 |             0.00000 |     0.00000 |          0.00000 |
|            3 | selection_procedure |       0.00000 |           0.00000 |          0.00000 |    0.00000 |    0.00000 |             0.00000 |     0.00000 |          0.00000 |
|            4 | selection_procedure |       0.00000 |           0.00000 |          0.00000 |    0.00000 |    0.00000 |             0.00000 |     0.00000 |          0.00000 |

## Tính nhất quán xác suất

q2=P(y>=2),q3=P(y>=3),q4=P(y>=4),q5=P(y>=5). P(read)=q2;
expected gain=(q2+q3+q4+q5)/4. PAV sửa4giá trị mỗi bài thành dãy không tăng,
không dùng nhãn, giữ tổng/gain. Khi pooled tails bằng nhau, một số xác suất lớp
giữa có thể bằng0; multiclass logloss có thể phạt mạnh dù H/gain vẫn hợp lệ.

Trên2494outer predictions: **164** dòng có crossing
trước projection. Toàn bộ corrections ở `outputs/projection_diagnostics.csv`.

| family     |   target | stage   |     AUC |   Brier |
|:-----------|---------:|:--------|--------:|--------:|
| multiclass |        2 | raw     | 0.79946 | 0.17201 |
| multiclass |        2 | tails   | 0.79946 | 0.17201 |
| multiclass |        3 | raw     | 0.80787 | 0.17734 |
| multiclass |        3 | tails   | 0.80787 | 0.17734 |
| multiclass |        4 | raw     | 0.84926 | 0.12965 |
| multiclass |        4 | tails   | 0.84926 | 0.12965 |
| multiclass |        5 | raw     | 0.87933 | 0.07293 |
| multiclass |        5 | tails   | 0.87933 | 0.07293 |
| ordinal    |        2 | raw     | 0.79871 | 0.17300 |
| ordinal    |        2 | tails   | 0.79874 | 0.17296 |
| ordinal    |        3 | raw     | 0.80471 | 0.17830 |
| ordinal    |        3 | tails   | 0.80512 | 0.17818 |
| ordinal    |        4 | raw     | 0.85034 | 0.12770 |
| ordinal    |        4 | tails   | 0.85038 | 0.12760 |
| ordinal    |        5 | raw     | 0.87760 | 0.07416 |
| ordinal    |        5 | tails   | 0.87760 | 0.07417 |

Các AUC/Brier này dùng để chẩn đoán từng ngưỡng; không thay H để chọn model.

## Failure và chọn cuối

So với multiclass, ordinal sửa **72** quyết định nhị phân,
phát sinh **98** lỗi mới. Chi tiết từng ID ở
`outputs/failure_transitions.csv`, theo venue/label ở `failure_breakdown.csv`.
Các thay đổi không chứng minh nguyên nhân sở thích của người gán nhãn.

```json
{
  "multiclass": {
    "family": "multiclass",
    "threshold": 0.55,
    "inner_score": 0.6386873823729493
  },
  "ordinal": {
    "family": "ordinal",
    "threshold": 0.55,
    "inner_score": 0.6322168077005849
  },
  "selection_procedure": {
    "family": "multiclass",
    "threshold": 0.55,
    "inner_score": 0.6386873823729493
  }
}
```

`outputs/submission_ordinal.csv`: đủ596ID, **364** priority
dương. `outputs/submission_candidate.csv`: chọn theo final inner policy, **356**
priority dương. Bản cũ không bị ghi đè; chưa gửi Kaggle. Bốn model lưu ở
`outputs/ordinal_ge2.cbm` tới `ordinal_ge5.cbm`, recipe ở `final_recipe.joblib`.

## Xác minh và giới hạn

4unit tests,120binary fits được audit,90inherited group crossfits,15outer metric
rows và80tail metric rows tính lại,18inner choices xác minh. PAV được tính lại
bằng solver isotonic độc lập. Rebuild test features từ recipe, reload4models cho
cùng xác suất/priority/toàn ranking; nhãn gốc và CSV incumbent giữ nguyên.
Trạng thái: **PASSED**, chi tiết `outputs/verification.json`.

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

## Follow-up: loss-dependent defaults và feature interaction

Đối chứng gốc giữ cùng explicit hyperparameters, nhưng saved models cho thấy
CatBoost thay bootstrap và leaf_estimation_iterations theo loss. Kết quả
gốc vẫn là so sánh hai pipeline cụ thể; không isolate riêng label formulation.
Các can thiệp và nguyên nhân đã được kiểm tra trong
[báo cáo chẩn đoán](../ordinal_failure_diagnosis/RESULTS.md).
