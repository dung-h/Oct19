# Phối hợp SPECTER2 với CatBoost — kiểm tra bổ sung

Đây là nghiên cứu bổ sung sau khi đã xem một phần outer của primary; không là
đánh giá độc lập. Weight và cutoff ở mỗi outer vẫn chỉ chọn bằng4 inner folds.
Giữ nguyên neural arm/epoch của primary. Không huấn luyện model bổ sung.

```json
{
  "H": 0.6414431569824293,
  "Q": 0.5593710154540499,
  "R": 0.7553968573702682,
  "AUC": 0.812063847648318,
  "logloss": 1.258108032378882,
  "FP": 268,
  "FN": 391,
  "missed_high": 55
}
```

|   outer_fold |   weight |   threshold |   inner_score |   score |   relevance |   coverage |        TP |       FP |       FN |   selected |   missed_high |     AUC |   logloss |
|-------------:|---------:|------------:|--------------:|--------:|------------:|-----------:|----------:|---------:|---------:|-----------:|--------------:|--------:|----------:|
|      0.00000 |  0.50000 |     0.60000 |       0.64564 | 0.65203 |     0.54782 |    0.80519 | 248.00000 | 67.00000 | 60.00000 |  315.00000 |       9.00000 | 0.81765 |   1.26091 |
|      1.00000 |  0.75000 |     0.60000 |       0.64976 | 0.60956 |     0.53764 |    0.70370 | 228.00000 | 53.00000 | 96.00000 |  281.00000 |      20.00000 | 0.78042 |   1.33259 |
|      2.00000 |  0.25000 |     0.55000 |       0.64977 | 0.63341 |     0.53057 |    0.78571 | 242.00000 | 68.00000 | 66.00000 |  310.00000 |       9.00000 | 0.80437 |   1.20725 |
|      3.00000 |  0.50000 |     0.60000 |       0.63858 | 0.65208 |     0.60893 |    0.70181 | 233.00000 | 37.00000 | 99.00000 |  270.00000 |      11.00000 | 0.82499 |   1.27721 |
|      4.00000 |  0.50000 |     0.55000 |       0.64257 | 0.66013 |     0.57189 |    0.78056 | 249.00000 | 43.00000 | 70.00000 |  292.00000 |       6.00000 | 0.83289 |   1.21258 |

|   outer_fold | reference           |   delta_H |   delta_FP |   delta_FN |
|-------------:|:--------------------|----------:|-----------:|-----------:|
|            0 | incumbent           |   0.00862 |        -12 |         10 |
|            1 | incumbent           |  -0.02371 |         -5 |         27 |
|            2 | incumbent           |   0.00809 |         -3 |         -3 |
|            3 | incumbent           |  -0.00219 |        -30 |         39 |
|            4 | incumbent           |   0.01503 |        -20 |          5 |
|            0 | selection_procedure |  -0.00099 |         -9 |         13 |
|            1 | selection_procedure |   0.00912 |         10 |        -15 |
|            2 | selection_procedure |  -0.00304 |         11 |         -7 |
|            3 | selection_procedure |   0.02324 |          7 |        -24 |
|            4 | selection_procedure |   0.01503 |        -20 |          5 |

Final inner selection:

```json
{
  "neural": {
    "family": "adapted",
    "epoch": 8,
    "threshold": 0.45,
    "inner_score": 0.6399621136029068,
    "inner_recall": 0.8296722384534565
  },
  "weight": 0.5,
  "threshold": 0.5,
  "inner_score": 0.6425185692855527
}
```

CSV `outputs/blend/submission_blend_candidate.csv`, đủ596 ID, 393
priority dương. Primary CSV/model cũ giữ riêng; chưa gửi Kaggle. Kiểm tra artifact:
`outputs/blend/verification.json`. Lựa chọn cuối không dùng outer labels; các
điểm local vẫn có giới hạn do dataset được xem lặp lại trong nghiên cứu.
