# Giữ gate CatBoost, thay riêng ranking bằng title SPECTER2

**Weak/mixed: candidate được lưu để tái tạo, nhưng incumbent vẫn là bản chính.**

Mỗi nhánh dùng đúng gate incumbent nên selected IDs, TP/FP/FN, coverage và missed-high phải giống nhau. Khác biệt H chỉ đến từ thứ tự của các bài đã chọn. Weight title được chọn bằng mean H của bốn inner folds, không chọn theo outer score. Đây là thí nghiệm cache-only, không train thêm.

## Kết quả outer 5 folds

| policy              |       H |   H_std |       Q |       R |        FP |        FN |   selected |   missed_high |
|:--------------------|--------:|--------:|--------:|--------:|----------:|----------:|-----------:|--------------:|
| incumbent_rank      | 0.64028 | 0.01120 | 0.53250 | 0.80324 | 338.00000 | 313.00000 | 1616.00000 |      38.00000 |
| title_rank          | 0.64144 | 0.01244 | 0.53403 | 0.80324 | 338.00000 | 313.00000 | 1616.00000 |      38.00000 |
| half_rank           | 0.64227 | 0.01145 | 0.53522 | 0.80324 | 338.00000 | 313.00000 | 1616.00000 |      38.00000 |
| inner_selected_rank | 0.64165 | 0.01147 | 0.53437 | 0.80324 | 338.00000 | 313.00000 | 1616.00000 |      38.00000 |

| policy              |   delta_H |   delta_Q |   delta_R |   improved_folds |
|:--------------------|----------:|----------:|----------:|-----------------:|
| title_rank          |   0.00116 |   0.00153 |   0.00000 |                2 |
| half_rank           |   0.00200 |   0.00272 |   0.00000 |                5 |
| inner_selected_rank |   0.00137 |   0.00187 |   0.00000 |                5 |

`incumbent_rank` tái tạo metric incumbent nguồn. Các nhánh còn lại giữ cùng R/FP/FN/missed-high; Q và H phản ánh riêng đổi ranking. H là mean của năm fold H.

## Final artifact

```json
{
  "choice": {
    "incumbent": {
      "family": "incumbent",
      "epoch": 0,
      "threshold": 0.55,
      "inner_score": 0.6386873823729493
    },
    "adapted": {
      "family": "adapted",
      "epoch": 8,
      "threshold": 0.45,
      "inner_score": 0.6399621136029068
    },
    "rank_policy": {
      "weight": 0.5,
      "inner_score": 0.6401437000120378
    }
  },
  "protocol_sha256": "aa01ccfff90e4f614d3bda8b15283bb0298ede4fc8f736aa29b558219e06e720",
  "selected": 356,
  "uploaded": false
}
```

Candidate ở `outputs/submission_candidate.csv`, đủ 596 ID và không ghi đè incumbent. Không upload Kaggle. `rank_changes.csv` và `rank_by_label.csv` ghi thay đổi vị trí theo ID/nhãn để giải thích, không dùng chúng để sửa prediction.

Verifier **PASSED**: 20 outer metric rows, 16 inner choices, gate của 5 folds, reload CatBoost/neural final và kiểm tra 596 IDs. Outer folds đã được xem nhiều lần nên kết quả exploratory.
