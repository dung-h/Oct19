# Kết quả sửa feature

**The selection policy did not improve mean H; retain the incumbent as the main deliverable.**

Giữ cùng CatBoost 300 trees/depth4/L2=8, seed và 5outer/4inner groups. Threshold
và arm chọn trong inner folds. Toàn selection policy delta H **-0.00163**;
arm outer cao nhất là `reference`, nhưng không dùng outer winner để chọn final.
Đây là các fold đã nghiên cứu nhiều lần, nên score là exploratory, không là
validation độc lập hay điểm Kaggle. Mốc0.7 chưa được suy ra từ feature retention.

## Kết quả toàn bộ các arm

| policy              |       H |       Q |       R |     AUC |   train_AUC |   gain_MAE |        FP |        FN |   missed_high |   AUC_gap |
|:--------------------|--------:|--------:|--------:|--------:|------------:|-----------:|----------:|----------:|--------------:|----------:|
| reference           | 0.64028 | 0.53250 | 0.80324 | 0.79946 |     0.87416 |    0.21904 | 338.00000 | 313.00000 |      38.00000 |   0.07470 |
| counts              | 0.63693 | 0.52271 | 0.81692 | 0.79997 |     0.87404 |    0.21841 | 373.00000 | 292.00000 |      33.00000 |   0.07407 |
| title128            | 0.63843 | 0.52222 | 0.82243 | 0.79940 |     0.88503 |    0.21875 | 375.00000 | 282.00000 |      36.00000 |   0.08563 |
| abstract64          | 0.63682 | 0.51871 | 0.82535 | 0.79899 |     0.88093 |    0.21878 | 388.00000 | 278.00000 |      33.00000 |   0.08194 |
| rules               | 0.63937 | 0.52338 | 0.82465 | 0.79808 |     0.87319 |    0.21896 | 374.00000 | 278.00000 |      31.00000 |   0.07512 |
| matched             | 0.63982 | 0.52985 | 0.80828 | 0.80046 |     0.87416 |    0.21981 | 354.00000 | 305.00000 |      34.00000 |   0.07370 |
| combined            | 0.63857 | 0.51840 | 0.83148 | 0.79873 |     0.88595 |    0.22062 | 384.00000 | 268.00000 |      30.00000 |   0.08723 |
| selection_procedure | 0.63864 | 0.51837 | 0.83311 | 0.79990 |     0.88075 |    0.21990 | 387.00000 | 266.00000 |      30.00000 |   0.08085 |

Mean H tính trên từng fold; không lấy harmonic mean của Q/R đã trung bình. FP/FN
và missed-high là tổng2.494 OOF records. Binary errors không thay thế H.

## Hiệu ứng so với reference cùng model

| arm                 |   delta_H |   folds_improved |   delta_Q |   delta_R |   delta_AUC |   delta_FP |   delta_FN |   delta_missed_high |
|:--------------------|----------:|-----------------:|----------:|----------:|------------:|-----------:|-----------:|--------------------:|
| counts              |  -0.00335 |                2 |  -0.00979 |   0.01368 |     0.00051 |         35 |        -21 |                  -5 |
| title128            |  -0.00185 |                2 |  -0.01028 |   0.01920 |    -0.00006 |         37 |        -31 |                  -2 |
| abstract64          |  -0.00345 |                1 |  -0.01379 |   0.02211 |    -0.00047 |         50 |        -35 |                  -5 |
| rules               |  -0.00091 |                1 |  -0.00912 |   0.02142 |    -0.00138 |         36 |        -35 |                  -7 |
| matched             |  -0.00045 |                2 |  -0.00265 |   0.00504 |     0.00100 |         16 |         -8 |                  -4 |
| combined            |  -0.00171 |                1 |  -0.01410 |   0.02824 |    -0.00073 |         46 |        -45 |                  -8 |
| selection_procedure |  -0.00163 |                2 |  -0.01413 |   0.02987 |     0.00044 |         49 |        -47 |                  -8 |

`counts` chỉ sửa raw reference/support fields. `title128` thay SVD32 bằng128;
`abstract64` thêm64 chiều lexical riêng và giữ presence flag đã có; `rules` sửa
wording education/neural và thêm tutorial; `matched` giữ nguyên model/training
frame reference, đổi inference sang ba reference subsets rồi average probability.
`combined` đo kết hợp trực tiếp. Không cộng delta từng arm để dự đoán combined.
Số feature từng arm: {'reference': 277, 'counts': 271, 'title128': 373, 'abstract64': 341, 'rules': 279, 'matched': 277, 'combined': 433}. Catalog theo đúng column order của
saved models ở `outputs/feature_catalog.csv`.

## Feature đã được sửa và giới hạn còn lại

- Full input/cache guard mới hash cả venue/year/authors/doi/Label, ZIP, sạch
  abstract, embedding caches, parent feature packs, splits và source.
- Có **92** thay đổi flag trên **50**
  records. Ledger và nguyên văn bằng chứng nằm trong `rule_changes.csv` và
  `rule_evidence.csv`; đây là lexical evidence, không phải nhãn sở thích mới.
- Trong năm outer training fits, SVD title32 giữ mean variance
  **0.1320**, title128 **0.3176**.
  Abstract64 giữ **0.1141**. Không diễn giải các
  tỷ lệ này thành phần trăm thông tin nhãn bị mất hoặc hiệu quả dự đoán chắc chắn.
- Mean OOV occurrence rate ở outer prediction: title word1/2
  **0.4293**, abstract **0.2951**.
  Bigrams/min_df/vocabulary cap góp vào OOV; đây không phải tỷ lệ toàn paper không
  có representation.
- Matched inference dùng đúng reference subset sizes của crossfit train; các
  query overlap groups được loại khỏi reference. Thay đổi này đồng thời ảnh
  hưởng Ridge phụ, vocab references, neighbor statistics và averaging; không
  quy delta riêng cho reference count.
- SVD vẫn không giám sát và nén thông tin; cờ vẫn không hiểu đóng góp cụ thể;
  authors chưa được thêm; chưa có encoder abstract học theo preference trong
  thí nghiệm này. Không tự gán nhãn sai cho các paper có preference khác nhau.

| feature                   |   changed |   before_sum |   after_sum |
|:--------------------------|----------:|-------------:|------------:|
| abstract_domain_education |        16 |            8 |           8 |
| abstract_domain_unknown   |         6 |            6 |           0 |
| abstract_method_neural    |        14 |           14 |           0 |
| abstract_method_unknown   |         6 |            0 |           6 |
| abstract_type_tutorial    |        16 |            0 |          16 |
| abstract_type_unknown     |        11 |           11 |           0 |
| title_domain_education    |         2 |            0 |           2 |
| title_domain_unknown      |         2 |            2 |           0 |
| title_method_neural       |         5 |            5 |           0 |
| title_method_unknown      |         4 |            0 |           4 |
| title_type_tutorial       |         5 |            0 |           5 |
| title_type_unknown        |         5 |            5 |           0 |

## Vì sao tăng feature chưa kéo được H?

Validation AUC của các arm sửa nằm trong **0.79808..
0.80046**, so với reference
**0.79946**. Tăng title dimensions đưa train AUC từ
**0.87416** lên **0.88503**,
nhưng validation AUC là **0.79940**. Combined train AUC
**0.88595**, validation
**0.79873**. Đây là bằng chứng rằng các thay đổi này fit
train tốt hơn mà chưa bổ sung khả năng phân biệt preference ngoài fit tương ứng;
không chứng minh encoder khác hoặc mọi biểu diễn abstract đều vô ích.

Combined so với reference: Q đổi **-0.01410**,
recall đổi **+0.02824**, FP đổi
**+46** và FN đổi
**-45**. Phần gain
recall đi cùng nhiều false positives hơn, nên H không tăng. Không quy kết chung
preprocessing gây plateau chỉ từ việc SVD giữ ít variance hoặc rule có lỗi từ ngữ.

Đối chiếu hậu nghiệm giữ read gate của một model và gain ranker của model kia,
giữ đúng thresholds đã chọn trong inner, cho phân rã sau:

| policy              |   delta_H |   gate_contribution |   rank_contribution |
|:--------------------|----------:|--------------------:|--------------------:|
| counts              |  -0.00335 |            -0.00333 |            -0.00001 |
| title128            |  -0.00185 |            -0.00270 |             0.00085 |
| abstract64          |  -0.00345 |            -0.00418 |             0.00073 |
| rules               |  -0.00091 |            -0.00156 |             0.00065 |
| matched             |  -0.00045 |            -0.00114 |             0.00069 |
| combined            |  -0.00171 |            -0.00162 |            -0.00009 |
| selection_procedure |  -0.00163 |            -0.00168 |             0.00005 |

Gate contribution và rank contribution dùng trung bình hai thứ tự thay component
(Shapley accounting), cộng lại đúng delta H. Đây là phân rã metric đã quan sát,
không là chứng minh nguyên nhân gốc và không là các candidate được chọn/triển khai.
Các hybrid chỉ dùng chẩn đoán, không dùng outer labels để chọn final recipe.

## Final candidate và kiểm chứng

Final inner chọn `reference`, cutoff **0.55**,
inner H **0.63869**. Candidate ở
`outputs/submission_candidate.csv`, model ở `outputs/final_reference.cbm`,
recipe ở `outputs/final_recipe.joblib`. Giữ nguyên incumbent; không upload.

Verifier `PASSED`: replay **30**
partitions của control; kiểm tra **90**
crossfits, **90** matched views và
**60** projections; tính lại
**40** outer metric rows và
**48** inner choices. Đã kiểm tra original
labels, missing abstract vectors, metadata-sensitive hashes, evidence spans,
heldout-label invariance, rebuild final features và model reload; CSV đủ596 ID.

Từng case/fold ở `outputs/failure_transitions.csv`, `failure_breakdown.csv` và
`paired_effects_by_fold.csv`. Xem [PLAN.md](PLAN.md),
[verification.json](outputs/verification.json), [feature audit](outputs/feature_audit.csv).
