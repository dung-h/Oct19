# Kiểm tra kết hợp các sửa đổi

Đây là kết quả thăm dò trên cùng năm outer folds đã được xem nhiều lần. Không phải điểm Kaggle hay đánh giá trên holdout độc lập. Dùng H theo phần Evaluation chi tiết của đề bài; mâu thuẫn mô tả ngắn 60/40 vẫn chưa được xác minh.

## Kết quả

| family                            |        H |        Q |        R |      AUC |   train_AUC |
|:----------------------------------|---------:|---------:|---------:|---------:|------------:|
| Multiclass hiện tại               | 0.640276 | 0.532500 | 0.803238 | 0.799457 |    0.874155 |
| Ordinal: khóa mặc định            | 0.638415 | 0.534934 | 0.793964 | 0.798730 |    0.893307 |
| Multiclass: bỏ láng giềng         | 0.637780 | 0.524482 | 0.816171 | 0.801230 |  nan        |
| Ordinal: chỉ bỏ láng giềng        | 0.639459 | 0.538506 | 0.791189 | 0.802636 |    0.932036 |
| Ordinal: kết hợp hai sửa đổi      | 0.637296 | 0.538108 | 0.784167 | 0.801666 |    0.871986 |
| Ordinal: kết hợp + regularization | 0.637555 | 0.526074 | 0.810979 | 0.800731 |    0.836590 |
| Ordinal: chuẩn hóa cột đếm        | 0.638909 | 0.531809 | 0.801094 | 0.801090 |    0.894318 |
| Multiclass: chuẩn hóa cột đếm     | 0.636930 | 0.522711 | 0.816921 | 0.799966 |    0.874039 |
| Quy trình chọn bằng inner CV      | 0.636459 | 0.537880 | 0.783202 | 0.804059 |    0.884583 |

Ordinal gốc ở thí nghiệm trước: H=0.635347. Mọi ngưỡng và lựa chọn mô hình trong bảng được quyết định bằng bốn inner folds; không chọn bằng nhãn outer. AUC train của multiclass bỏ láng giềng không có trong cache cũ.

## Các đối chứng đã định trước

| contrast                               | treated                  | control               |   delta_H |   improved_folds |   min_fold_delta |   max_fold_delta |   delta_AUC |   delta_train_AUC |
|:---------------------------------------|:-------------------------|:----------------------|----------:|-----------------:|-----------------:|-----------------:|------------:|------------------:|
| Combined minus defaults only           | ORD_combined             | ORD_defaults_full     | -0.001119 |                3 |        -0.014219 |         0.006363 |    0.002936 |         -0.021321 |
| Combined minus neighbor removal only   | ORD_combined             | ORD_without_neighbors | -0.002163 |                3 |        -0.012171 |         0.002433 |   -0.000971 |         -0.060050 |
| Stronger regularization after combined | ORD_combined_regularized | ORD_combined          |  0.000259 |                3 |        -0.007982 |         0.008001 |   -0.000935 |         -0.035396 |
| Count fractions, ordinal               | ORD_count_fraction       | ORD_defaults_full     |  0.000495 |                3 |        -0.006525 |         0.005053 |    0.002360 |          0.001012 |
| Count fractions, multiclass            | MC_count_fraction        | MC_full               | -0.003346 |                2 |        -0.010989 |         0.001320 |    0.000509 |         -0.000116 |

## Các lỗi chọn/bỏ bài so với multiclass hiện tại

| configuration            |   recovered_positive |   newly_missed_positive |   removed_false_positive |   added_false_positive |   recovered_high |   newly_missed_high |   FP |   FN |   missed_high |
|:-------------------------|---------------------:|------------------------:|-------------------------:|-----------------------:|-----------------:|--------------------:|-----:|-----:|--------------:|
| MC_full                  |                    0 |                       0 |                        0 |                      0 |                0 |                   0 |  338 |  313 |            38 |
| ORD_defaults_full        |                   30 |                      44 |                       31 |                     24 |                4 |                  10 |  331 |  327 |            44 |
| MC_without_neighbors     |                   74 |                      53 |                       47 |                     71 |               14 |                  11 |  362 |  292 |            35 |
| ORD_without_neighbors    |                   68 |                      88 |                       78 |                     59 |               11 |                  19 |  319 |  333 |            46 |
| ORD_combined             |                   53 |                      83 |                       67 |                     45 |                6 |                  19 |  316 |  343 |            51 |
| ORD_combined_regularized |                   72 |                      59 |                       56 |                     74 |               13 |                  11 |  356 |  300 |            36 |
| ORD_count_fraction       |                   31 |                      35 |                       19 |                     24 |                4 |                   8 |  343 |  317 |            42 |
| MC_count_fraction        |                   46 |                      25 |                       13 |                     48 |                9 |                   4 |  373 |  292 |            33 |

Các số ở trên là tổng trên 2.494 dự đoán ngoài fold. FN là bài nhãn >1 bị bỏ; missed_high là bài nhãn 4/5 bị bỏ. H trong bảng đầu là trung bình năm fold, không tính lại trên một danh sách gộp.

## Tách thay đổi chọn bài và xếp hạng

| family                   |   delta_H |   gate_effect |   rank_effect |
|:-------------------------|----------:|--------------:|--------------:|
| MC_full                  |  0.000000 |      0.000000 |      0.000000 |
| ORD_defaults_full        | -0.001861 |     -0.002685 |      0.000824 |
| MC_without_neighbors     | -0.002496 |     -0.001610 |     -0.000886 |
| ORD_without_neighbors    | -0.000817 |     -0.000416 |     -0.000401 |
| ORD_combined             | -0.002980 |     -0.002791 |     -0.000189 |
| ORD_combined_regularized | -0.002721 |     -0.002069 |     -0.000653 |
| ORD_count_fraction       | -0.001367 |     -0.002055 |      0.000688 |
| MC_count_fraction        | -0.003346 |     -0.003333 |     -0.000013 |

Hoán đổi gate/ranker với multiclass hiện tại rồi chia đều phần tương tác theo Shapley. Đây là phân rã thay đổi H trên dự đoán đã có, không phải bằng chứng riêng về nguyên nhân trong biểu diễn hoặc dữ liệu. Hai thành phần cộng đúng bằng delta H.

## Cột đếm trước và sau chuẩn hóa

| column                           |   raw_count |   support_fraction |
|:---------------------------------|------------:|-------------------:|
| neighbor_abstract_all_support    |    0.960697 |           0.000000 |
| neighbor_abstract_venue_support  |    0.882060 |           0.000000 |
| neighbor_retrieval_all_support   |    1.000000 |           0.000000 |
| neighbor_retrieval_venue_support |    0.953486 |           0.000000 |
| neighbor_title_all_support       |    0.129906 |           0.000403 |
| neighbor_title_venue_support     |    0.091844 |           0.000403 |

Mỗi số là tỷ lệ query ngoài khoảng min/max của train, lấy trung bình năm fold. Chỉ số này mô tả phạm vi giá trị, không đo đầy đủ distribution shift.

## Giới hạn suy luận

Phép chuẩn hóa chỉ loại ảnh hưởng của đơn vị đếm ở sáu cột support và bỏ sáu cột references. Những thay đổi về hàng xóm gần nhất, phân bố nhãn hàng xóm và số ví dụ tham chiếu thực tế vẫn còn. Vì vậy không thể dùng kết quả này để khẳng định đã giải quyết mọi chênh lệch train/predict của history features.

Khóa mặc định là thay đổi đồng thời bootstrap và số lần cập nhật leaf. Bỏ láng giềng là thay đổi cả nhóm 150 feature. Đây là tác động của từng gói sửa đổi trong cấu hình cố định, chưa tách riêng từng nguyên nhân.

## Candidate cuối

Quy trình inner CV cuối chọn `ORD_combined_regularized`, cutoff=0.50, inner H=0.639034. `outputs/submission_candidate.csv` lưu đúng lựa chọn này; việc lưu candidate không chứng minh nó tốt hơn incumbent trên dữ liệu mới.

Giữ nguyên incumbent ở `../preference_augmentation/outputs/submission_candidate.csv`. Không upload Kaggle. Xem `outputs/verification.json` cho kết quả kiểm tra độc lập, `count_drift.csv` cho thống kê trước/sau chuẩn hóa và `gate_changes.csv` cho lỗi thay đổi.
