# Kiểm soát độ nhạy và biểu diễn mệnh đề đóng góp — kết quả

## Kết quả chính

Quy trình chọn arm/ngưỡng bằng inner CV đạt **H=0.63768** ở seed chính; mốc reference cố định đạt **0.64028**. Quy trình cũ tự chọn nhiều nhánh đạt 0.63734; đây là một mốc lịch sử khác với reference cố định. Không dùng điểm của arm thắng outer để thay điểm toàn quy trình.

| family              |       H |       Q |       R |   FP |   FN |   missed_high |     AUC |   AUC_gap |   logloss |
|:--------------------|--------:|--------:|--------:|-----:|-----:|--------------:|--------:|----------:|----------:|
| reference           | 0.64028 | 0.53250 | 0.80324 |  338 |  313 |            38 | 0.79946 |   0.07470 |   1.28296 |
| regularized         | 0.63863 | 0.52933 | 0.80667 |  348 |  307 |            37 | 0.79864 |   0.03816 |   1.28152 |
| repaired            | 0.63931 | 0.51742 | 0.83763 |  392 |  258 |            25 | 0.80083 |   0.03628 |   1.27976 |
| prefix              | 0.63957 | 0.52587 | 0.81639 |  360 |  292 |            39 | 0.79891 |   0.04084 |   1.28367 |
| claims              | 0.63885 | 0.52745 | 0.81277 |  356 |  299 |            36 | 0.80049 |   0.03719 |   1.28199 |
| selection_procedure | 0.63768 | 0.52540 | 0.81310 |  357 |  298 |            40 | 0.79963 |   0.03773 |   1.28255 |

- Gói regularization depth3/L2=32 thay cho depth4/L2=8: delta H **-0.00164**, delta AUC gap **-0.03654**, delta validation AUC **-0.00082**. Giảm gap chỉ có ý nghĩa khi xem cùng hiệu năng validation; không tự coi gap nhỏ là model tốt hơn.
- Câu mệnh đề so với câu đầu abstract, cùng encoder/max budget/head: delta H seed chính **-0.00072**, tốt hơn ở **2/5 folds**. Các seed phụ được báo đầy đủ phía dưới; không chọn seed cao điểm nhất.
- Diễn giải quyết định: **Chưa có bằng chứng nhất quán để thay model đối chứng; giữ các nhánh mới ở dạng thí nghiệm.** Không áp dụng lại ngưỡng tăng 0.02 H tự đặt như điều kiện bắt buộc của assignment.

## Đối chứng và độ nhạy

|     seed | arm                 | reference   |   delta_H |   improved_folds |   delta_AUC |   delta_auc_gap |   delta_FP |   delta_FN |   delta_high_FN |
|---------:|:--------------------|:------------|----------:|-----------------:|------------:|----------------:|-----------:|-----------:|----------------:|
| 20261018 | regularized         | reference   |  -0.00164 |                2 |    -0.00082 |        -0.03654 |         10 |         -6 |              -1 |
| 20261018 | repaired            | regularized |   0.00068 |                3 |     0.00220 |        -0.00188 |         44 |        -49 |             -12 |
| 20261018 | prefix              | repaired    |   0.00026 |                4 |    -0.00193 |         0.00457 |        -32 |         34 |              14 |
| 20261018 | claims              | repaired    |  -0.00046 |                3 |    -0.00035 |         0.00091 |        -36 |         41 |              11 |
| 20261018 | claims              | prefix      |  -0.00072 |                2 |     0.00158 |        -0.00365 |         -4 |          7 |              -3 |
| 20261018 | selection_procedure | reference   |  -0.00259 |                2 |     0.00018 |        -0.03697 |         19 |        -15 |               2 |
| 20261019 | regularized         | reference   |   0.00252 |                3 |    -0.00010 |        -0.03679 |          6 |         -8 |              -7 |
| 20261019 | repaired            | regularized |  -0.00113 |                2 |     0.00007 |        -0.00063 |         34 |        -38 |              -6 |
| 20261019 | prefix              | repaired    |   0.00007 |                2 |    -0.00016 |         0.00147 |        -23 |         21 |              13 |
| 20261019 | claims              | repaired    |  -0.00205 |                2 |     0.00259 |        -0.00281 |        -27 |         42 |               8 |
| 20261019 | claims              | prefix      |  -0.00212 |                1 |     0.00275 |        -0.00429 |         -4 |         21 |              -5 |
| 20261019 | selection_procedure | reference   |   0.00195 |                4 |     0.00079 |        -0.03784 |         17 |        -24 |              -5 |
| 20261020 | regularized         | reference   |   0.00213 |                2 |     0.00038 |        -0.03816 |          8 |        -23 |              -6 |
| 20261020 | repaired            | regularized |  -0.00122 |                2 |    -0.00090 |         0.00112 |         38 |        -35 |              -5 |
| 20261020 | prefix              | repaired    |  -0.00074 |                2 |    -0.00144 |         0.00321 |        -26 |         33 |              13 |
| 20261020 | claims              | repaired    |   0.00050 |                3 |    -0.00049 |         0.00173 |        -33 |         39 |               8 |
| 20261020 | claims              | prefix      |   0.00124 |                2 |     0.00094 |        -0.00148 |         -7 |          6 |              -5 |
| 20261020 | selection_procedure | reference   |  -0.00150 |                2 |    -0.00019 |        -0.03665 |         28 |        -23 |              -2 |

Seed20261018 là seed chính: chọn cutoff0..0.90 bước0.05 bằng bốn inner folds cho từng arm và chọn arm chung. Seeds20261019/20261020 chỉ fit lại outer heads, giữ nguyên cutoff và lựa chọn arm của seed chính. Chúng đo độ nhạy khởi tạo/head, không phải ba CV độc lập hoặc ba lần tune. Input/history/projection giữ nguyên. Không chọn seed tốt nhất để fit model cuối.

| family              |   H_primary |   H_mean |   H_min |   H_max |   H_range |   mean_AUC |   mean_auc_gap |   selection_flips_seed20261019 |   selection_flips_seed20261020 |
|:--------------------|------------:|---------:|--------:|--------:|----------:|-----------:|---------------:|-------------------------------:|-------------------------------:|
| reference           |     0.64028 |  0.63852 | 0.63749 | 0.64028 |   0.00279 |    0.79948 |        0.07487 |                             97 |                             95 |
| regularized         |     0.63863 |  0.63953 | 0.63863 | 0.64001 |   0.00138 |    0.79930 |        0.03771 |                             85 |                             80 |
| repaired            |     0.63931 |  0.63897 | 0.63871 | 0.63931 |   0.00060 |    0.79976 |        0.03725 |                             86 |                             76 |
| prefix              |     0.63957 |  0.63883 | 0.63797 | 0.63957 |   0.00160 |    0.79858 |        0.04033 |                             90 |                             81 |
| claims              |     0.63885 |  0.63830 | 0.63683 | 0.63921 |   0.00238 |    0.80034 |        0.03719 |                             84 |                             89 |
| selection_procedure |     0.63768 |  0.63781 | 0.63630 | 0.63943 |   0.00313 |    0.79974 |        0.03772 |                             94 |                             90 |

Biên độ H qua ba head seeds: reference **0.00279**, regularized **0.00138**. Độ lệch chuẩn H giữa năm folds ở seed chính lần lượt **0.01120** và **0.01555**. Đây là hai mô tả độ nhạy khác nhau, không phải một phép phân rã phương sai. Không được coi giảm train–validation gap là đã giảm mọi loại độ nhạy, hoặc chọn seed mới để chữa lỗi.

![Độ nhạy qua các seed](outputs/stability_comparison.png)

[PDF để xuất](outputs/stability_comparison.pdf). Fold H thay đổi giữa các seed và các folds là mô tả, không phải khoảng tin cậy độc lập. Dữ liệu outer đã được xem trong các vòng nghiên cứu trước; kết quả exploratory chưa chứng minh tăng điểm Private Kaggle.

## Đã thay đổi gì trong biểu diễn

Năm arm cố định: reference, regularized cùng feature cũ; repaired sửa education và tutorial; prefix thêm ba khối câu đầu abstract; claims thêm ba khối ứng viên problem/method/result. Cả hai text arm cùng frozen MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, mean pooling/L2, max128 tokens mỗi khối, SVD16 fit riêng trong training partition và một presence indicator mỗi khối. Tổng51 feature mới ở cả hai arm. Không fine-tune encoder và không sinh văn bản/nhãn sở thích.

SVD không học từ validation/test. Các features lịch sử cũ vẫn được cross-fit trong fit partition tương ứng, giữ nguyên thứ tự cột cũ. Câu được chọn bằng cues tổng quát; vai trò là ứng viên, không phải phân tích đóng góp ngữ nghĩa đã được xác minh. Không fallback sang toàn abstract khi không có cue; trường thiếu giữ vector0 và presence0.

| kind   | role    |    n |   available |
|:-------|:--------|-----:|------------:|
| prefix | problem | 2494 |        2396 |
| prefix | method  | 2494 |        2304 |
| prefix | result  | 2494 |        1807 |
| claims | problem | 2494 |         651 |
| claims | method  | 2494 |        1585 |
| claims | result  | 2494 |         998 |

| kind   | role    |   available |   truncated |   mean_tokens |
|:-------|:--------|------------:|------------:|--------------:|
| prefix | problem |        2974 |          57 |      64.50471 |
| prefix | method  |        2853 |          48 |      64.45952 |
| prefix | result  |        2226 |          36 |      57.81177 |
| claims | problem |         816 |           5 |      41.36152 |
| claims | method  |        1971 |          19 |      49.52308 |
| claims | result  |        1271 |          27 |      48.92840 |

Prefix và claims có cùng số khối và trần token, nhưng không có cùng số từ thực tế. Claims có nhiều khối unknown; prefix lấy sáu câu đầu chia ba khối. Vì vậy chênh lệch không chỉ đo “hiểu vai trò ngữ nghĩa”, mà có thể gồm ảnh hưởng của lượng/vị trí văn bản và coverage. Đây là đối chứng mạnh hơn việc chỉ thêm một encoder mới, nhưng chưa tách hết các yếu tố.

Các câu và offsets ở [sentence_evidence.csv](outputs/sentence_evidence.csv); văn bản model nhận ở [extracted_texts.csv](outputs/extracted_texts.csv). Sửa education chỉ đòi hỏi từ ngữ giáo dục rõ thay standalone student; tutorial là literal document-type flag. Các sửa chữa là label-independent, không bảo đảm làm tăng H. [repair_ledger.csv](outputs/repair_ledger.csv) lưu thay đổi.

## Những failure còn lại

So với quy trình tự chọn cũ: sửa **110** quyết định, phát sinh **110** lỗi mới; tổng lỗi nhị phân **655 → 655**. H còn phụ thuộc mức nhãn/thứ tự nên tổng lỗi không thay thế metric chính.

|   id | title                                                                                                                                                                                                                           |   Label | old_status   | new_status   |   claims_fields_present |
|-----:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|:-------------|:-------------|------------------------:|
|  206 | Explainable and Trustworthy Traffic Sign Detection for Safe Autonomous Driving: An Inductive Logic Programming Approach.                                                                                                        |       1 | FP           | FP           |                       3 |
|  217 | Epistemic Logic Programs: a Novel Perspective and Some Extensions.                                                                                                                                                              |       1 | FP           | FP           |                       1 |
|  355 | Epistemic Logic Programs: A Different World View.                                                                                                                                                                               |       5 | TP           | TP           |                       2 |
| 1000 | Exploring Antimicrobial Resistance Prediction Using Post-hoc Interpretable Methods.                                                                                                                                             |       5 | FN           | TP           |                       0 |
| 1001 | Towards Understanding ICU Treatments Using Patient Health Trajectories.                                                                                                                                                         |       4 | TP           | TP           |                       1 |
| 1045 | Artificial Intelligence in Medicine: Knowledge Representation and Transparent and Explainable Systems - AIME 2019 International Workshops, KR4HC/ProHealth and TEAAM, Poznan, Poland, June 26-29, 2019, Revised Selected Papers |       5 | FN           | FN           |                       0 |
| 1184 | Knowledge Representation for Health Care - HEC 2016 International Joint Workshop, KR4HC/ProHealth 2016, Munich, Germany, September 2, 2016, Revised Selected Papers                                                             |       1 | TN           | TN           |                       0 |
| 2383 | Soundness of reset workflow nets.                                                                                                                                                                                               |       5 | FN           | TP           |                       0 |

Trong input hiện có, ID1045,1000,2383 không có mệnh đề nào khớp cues nên cả ba claim fields đều unknown; prefix vẫn có thể nhận phần văn bản đầu abstract. Đây là giới hạn cụ thể của extractor theo cách diễn đạt, không phải bằng chứng rằng các bài thiếu abstract hoặc không có đóng góp. Các số này phải được xem cùng đối chứng prefix trước khi quy kết hiệu quả cho semantic role extraction.

|   Label |   n |   no_claim_fields |   no_prefix_fields |
|--------:|----:|------------------:|-------------------:|
|       1 | 903 |               209 |                 85 |
|       2 | 514 |                69 |                  5 |
|       3 | 438 |                61 |                  3 |
|       4 | 367 |                58 |                  2 |
|       5 | 272 |                58 |                  3 |

Hai cặp từng có cùng category flags giờ có thể nhận các đoạn câu khác nhau:

| pair_id   |   a_id |   b_id |   category_flag_differences |   different_claim_blocks |
|:----------|-------:|-------:|----------------------------:|-------------------------:|
| P05       |    217 |    355 |                           0 |                        2 |
| P16       |   3016 |   2758 |                           0 |                        3 |

Đây là kiểm tra input representation, chưa chứng minh model hiểu được hay dùng đúng chi tiết đó. Các cặp20 bài đã giúp thiết kế extractor nên là diagnostic probes, không phải tập kiểm tra chất lượng extraction chưa từng thấy. Không gán lại preference hay ghép quyết định theo ID. Mỗi arm/seed có priority riêng trong `oof_predictions.csv`; chuyển đổi từng lỗi trong `failure_transitions.csv`.

## Model cuối và tái lập

Chọn lại arm/cutoff trên bốn inner folds của toàn train, cùng membership đã dùng trước, rồi fit toàn2494 labels bằng seed chính:

```json
{
  "family": "reference",
  "threshold": 0.55,
  "inner_score": 0.6386873823729493
}
```

Ứng viên [submission_candidate.csv](outputs/submission_candidate.csv) đủ596 ID, 356 priority dương. Classifier và recipe ở `final_classifier.cbm`, `final_recipe.joblib`; bản trước không bị thay thế, chưa gửi Kaggle. Lựa chọn toàn train có thể khác arm có outer H cao nhất; không chọn cuối bằng outer labels.

Kiểm tra SHA-256 cho thấy CSV cuối **trùng hoàn toàn với ứng viên trước**. Đây là kết quả chọn lại bằng inner CV, không phải loại model chỉ vì không vượt một ngưỡng tăng điểm tự đặt.

Đã qua5 unit tests, kiểm tra 20670 evidence spans, 90 inherited crossfits và 180 projection blocks; tính lại 90 outer metric rows và tất cả inner choices. Dựng lại feature sau khi tải model cho cùng selection và toàn bộ thứ tự submission.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/claim_stability/run_plan.py
```

Xem [PLAN.md](PLAN.md), [outputs/verification.json](outputs/verification.json), [outputs/paired_effects.csv](outputs/paired_effects.csv), [outputs/decision.json](outputs/decision.json). Classification là phương pháp môn học; các embedding chỉ là biểu diễn đầu vào.
