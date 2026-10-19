# Kế hoạch: sửa false positive bằng disagreement gate có nested cross-fitting

Trạng thái: kế hoạch, chưa chạy model mới và chưa tạo submission. Kế hoạch này
được lập sau audit `cross_model_error_audit`; không dùng metric của audit để
chọn weight, threshold hay model cuối.

## 1. Điều cần sửa

Incumbent có TP=1.278, FP=338, FN=313 và H=0.640276 trên 2.494 outer OOF rows.
137/338 FP được cả 11 prediction streams hiện có chọn. Mỗi source model dùng
riêng như veto đều loại TP nhiều hơn FP: `title_margin` loại 35 FP nhưng loại
50 TP, còn `neural_adapted` loại 135 FP nhưng loại 203 TP.

Hypothesis mới là: mẫu score disagreement giữa một số model có thể dự đoán
paper incumbent chọn nhầm tốt hơn hard vote, khi pattern đó chỉ được học trong
outer-train. Đây không phải hypothesis rằng một model riêng là veto.

## 2. Tín hiệu khóa trước

Ba source model được dùng, không thêm source sau khi xem kết quả:

| Source | Vai trò |
|---|---|
| `incumbent` | CatBoost 277 features, tín hiệu chính ổn định nhất |
| `title_margin` | Incumbent cộng top-k positive/negative title-MiniLM margins; H OOF tốt nhất trong audit |
| `neural_adapted` | Preference-adapter; abstention pattern khác nhất, nhưng không dùng như veto độc lập |

Không dùng 11 streams, per-ID rule, venue-specific threshold, Label, ID, title
text, author hay external feature trong meta gate.

## 3. Meta-gate cố định

Meta feature gồm 9 số: `logit(P_read)` và expected gain của từng ba source,
cộng ba chênh lệch pairwise của `logit(P_read)`. Clip probability vào
`[1e-5, 1-1e-5]` trước logit.

Fit `StandardScaler` trên meta-train, sau đó `LogisticRegression` L2, solver
`lbfgs`, `C=0.2`, `class_weight=None`, seed cố định. C được khóa, không mở grid.
Target là `Label > 1`. Meta probability quyết định nhận/bỏ; thứ tự các paper
được nhận vẫn theo incumbent expected gain. Lượt này chỉ sửa gate, không thử
ranker mới.

## 4. Nested protocol chống leakage

Giữ 5 publication-group outer folds và 4 grouped inner folds.

1. Với mỗi outer fold, fit ba source trên ba inner folds rồi dự đoán inner
   validation, đủ bốn lượt. Đây là meta-feature OOF của outer-train.
2. Cross-fit meta logistic theo bốn inner partitions: meta prediction của một
   partition chỉ học từ ba partition còn lại. Dùng những meta prediction này
   để chọn threshold trong grid cố định `{0.45,0.50,0.55,0.60,0.65}` theo mean
   inner H; tie chọn threshold cao hơn.
3. Refit ba source trên toàn outer-train và dự đoán outer validation. Fit
   logistic cuối trên toàn meta-OOF outer-train rồi dự đoán outer validation.
4. Gate strict là `P_meta(read) > threshold`; priority dương dùng incumbent
   expected gain. Không xem outer labels để thay source, C, grid, ranker hay seed.

Không tái dùng outer OOF prediction cache làm meta-train: source predictions
trong meta-train phải được tạo chỉ từ inner-train tương ứng.

## 5. Audit bắt buộc

- Hash dữ liệu, labels, splits, abstract cache, embeddings, code và plan trước run.
- Assert group của query rời source/meta train và reference history.
- Check probability finite/tổng bằng 1, IDs/order, calibration và source replay.
- Perturb Label của query: meta feature inference không được thay đổi.
- Reload model và rebuild feature để tái lập raw probability, gate, priority, H.
- Lưu score contributions chỉ để audit; không viết luật theo title hoặc ID.

## 6. Rule quyết định khóa trước

Báo từng fold và pooled count: H/Q/R, accepted/dropped, TP/FP/FN/TN, missed
Label4/5, AUC/Brier, calibration và error transitions.

Chỉ coi candidate có triển vọng khi đồng thời:

1. Mean outer H `>= 0.640276` và H cao hơn incumbent ở ít nhất 3/5 folds.
2. FP `<=338`, FN `<=313`, ít nhất một số giảm.
3. Missed Label4 `<=33`, Label5 `<=5`, và `FP+FN` không tăng.
4. Source replay, group isolation và independent verifier đều pass.

Nếu đạt, tạo candidate CSV riêng sau full-train inner selection; không ghi đè
incumbent hoặc upload Kaggle.

## 7. Điểm dừng và hướng sau đó

Nếu không qua toàn bộ criteria, dừng nhánh score-only ensemble: không thêm
model thứ 4/5, vote threshold khác, capacity hay seed theo outer score. 40,5%
FP là lỗi consensus nên các biến thể của cùng score khó sửa phần chính này.

Khi đó phải có một nguồn thông tin mới, audit trước khi train: graph/citation
metadata công khai theo DOI, paper type, fields-of-study, reference/citation
relationships hoặc author-topic lineage. Có identity-match ledger, coverage,
source timestamp và train-fold-only label aggregation. Không dùng citation count
hiện tại một cách âm thầm nếu nó phản ánh tương lai của paper. Trước model, so
sánh 137 consensus FP với TP cùng venue/year để xác minh feature thực sự khác
biệt và có coverage. Không gán nhãn tay hay rule theo ID.

## 8. Ngân sách và deliverables

Ba source × (5 outer × 4 inner + 5 outer refit) = 75 source fits, thêm 25 meta
logistic fits nhẹ. Neural adapter là phần tốn thời gian. Nếu không tái lập được
đúng recipe/cache, dừng trước training thay vì thay source sau protocol lock.

Deliverables: runner, unit tests, protocol/lineage hashes, `outer_predictions`,
`fold_metrics`, calibration, transitions, independent verifier, `RESULTS.md`,
`DECISION.md`; chỉ khi qua criteria mới có `submission_candidate.csv` 596 IDs.
