# Plan: kiểm tra author history và venue calibration cho quyết định nhận/rớt

Trạng thái lúc lập plan: chưa train model hoặc tính kết quả mới của lượt này.
Mục tiêu: sửa quyết định chọn/bỏ, đo trực tiếp số mẫu thay vì dùng H làm tiêu chí
chọn model. Đây vẫn là classification đúng yêu cầu assignment; H được báo phụ.

## 1. Điều đã biết và giả thuyết cần kiểm tra

Incumbent OOF trên 2.494 train: TP=1.278, FP=338, FN=313, TN=565, tổng lỗi=651.
Trong FN có 183 nhãn2, 92 nhãn3, 33 nhãn4 và 5 nhãn5. Test có 596 bài, hiện nhận
356/rớt240; nhãn test không biết nên không báo số đúng/sai trên test.

Fixed-gate ranking đã hoàn thành và không đổi các quyết định này. Lượt mới thay
gate; ranking giữ công thức 50% incumbent gain + 50% adapted-title gain đã có.

Hai phát biểu trước cần thu hẹp:

- Tỷ lệ bài nhận khác tỷ lệ nhãn dương trong venue không chứng minh miscalibration.
  Calibration phải so xác suất với tần suất nhãn, không so số được nhận với số
  nhãn dương. Ví dụ LICS có mean P(read) khoảng0.514 và tỷ lệ dương0.523, dù chỉ
  nhận39.2% ở cutoff0.55. Chưa đủ căn cứ nói venue là nguyên nhân chính.
- Author từng được thử bằng multi-hot/surname Ridge, cả metadata đã audit, nhưng
  chưa cho lợi ích nhất quán. Coverage tên lặp lại chỉ cho biết có thể tạo feature;
  không chứng minh author dự đoán sở thích tốt. Thay đổi cụ thể lần này là
  smoothed author-history features trên cùng learner và base features.

Giả thuyết khóa trước kết quả:

H1: binary read objective cải thiện quyết định so với incumbent multiclass.
H2: author-history block thêm tín hiệu ngoài277 features hiện có.
H3: venue-conditioned calibration tốt hơn global calibration cùng recipe.
H4: kết hợp H2+H3 giảm lỗi chọn/bỏ mà không đánh đổi quá nhiều FP với FN.

## 2. Baselines và sáu nhánh kiểm soát

| Nhánh | Base binary model | Calibration | Decision cutoff |
|---|---|---|---:|
| `binary_raw` | 277 features gốc | Không | 0.50 |
| `binary_global` | Cùng `binary_raw` | Global logistic | 0.50 |
| `binary_venue` | Cùng `binary_raw` | Logistic + venue | 0.50 |
| `author_raw` | 277 features +19 author fields | Không | 0.50 |
| `author_global` | Cùng `author_raw` | Global logistic | 0.50 |
| `author_venue` | Cùng `author_raw` | Logistic + venue | 0.50 |

Hai đối chứng source: `incumbent_deployed` dùng cutoff cũ chọn bằng4inner H
(các outer hiện tại0.55); `incumbent_at_050` dùng cùng probability tại0.50.
Đối chứng0.50 giúp phân biệt hiệu ứng đổi threshold với đổi model.

Chỉ train hai loại CatBoost; sáu nhánh không phải sáu encoder/models lớn. Mọi
nhánh, cutoff và tham số cố định trước khi xem kết quả. Không search threshold,
không dùng H để chọn nhánh, không đổi cutoff riêng cho từng venue bằng outer labels.

Primary candidate là `author_venue`, định trước. Các nhánh còn lại là đối chứng
để giải thích hiệu ứng, không chọn nhánh thắng outer rồi gọi là primary. Nếu
primary thất bại nhưng nhánh phụ tốt, chỉ báo phát hiện thăm dò; không tự đổi
candidate chính hoặc mở một vòng search trong lượt này.

## 3. Đích học và learner cố định

Binary target = 1 nếu Label>1, ngược lại0. Giữ labels1..5 gốc cho history gain,
failure breakdown và reporting, không tạo hoặc sửa preference labels.

CatBoost Logloss: iterations300, depth4, learning_rate0.05, l2_leaf_reg8,
seed20261018, thread_count4, Bayesian bootstrap/bagging_temperature1,
leaf_estimation_iterations1, verboseFalse, allow_writing_filesFalse.
Không class weights, resampling hoặc early-stop theo outer.

Binary control và author model dùng cùng defaults được khóa tường minh. Chênh
lệch với incumbent multiclass là hiệu ứng binary formulation/package, không quy
tất cả cho một tham số. So author_raw-binary_raw mới là phép thử author block.

## 4. Pha A: audit đầu vào và calibration hiện tại

1. Đọc rules/metric trong markdown local; source labels từ ZIP giữ nguyên.
2. Giữ grouping/folds cũ của `specter2_v2/outputs/nested_splits.csv` và
   `preference_augmentation/outputs/final_inner_splits.csv`.
3. Đọc incumbent probabilities từ `claim_stability/outputs/*_predictions.npz`
   trường `reference`; replay cutoff source bằng đúng inner partitions.
4. Audit P(read) theo outcome TP/FP/FN/TN, label, venue; reliability bins cố định
   [0,.1),[.1,.2),...,[.9,1], mean P(read), fraction positive, Brier, binary
   logloss. Báo n của từng bin và mỗi venue; không suy từ bin nhỏ ra nguyên nhân.
5. Kiểm tra lineage của cached277 feature frames từ preference_augmentation:
   fit/pred indices, columns, categorical venue, finite numerics, group isolation,
   feature source/protocol hashes và inherited target-derived crossfits.
6. Source author dùng snapshot `input_ablation/outputs/metadata_audit.csv`:
   verified authors khi đã qua kiểm tra danh tính nguồn, fallback original khi
   chưa xác minh. Giữ riêng raw/verified/used/provenance; không thay year hoặc
   sửa277 feature source trong lượt này. Join one-to-one theo `(split,id)`, không
   theo row number hoặc chỉ một trường metadata. Không gọi API để mở rộng audit.
7. Record input digest đầy đủ title/abstract/venue/year/authors/doi/Label,
   metadata snapshot, rawZIP, groups, splits và hashes mọi cache/source đọc.

Pha A ghi chẩn đoán nhưng không thay recipe/nhánh trong plan dựa vào kết quả đó.
Mismatch source/cache là lỗi phải xử lý trước train; không bỏ một fold cho qua.

## 5. Author identity và19 features

Parser label-blind: HTML decode, Unicode NFC, casefold, gộp whitespace. Giữ tên
đầy đủ và dấu; không gộp người theo surname hoặc fuzzy match. Tách theo format
thực tế của từng nguồn (comma/semicolon/and), lưu tokens cùng raw source. Dedupe
author trong cùng paper. `et al.`, danh sách không rõ, tên lỗi encoding/suffix
phải gắn cờ; không đoán danh tính để tăng coverage.

Audit missing/ambiguous/collision trước train và kiểm tra các ca author đã biết
trong ledger. Nếu format không parse an toàn, dùng phần tokens chắc chắn hoặc
unknown theo rule ghi rõ trước scoring. Parser không được sửa theo nhãn/failure ID.

Trong một reference set hợp lệ R, mỗi author a có:

```text
n_a = số paper khác nhau chứa a trong R
read_a = (sum(1[Label>1]) + 10*mu_read_R) / (n_a+10)
gain_a = (sum((Label-1)/4) + 10*mu_gain_R) / (n_a+10)
```

Smoothing10 là cố định. Query duplicate group bị loại khỏi R trước cả counts và
global priors. Người chưa gặp dùng priors, không tự coi là negative.

19 fields cố định:

1. `author_count`
2. `author_missing`
3. `author_source_verified`
4. `author_format_uncertain`
5. `known_author_count`
6. `known_author_fraction`
7. `author_cold_start`
8. `author_mean_log_support`
9. `author_max_log_support`
10. `author_unique_reference_papers_fraction` (union paper IDs / eligible |R|)
11. `author_read_mean`
12. `author_read_min`
13. `author_read_max`
14. `author_read_support_weighted_mean` (weights log1p(n_a))
15. `author_read_std`
16. `author_gain_mean`
17. `author_gain_min`
18. `author_gain_max`
19. `author_gain_support_weighted_mean`

Nếu không có author hợp lệ/known author, counts/support=0, unknown flags=1,
read/gain aggregates dùng reference prior, std=0. Số author, parse flags và
source_verified là label-free; history fields chỉ dùng reference labels.

Train rows: grouped crossfit3parts bên trong chính fit partition, sử dụng cùng
membership hợp lệ với inherited base-history features khi có thể. Mỗi query chỉ
đọc reference labels từ hai parts còn lại. Inner validation/outer validation/test
rows: references là toàn fit partition, loại toàn duplicate group của query.
Lưu row-level support, unseen flags và reference membership/hash để audit drift.
Không giả định full-train coverage tương đương outer coverage; đo lại sau parser.

## 6. Calibration recipe và chống leakage

`global`: LogisticRegression trên logit(P_binary(read)), sau StandardScaler.
`venue`: cùng scalar đã scale + venue one-hot (unknown ignored), additive effects.
C=1, solverlbfgs, max_iter1000, interceptTrue, class_weightNone. Clip xác suất
vào[1e-6,1-1e-6] trước logit; scaler/vocab fit chỉ trên calibration training rows.

Với mỗi outer train T:

1. Fit binary/author model trên từng3/4 inner training subset, predict inner
   holdout tương ứng. Features/history của mỗi fit không nhìn labels holdout.
2. Ghép bốn inner predictions để có base-model OOF probabilities cho toàn T.
3. Fit global/venue calibrators bằng các OOF probabilities, venue và binary
   targets trong T. Không dùng base-model in-sample predictions để train calibrator.
4. Refit hai base models trên toàn T. Dùng calibrator đã fit để map base-model
   predictions trên outer validation; chọn read nếu calibrated probability>0.50.

Đây là calibration theo OOF với hyperparameters/cutoff cố định. Không báo score
trên chính OOF rows sau khi calibrator đã fit chúng là validation score; chỉ outer
validation dùng để đánh giá calibration và các quyết định cuối.

Không lấy bộ OOF toàn2494 để train calibrator rồi chấm trên chính2494 rows.
Không cross-validate calibrator một cách ngây thơ trên các inner probability
rows rồi gọi là fully nested: base predictions của training-calibrator rows có
thể đến từ model đã nhìn labels validation-calibrator. Nếu muốn tune C/threshold
hoặc chọn nhiều calibrators, cần thêm nested level hoặc calibration holdout độc
lập. Lượt này tránh nhu cầu đó bằng recipe cố định, không tune.

So sánh venue-global giữ cùng base model, target, scalar/scaler và C. Calibration
có thể sửa probability scale hoặc thứ tự giữa venues; không tự tạo tín hiệu
ngữ nghĩa mới bên trong một venue. Không hứa giảm đồng thời FP và FN.

## 7. Báo cáo chính: mẫu nhận/rớt và thay đổi từng ID

Bảng bắt buộc cho mọi nhánh, từng fold và tổng2494 OOF:

- Total, accepted, dropped, TP, FP, FN, TN và total_error=FP+FN.
- Recall, precision, specificity, binary balanced accuracy (số phụ có ý nghĩa).
- Nhãn1..5: total/accepted/dropped; nhãn4/5 bị rớt riêng.
- Theo venue, author known/unseen/missing, metadata verified/unverified và
  reference-support bins cố định0,1,2-4,>=5; hiển thị denominators.
- So incumbent theo cùng ID: FN rescued, FP removed, newFP, lostTP. Kiểm tra
  `delta_error = newFP + lostTP - FN_rescued - FP_removed`.
- Brier/logloss/reliability để chẩn đoán calibration; không thay bảng quyết định.

Contrasts khóa trước: binary_raw-incumbent_at_050; author_raw-binary_raw;
binary_venue-binary_global; author_venue-author_global;
author_venue-incumbent_deployed (primary deployment contrast).

H/Q/R vẫn báo để đối chiếu assignment, không dùng ngưỡng deltaH0.003 hoặc mốc0.7
để loại một cải thiện thực tế trong quyết định chọn/bỏ.

## 8. Tiêu chí đi tiếp/dừng

User chưa gán chi phí FP/FN khác nhau, nên tạm coi mỗi lỗi chọn/bỏ có cùng trọng
lượng trong total_error. Luôn báo hai số riêng; không che tradeoff bằng tổng lỗi.

Primary `author_venue` so incumbent_deployed:

- **Cải thiện cả hai phía:** tổng FP<=338, FN<=313, ít nhất một giảm; nhãn4/5
  bị rớt<=38. Ghi thêm số fold total_error giảm; chỉ đề xuất dùng candidate nếu
  total_error giảm ít nhất4/5folds và verifier pass.
- **Tradeoff:** một trong FP/FN tăng dù total_error giảm hoặc calibration đẹp
  hơn. Báo số được cứu và số lỗi mới; không tự gọi là cải thiện cả hai phía,
  không tự chọn chi phí ưu tiên thay user.
- **Không cải thiện:** total_error không giảm, hoặc cải thiện quá tập trung ở
  một fold/subgroup. Giữ incumbent, dừng mở rộng author/calibration ở lượt này.
- **Invalid:** lineage/label/metric lỗi; sửa triển khai và ghi amendment, chưa
  kết luận về hypothesis.

Các điều kiện này là research decision rules, không là yêu cầu môn học. Outer
folds đã được xem nhiều lần, nên không gọi chúng là independent test hoặc claim
ý nghĩa thống kê chỉ từ số folds thắng. Không chọn seed tốt nhất.

## 9. Final và artifact

Final train dùng bốn inner folds toàn train để tạo OOF probabilities cho hai
base learners, fit bốn calibrators (global/venue cho mỗi learner), rồi refit hai
base learners trên2494 rows. Recipe/cutoff0.50 giữ nguyên; không chọn model theo
outer winner. Xuất mỗi branch riêng để tái tạo, primary experimental là
author_venue. Incumbent và các model/CSV cũ giữ nguyên, không upload Kaggle.

Priority = 1+9*(0.5*incumbent_gain +0.5*adapted_title_gain) nếu gate nhận, còn lại0.
Dùng adapted-title epoch đã chọn trong thí nghiệm source tương ứng, không train
encoder hoặc chọn weight ranking lại. Không quy deltaH chỉ cho gate vì tập chọn
mới có thành phần khác; ranking formula được giữ cố định để hạn chế thay đổi.

Test chỉ báo596 total, accepted/dropped và thay đổi selected IDs so incumbent.
Không ước lượng TP/FP/FN test từ model như thể là nhãn thật. Hai duplicate groups
giao source train/test được giữ nhưng loại khỏi author-history references của
query; ghi giới hạn, không gọi test group-independent.

Files cần triển khai (chưa tồn tại khi viết plan):

- `author_features.py`: parser, source ledger, crossfit histories, rebuild.
- `run_gate.py`: audit, fixed2base learners x30partitions, OOF calibrators,
  outer/final predictions; protocol lock/resume.
- `verify_gate.py`: labels/groups/source hashes, calibration fit indices,
  query-label invariance, raw-model reload, recompute every selection/count.
- `write_report.py`: counts, transitions, reliability và quyết định.
- `tests/`: parser edge cases, group exclusion/self-label perturbation, unknown
  author fallback, counts transitions và gate boundary strict >.
- `outputs/`: protocol, metadata ledger/tokens, feature catalog/lineage, inner
  base probabilities, calibrated outer probabilities, outcomes_by_id.csv,
  counts_by_label/venue/author_status.csv, transition_summary.csv, final recipes,
  primary/submissions per branch, verification.json.
- `RESULTS.md`, `DECISION.md`, `README.md` có cách tái tạo.

Verifier phải reload hai CatBoost binary models cùng scalers/onehots/calibrators,
rebuild final author features từ nguồn đã hash, so probabilities/selected IDs/
full order; kiểm tra596 IDs, finite nonnegative priorities và SHA incumbent cũ
`8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf`.

## 10. Ngân sách và trình tự thực thi

Cache277 feature frames cho20outer-inner+5outer+4final-inner+1full-final đã có.
Trước khi reuse phải audit lineage; không giả định mọi cached columns đều hợp lệ.
New fits tối đa: 60 CatBoost binary fits (2models x30partitions) và24 small
logistic fits (4calibrators x5outer+1final). Không GPU encoder training, không
search architecture/LR/depth/seed. Không refit các model đã qua protocol check.

Trình tự: audit/parser -> feature builders+tests -> runner/verifier -> khóa
protocol -> 20inner pairs+5outer pairs -> outer predictions/counts -> final4inner
pairs+1full pair -> calibrator/model reload -> reports. Chạy preflight trên một
inner training partition để đo thời gian/memory, không xem outer labels để đổi
recipe. Sau preflight báo ETA thực tế; lỗi tài nguyên thì ghi amendment trước
scoring, không âm thầm đổi tham số.

Mọi source, config và cache lineage phải khóa trước results mới. Sửa bug sau
scoring phải lưu bản cũ/log lỗi và ghi rõ phạm vi artifact chạy lại. Không mở
thêm trial nếu kết quả không như mong muốn.

## 11. Readiness khi lập plan

Đã kiểm tra availability của30 feature caches theo đúng tên partition và source
metadata3090 rows có authors_original/authors_verified/author_verified/source_url.
JSON spec parse hợp lệ. Đây chưa phải full lineage/parser audit; chưa có score,
binary fit, calibrator fit hoặc submission mới của lượt này.
