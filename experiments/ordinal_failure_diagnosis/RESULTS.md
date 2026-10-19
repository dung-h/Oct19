# Vì sao ordinal kém hơn — chẩn đoán bằng đối chứng

**Phần tụt H tập trung ở chọn/bỏ bài; có ảnh hưởng thực nghiệm từ training defaults
và tương tác giữa ordinal với nhóm feature láng giềng. Chưa có bằng chứng cleaning
text/đọc dữ liệu bị lỗi. PAV không làm tụt H trong lượt này.** Giữ incumbent làm
bản chính; các sửa đổi mới vẫn là thí nghiệm exploratory.

## 1. Khoanh chính xác phần H bị mất

Đổi chéo read gate và gain ranker từ hai model, giữ nguyên cutoff đã chọn bằng
inner CV. Shapley accounting trung bình5folds:

- Read gate ordinal đóng góp **-0.00584 H**.
- Gain ranker ordinal đóng góp **+0.00091 H**.
- Tổng: **-0.00493 H**.

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

| parameter                  | multiclass   | ordinal      |
|:---------------------------|:-------------|:-------------|
| bootstrap_type             | Bayesian     | MVS          |
| subsample                  |              | 0.8000000119 |
| bagging_temperature        | 1            |              |
| leaf_estimation_iterations | 1            | 10           |

CTR target border count cũng đổi4→1 theo5-class/binary target; không ép4borders
cho binary. Cấu trúc cây cũng được học riêng cho4tasks thay vì chia sẻ splits giữa
5classes. Vì vậy không thể quy mọi chênh lệch cho tên loss hoặc chất lượng feature.

Đối chứng mới giữ277features/depth4/L2=8, chỉ khóa Bayesian/bagging1 và leaf updates1:

| family              |       H |       Q |       R |     AUC |   train_AUC |   AUC_gap |   FP |   FN |   high_FN |
|:--------------------|--------:|--------:|--------:|--------:|------------:|----------:|-----:|-----:|----------:|
| multiclass          | 0.64028 | 0.53250 | 0.80324 | 0.79946 |     0.87416 |   0.07470 |  338 |  313 |        38 |
| ordinal             | 0.63841 | 0.53493 | 0.79396 | 0.79873 |     0.89331 |   0.09458 |  331 |  327 |        44 |
| selection_procedure | 0.63829 | 0.53236 | 0.79981 | 0.79871 |     0.89042 |   0.09172 |  341 |  318 |        42 |

Ordinal H **0.63535→0.63841**; train AUC
**0.96438→0.89331**,
validation AUC **0.79874→0.79873**.
Gói mặc định ảnh hưởng rõ độ fit trên train và sửa một phần H, không tạo ra cải
thiện tương ứng về khả năng xếp hạng đọc/không đọc trên validation. Chưa tách riêng
bootstrap với leaf updates, cũng chưa loại hết khác biệt giữa hai formulation.

## 3. Feature và độ phức tạp: hai can thiệp giữ các phần khác cố định

`regularized`:depth3/L2=32 thaydepth4/L2=8; `without_neighbors`:bỏ150neighbor
fields, giữ35base +92content. `ridge_rating` vẫn còn và vẫn cross-fit. Mọi threshold
chọn trên4inner; score dưới là mean5outer. MC control dùng raw caches matched.

| family                |       H |       Q |       R |     AUC |   train_AUC |   AUC_gap |   FP |   FN |   high_FN |
|:----------------------|--------:|--------:|--------:|--------:|------------:|----------:|-----:|-----:|----------:|
| MC_full               | 0.64028 | 0.53250 | 0.80324 | 0.79946 |     0.87416 |   0.07470 |  338 |  313 |        38 |
| ORD_full              | 0.63535 | 0.53260 | 0.78886 | 0.79874 |     0.96438 |   0.16565 |  340 |  337 |        49 |
| MC_regularized        | 0.63863 | 0.52933 | 0.80667 | 0.79864 |     0.83680 |   0.03816 |  348 |  307 |        37 |
| ORD_regularized       | 0.63822 | 0.52831 | 0.80818 | 0.79850 |     0.90871 |   0.11021 |  354 |  304 |        38 |
| MC_without_neighbors  | 0.63778 | 0.52448 | 0.81617 | 0.80123 |   nan       | nan       |  362 |  292 |        35 |
| ORD_without_neighbors | 0.63946 | 0.53851 | 0.79119 | 0.80264 |     0.93204 |   0.12940 |  319 |  333 |        46 |
| selection_procedure   | 0.63818 | 0.53992 | 0.78055 | 0.80316 |     0.89950 |   0.09172 |  312 |  349 |        48 |

- Regularization sửa ordinal **+0.00288 H**,
  train AUC giảm0.96438→0.90871.
  Giảm train–validation gap chưa tự chứng minh nguyên nhân duy nhất hoặc model tốt.
- Bỏ neighbor sửa ordinal **+0.00411 H**,
  trong khi multiclass đổi **-0.00250 H**.
  Interaction của phép bỏ nhóm feature giữa hai formulation là **+0.00661 H**.
  Đây là bằng chứng về tương tác feature×model ở cấu hình này, không phải nhóm
  feature đó vô ích cho mọi model hoặc toàn bộ preprocessing sai.
- Các mức tăng thuộc những can thiệp riêng, **không cộng chúng lại** để dự đoán
  hiệu quả của cấu hình kết hợp chưa được chạy.

## 4. Preprocessing: điều gì đã kiểm tra, điều gì chưa biết

Hai model gốc nhận **cùng các file feature đã hash**, cùng column order và cùng
folds. Audit không có numericNaN/inf (0 entries).
Nhãn nguồn, grouping, inherited crossfits, rebuild test features và model reload
đều được kiểm tra. Điều này chưa chứng minh representation đủ thông tin ngữ nghĩa.

Classifier nhận title TF-IDF SVD32 +venue/year +Ridge score,92lexical content
flags và150neighbor statistics. Abstract đi vào qua flags/similarity-derived
statistics; đây không phải một encoder abstract được học trực tiếp theo sở thích.
SVD giữ khoảng **13.2% TF-IDF variance**;
con số đó **không** là phần trăm thông tin sở thích được giữ hoặc bị mất.

History train rows dùng reference subsets của crossfit, validation rows dùng toàn
outer train. Có drift về references/support trong `history_size_drift.csv`; training
importance của nhóm count không chứng minh nó là cột gây hại. Phép bỏ toàn bộ
neighbor mới kiểm tra hiệu ứng cả nhóm, chưa tách reference-size drift khỏi noise,
redundancy hay tác dụng regularization do bỏ150fields.

## 5. Vì sao logloss tăng mạnh mà PAV không gây mất H?

PAV tạo xác suất lớp thật bằng0 ở **16**bài (8label2,5label3,3label4).
Những dòng này đóng góp khoảng **0.21902** vào
pooled Δlogloss **0.26093**. Đây là phân rã pooled per-item,
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
{
  "family": "MC_full",
  "threshold": 0.55,
  "inner_score": 0.6386873823729493
}
```
Default-matched policy final:
```json
{
  "family": "multiclass",
  "threshold": 0.55,
  "inner_score": 0.6386873823729493
}
```

Mỗi candidate theo inner selection của chính policy, không lấy family thắng outer
để tạo model cuối. Policy scores báo riêng trong hai bảng trên. Caches/models/CSV
đều ở `outputs`, default-matched ở `outputs/defaults`. Bản incumbent giữ nguyên;
chưa gửi Kaggle. Dataset outer được nghiên cứu lặp lại nên đây là bằng chứng
exploratory ở mức can thiệp cấu hình, không là kiểm định nhân quả độc lập.

Verify **PASSED**:360new binary fits,50outer metric rows,60inner
choices,15final model reloads, rebuilt features, cùng priority/rank và nhãn nguồn.
Xem `outputs/verification.json`, `control_summary.csv`, `intervention_paired_effects.csv`,
`mechanisms/default_parameter_differences.json`, `mechanisms/H_decomposition.csv`.

Tham khảo: [CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning).

Đã kiểm tra tiếp các sửa đổi kết hợp ở [ordinal_combined_fixes](../ordinal_combined_fixes/DECISION.md): khóa mặc định + bỏ láng giềng H=0.637296; thêm regularization H=0.637555; ordinal chuẩn hóa cột đếm H=0.638909. Cả ba đều thấp hơn multiclass hiện tại H=0.640276. Vì vậy các lợi ích riêng lẻ phía trên không được cộng dồn hoặc coi là bảo đảm tăng điểm. Kết quả và kiểm tra độc lập đầy đủ nằm trong thư mục thí nghiệm tiếp theo.
