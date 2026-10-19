# Plan: kiểm tra TabPFN rồi mới đầu tư cross-encoder theo sở thích

Ngày lập: 2026-10-17. Trạng thái: DESIGN, chưa cài package, chưa chạy model.
Kế hoạch chính là thí nghiệm A; thí nghiệm B là thiết kế giai đoạn tiếp theo.
Code/package/checkpoint phải được khóa trong protocol trước khi xem score mới.

## 1. Câu hỏi và mục tiêu

Incumbent có mean outer H=0.6402759969, Q=0.5324996394, R=0.8032377507;
TP=1278, FP=338, FN=313, missed Label4=33, missed Label5=5 trên 2494 rows.

Ở R hiện tại, H=0.7 cần Q xấp xỉ 0.6203. Oracle sắp xếp hoàn hảo trên đúng
tập incumbent đang chọn chỉ cho mean H=0.660783. Vì thế mục tiêu nghiên cứu
là cải thiện selection, đồng thời giữ ranking hợp lý. Không đặt giả định
rằng thay model sẽ đạt 0.7 hoặc lấy oracle làm mức tăng dự kiến.

- A: cùng 277 feature, TabPFN có cải thiện khả năng học so với CatBoost không?
- B: nếu A không đem lại bước tiến đủ rõ, cho model đọc đồng thời query và
  các bài tham chiếu có nhãn có giúp phân biệt sở thích bên trong chủ đề không?

Assignment dùng classification substantively: target thật Label1..5, không
gán thêm nhãn; xuất id,priority cho đủ 596 test IDs khi có candidate phù hợp.
Metric theo phần Evaluation chi tiết trong overview.md: Q weighted gain,
R recall của Label>1, H=2QR/(Q+R), không phải F1 chuẩn hay title diversity.

## 2. Phạm vi A và các điều khóa trước

Chỉ thay classifier; không thêm abstract, graph, feature, encoder, source model
hay một vòng ensemble trong A. Không dùng danh sách lỗi OOF làm train weights.

| Thành phần | Cố định |
|---|---|
| Dữ liệu | 2494 train labels gốc, 596 test không nhãn |
| Feature | cùng 277 cột incumbent trong từng partition |
| Split | 5 grouped outer, 4 grouped inner đã có |
| CatBoost | 300 iterations, depth4, lr0.05, L2=8, seed20261018 |
| TabPFN | explicit V2 classifier checkpoint, n_estimators=4, seed20261018, GPU |
| Target | multiclass Label1..5, map class order rõ ràng |
| P(read) | 1-P(Label1) |
| Expected gain | sum P(Label=c)*(c-1)/4 |
| Threshold grid | 0.00,0.05,...,0.90, strict P(read)>t |
| Tie threshold | H bằng nhau trong 1e-12: chọn threshold thấp hơn |
| Priority | 0 nếu bỏ; nếu nhận: 1+9*expected_gain |
| Search | không HPO, không chọn model seed, không đổi checkpoint theo score |

V2 được chọn explicit để tránh default checkpoint/package tự đổi, và để bám
nghiên cứu về small tabular data. Compatibility, license thực tế của weights,
package version, checkpoint revision/SHA và các mặc định inference phải được
ghi trước run. Không suy ra license weights từ license Python code. Nếu V2
không dùng được thì ghi technical blocker, không tự đổi sang V3/V3.5 giữa run.

## 3. P0 — preflight môi trường và tài nguyên

Đã đọc môi trường: RTX4060 Laptop 8 GiB; Torch2.1.2+cu121 trong venv hiện có;
TabPFN chưa cài. Không coi cấu hình này mặc nhiên tương thích package mới.

1. Dùng venv riêng cho thí nghiệm; giữ nguyên env của incumbent.
2. Cài bộ dependency tương thích rồi lưu exact versions/lockfile; kiểm tra
   CUDA và tải đúng checkpoint V2, kiểm tra điều khoản weights.
3. Smoke test bằng dữ liệu synthetic 5 classes, sau đó benchmark synthetic
   1500 rows x277 features và 64 query rows. Không dùng score outer để chọn
   cấu hình. Đo thời gian, peak GPU memory và train-context size.
4. Khóa n_estimators=4. Nếu OOM do query batch, chỉ giảm batch64→32→16,
   ghi resource fallback và kiểm tra probability không đổi quá tolerance.
   Không giảm feature/đổi model/đổi ensemble âm thầm. Nếu vẫn OOM: stop A
   với trạng thái resource-blocked, không kết luận predictive failure.
5. Chốt batch/device/precision có hỗ trợ, versions/checkpoint, và ước lượng
   tổng thời gian từ benchmark trước khi tạo protocol.json.

Ngân sách A: tối đa 4 GPU-hours cho run chính; đây là giới hạn tài nguyên,
không phải lời hứa runtime. Hết budget thì checkpoint và báo INCOMPLETE,
không chọn subset fold đẹp hoặc đánh giá partial run như kết quả hoàn chỉnh.

## 4. P1 — feature contract và provenance

Nguồn chính: preference_augmentation/outputs/{partition}_features.joblib.
Mỗi pack có fit/pred indices, a/b frames, recipe và history lineage.

- Assert đúng ID/order, 277 columns, dtype, finite values, class counts,
  train/validation publication groups rời nhau, labels gốc nguyên vẹn.
- SVD/vocabulary/Ridge chỉ fit trong fit partition tương ứng. History của
  train row được cross-fit theo publication groups; query-held-out labels
  không được dùng làm reference hay fit feature transform.
- Kiểm tra input dependencies của upstream builder; hash cả imported local
  modules được dùng, raw data, splits, texts/embeddings/feature packs.
- Chỉ tái dùng cache có fit/pred/lineage/hash hợp lệ; nếu không, rebuild từ
  upstream builder theo cùng recipe, không patch số feature sau khi xem score.
- 276 số giữ nguyên, không standardize/SVD/feature-select thêm bên ngoài.
- Venue giữ semantics categorical: TabPFN nhận mã category fit trên fit rows,
  đánh dấu categorical column explicit; unknown venue thành missing theo API.
  CatBoost vẫn nhận chuỗi category gốc. Mapping lưu trong manifest. Đây là
  điều chỉnh representation bắt buộc cho API, không thêm thông tin.
- Loại ID, split, Label khỏi X; ID chỉ dùng alignment và tie-break metric.

Phải replay CatBoost trên 5 outer và 20 inner pack, so raw probabilities
với cache reference. Mục đích là chứng minh A dùng đúng cùng input contract.

## 5. P2 — đánh giá và chọn threshold không có meta layer

Với mỗi outer fold f:

1. Cách ly outer validation V ngay từ đầu. U là outer train.
2. Chia U theo đúng bốn inner group partitions U0..U3.
3. Với mỗi k, fit mọi feature có học từ data và cả classifier chỉ trên U\Uk;
   dự đoán Uk. Fit độc lập CatBoost/TabPFN trên cùng partition inputs.
4. Mỗi row trong U có đúng một source probability vector OOF. Không fit
   meta-logistic/scaler probability trên bảng này, nên không có bước cross-fit
   meta trên source OOF chung gây phụ thuộc inner-held-out như lượt trước.
5. Chọn threshold riêng cho mỗi classifier bằng mean H của bốn inner folds,
   dùng expected gain của chính classifier để rank. Inner H chỉ để chọn
   policy, không báo như generalization score không thiên lệch.
6. Refit đúng recipe trên toàn U, tạo features/predictions cho V, áp threshold
   đã chọn. Không đọc labels V cho đến khi hoàn tất inference/policy freeze.
7. Ghi tất cả 5 outer folds. Không chỉnh recipe sau fold đầu.

Không fit Platt/isotonic trong A: calibration ở đây là phần chọn policy từ
inner predictions. Reliability/Brier dùng để chẩn đoán, không tự hứa calibrated
probabilities. Phân bố probability có thể đổi khi train size tăng; ghi shift
inner→outer trong report, không sửa outer threshold theo shift đã thấy nhãn.

Các outer folds đã được xem nhiều lần trong nghiên cứu trước. Đổi seed hay
chia lại những labels đã xem không tạo holdout hoàn toàn mới. Báo cáo kết
quả là exploratory; Kaggle test labels vẫn không được truy cập.

## 6. Năm policy cần báo cáo

| Arm | Gate | Ranking | Vai trò |
|---|---|---|---|
| C0 deployed | incumbent P(read)>0.55 | incumbent gain | mốc submission hiện tại |
| C1 matched | incumbent, threshold chọn inner | incumbent gain | đối chứng được cùng budget chọn policy |
| T1 primary | TabPFN, threshold chọn inner | TabPFN gain | candidate duy nhất của A |
| T2 diagnostic | gate và threshold T1 | incumbent gain | cô lập tác động thay thứ tự trên tập T1 |
| T3 diagnostic | gate C0 | TabPFN gain | cô lập ranking trên tập incumbent |

Không chọn T2/T3 để promotion theo outer score. T1−C1 là đối chiếu hai
classifier với cùng thủ tục chọn policy; T1−C0 trả lời có hơn incumbent deployed
không. T1−T2, T3−C0 giúp hiểu phần ranking, không thêm một vòng tuning.

## 7. P3 — report bắt buộc, tách representation và policy

### 7.1 Kết quả có thể triển khai theo protocol

Mean của 5 fold H/Q/R và từng fold riêng; pooled counts accepted/dropped,
TP/FP/FN/TN, missed từng Label2/3/4/5; error transitions với C0.
Read AUC, multiclass logloss, Brier, reliability bins và phân bố read scores.
Ghi riêng ngoài metric chính: lỗi theo venue, abstract availability và
consensus-FP cohort đã được định danh từ nghiên cứu cũ.

### 7.2 Chẩn đoán score — không dùng chọn policy

- FPR tại recall mục tiêu cố định 0.80, lấy trên ROC outer để so phân biệt.
  Đây là diagnostic có dùng nhãn outer, không phải threshold triển khai.
- Top read scores với cùng số accepted của C0 trong từng fold: so FP/FN và
  high-label misses. Đây cũng là diagnostic, không copy K này vào submission.
- So T1/T2 để xác định ranking có làm mất lợi ích của gate không.
- Nếu AUC tăng mà H không tăng: ghi policy/calibration/ranking mismatch là
  giả thuyết; không kết luận feature không có tín hiệu.

### 7.3 Bất định

Báo paired delta theo 5 fold và group-level bootstrap cố định 2000 resamples
seed20261018 từ saved OOF, tính lại metric trong từng outer fold, rồi mean.
Khoảng bootstrap chỉ là mô tả độ nhạy của saved predictions, không tính hết
training variance hoặc adaptive model selection; không tuyên bố independent
significance. Không chọn seed tốt nhất, không xem xét riêng fold thắng.

## 8. P4 — verifier phải thực sự kiểm tra điều đã tuyên bố

1. Metric reference viết độc lập, không import hàm production metrics; test
   empty list, ties ID dạng text, all Label1, priorities0, perfect simple list.
2. Từ saved raw probabilities tính lại gate, priority, order, H/Q/R/counts
   và inner threshold choice cho toàn bộ arms.
3. Reload TabPFN recipe/checkpoint/train context và CatBoost models, predict
   lại held-out rows; kiểm tra classes order, probability finite/tổng1 và
   tolerance công bố trước: probability atol1e-5/rtol1e-5. Gate/order phải
   khớp; nếu gần threshold/ties làm khác order thì báo fail, không bỏ qua.
4. Perturbation thật: tạo context copy, thay toàn bộ held-out query labels
   thành1 rồi thành5; rebuild feature pipeline và predict lại từ train context
   cố định, không sửa nhãn trong manifest gốc. So feature arrays/predictions
   với bản gốc. Không được chỉ so một feature array với chính nó.
5. Kiểm tra group lineage/history exclusion và hash trước/sau run.
6. Final candidate nếu có: kiểm tra 596 IDs exact, nonnegative finite priorities,
   reload full-train inference và kiểm tra selection/order.

Verifier chỉ ghi pass cho từng hạng mục thực sự đã chạy; replay không được
gọi là bằng chứng loại trừ mọi loại leakage.

## 9. Quyết định sau A

Không quay lại sửa các ngưỡng này sau khi thấy T1 outer results.

| Kết quả T1 | Quyết định |
|---|---|
| Verifier fail/run incomplete | chưa có kết luận model, xử lý technical issue và ghi provenance |
| H tăng >=0.005 so C0 và C1, thắng cả hai ở >=3/5 fold | PROMISING, tạo experimental candidate riêng |
| H tăng nhưng chưa đủ điều kiện trên | SMALL_OR_UNSTABLE_GAIN, lưu kết quả, giữ incumbent |
| H không tăng, score diagnostics tốt hơn | POLICY_LIMITATION_HYPOTHESIS, chưa mở HPO/blend; dùng để thiết kế protocol kế tiếp |
| H và score diagnostics đều không tốt hơn | NO_GAIN_THIS_RECIPE; dừng TabPFN recipe này |
| Mean H>=0.7 | ghi đạt mục tiêu trên exploratory OOF; không suy thành Private Kaggle>=0.7 |

Mốc0.005 là ngưỡng thực dụng tự chọn để đầu tư tiếp, không phải yêu cầu môn
học hoặc ngưỡng ý nghĩa thống kê. FP/FN/high-label losses là kết quả bắt buộc
để diễn giải, không ép tất cả cùng giảm khi metric chính là H. Candidate nào
làm mất thêm Label4/5 phải được nêu rõ trong decision, kể cả H tăng.

Nếu PROMISING: dùng đúng 4 final-inner folds để chọn t cho T1, refit tất cả
2494 labels, predict596 và tạo submission_candidate.csv. Không thay incumbent
hoặc tự upload Kaggle. Không chọn arm/checkpoint mới bằng outer results.

## 10. Deliverables và ngân sách A

Implementation dự kiến, chưa có các script này ở thời điểm viết plan:

- environment/lockfile, preflight.json, protocol.json, source_manifest.json.
- input_adapter.py: feature/venue contract và cache lineage.
- run.py: audit → nested outer → report → final có điều kiện.
- verify.py và metric_reference.py: metric/model/feature/perturbation checks.
- prediction caches: probabilities, IDs/order, fit/pred groups và recipe hashes.
- inner_thresholds.csv, fold_metrics.csv, outer_predictions.csv, calibration.csv,
  score_diagnostics.csv, transitions.csv, bootstrap_summary.json.
- RESULTS.md, DECISION.md, verification.json; final CSV chỉ có khi phù hợp.

Core run gồm25 TabPFN fit/context configurations (20 inner+5 outer) và25
CatBoost parity fits. n_estimators4 là ensemble nội bộ, phải ghi thêm thời gian
inference/context và VRAM, không gọi 25 fits là toàn bộ compute cost. Final nếu
qua criteria thêm5 configurations mỗi classifier nếu chạy cả control (4 inner
+1 full-train); chỉ T1 mới sinh candidate. Verifier replay là chi phí riêng.

## 11. B — thiết kế có điều kiện cho cross-encoder theo lịch sử sở thích

Không triển khai B đồng thời với A. Đây là blueprint; trước B phải có PLAN_B
và protocol riêng chốt checkpoint, token allocation, seed và pilot budget
trên cơ sở resource preflight, không dựa vào score các checkpoint trên outer.

### Câu hỏi khác A

Thay vì encode paper độc lập rồi đo khoảng cách, cho model đọc query cùng
những ví dụ preference gần chủ đề để học quan hệ problem/assumption/method/
contribution. Đây là giả thuyết, không phải giải thích đã biết của nhãn thật.

### Input và references

- Query: title+abstract hiện có, venue/year; không có query Label hoặc ID.
- References:3 Label1 và3 Label2..5 gần query theo frozen retrieval đã có,
  mỗi publication group tối đa1, chỉ trong fitting/reference pool.
- Thử cùng-venue trước; nếu thiếu một phía thì bổ sung toàn pool bằng cùng
  similarity rule, không viết ngoại lệ theo paper. Tie retrieval dùng ID chỉ
  cho deterministic order, không đưa ID vào encoder.
- Giữ nhãn reference thật1..5 để không coi Label2 tương đương Label5.
- Reference selection của train query phải loại query và toàn group. Với
  validation/test, pool chỉ là train; không xây profile từ held-out labels.
- Ngân sách ban đầu: query~160 tokens và6 reference blocks~48 tokens/block,
  cộng nhãn/separators trong max512; audit truncation/thiếu abstract. Số này
  là thiết kế khởi điểm phải qua tokenizer preflight trước khi protocol B lock.

### Ba đối chứng tối thiểu

1. Query-only classifier với cùng pretrained backbone, trainable modules,
   loss, token ceiling và training schedule.
2. Retrieval baseline: cùng6 references, dùng similarity+nhãn reference qua
   classifier nhẹ; không cho cross-attention giữa các paper.
3. Query+references cross-encoder: đọc đồng thời toàn bộ text blocks.

Query-only và contextual branch khác lượng text; report token counts, không
gọi là đối chứng hoàn hảo chỉ khác attention. Không dùng semantic-similarity
reranker zero-shot trực tiếp như xác suất sở thích.

### Training/evaluation

Target query vẫn là5 classes, cross-entropy; inference lấy P(read)/gain như A.
Khởi điểm dùng LoRA+classification head trên backbone text pretrained phù hợp
English scientific text; giới hạn GPU8GiB, max512, mixed precision nếu hỗ trợ,
effective batch16 và3 epochs cố định. Checkpoint/model/version cụ thể phải
chốt trước score mới, không chọn chỉ bằng tên kiến trúc.

Pilot: chỉ inner training/calibration của outer0, không đọc outer0 score để
quyết định token allocation/checkpoint. Điều kiện lên full CV: run/gradient/
reference isolation pass, hoàn thành epochs và resource budget khả thi;
không chọn chỉ vì một calibration score đẹp. Full CV giữ5 outer/4 inner;
references của mỗi inner validation chỉ đến từ đúng inner train. Chọn
threshold từ inner predictions, không dùng outer để mining hard examples.

Report chính: H/Q/R và FPR tại recall0.80; transitions của Label4/5, lỗi
consensus và effect contextual−query_only cùng contextual−retrieval. Những
cohort đã xem trước chỉ là diagnostics, không phải test benchmark mới.

### Điểm dừng B

Nếu contextual branch không hơn query-only/retrieval trên protocol khóa,
dừng hypothesis này ở recipe đã thử. Không tiếp tục thêm prompt/checkpoint
theo các outer examples đã biết. Khi đó đánh giá provenance/coverage của
nguồn thông tin mới trước khi đầu tư collect full-text/graph/author history.

## 12. Nguồn và giới hạn suy luận

- [Nghiên cứu TabPFN, Nature2025](https://www.nature.com/articles/s41586-024-08328-6):
  cơ sở xem xét một classifier pretrained cho small tabular data, không phải
  bằng chứng performance của assignment này.
- [Official TabPFN repository](https://github.com/PriorLabs/TabPFN): version
  selection, môi trường/GPU và khác biệt code/weights license. API phải được
  kiểm tra lại theo đúng package đã pin khi implementation.
- [End-to-end Training for Recommendation with Language-based User Profiles](https://arxiv.org/abs/2410.18870):
  hướng tham khảo về học biểu diễn sở thích bằng ngôn ngữ. Thiết kế một-user
  scientific-paper cross-encoder ở B là đề xuất của dự án, không phải một
  kết quả được paper này bảo đảm.
