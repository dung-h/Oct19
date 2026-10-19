# Kiểm tra ranking bổ sung khi giữ nguyên selection của incumbent

Trạng thái: **đã lập kế hoạch, chưa chạy thí nghiệm này**.

## 1. Câu hỏi cần trả lời

Với đúng tập bài mà incumbent CatBoost chọn, title SPECTER2 adapted có xếp các
bài đó tốt hơn không? Nếu phối hợp hai điểm ranking, lợi ích có giữ được khi
trọng số chỉ chọn bằng inner CV không?

Đây là classification kết hợp expected gain, đáp ứng yêu cầu môn học. Metric
theo phần Evaluation chi tiết trong `overview.md`: H = 2QR/(Q+R), coverage là
recall của Label > 1, không có thành phần title diversity. Mục tiêu nghiên cứu
là cải thiện H; không cam kết vượt 0.7.

## 2. Bằng chứng xuất phát và phạm vi

- Thí nghiệm title trước đã có 5 outer / 4 inner grouped folds, incumbent
  H=0.64028; adapted title và incumbent có cả inner/outer probabilities.
- Probability blend trước đạt H=0.64144, chỉ tăng khoảng 0.00117, recall giảm
  0.80324 xuống 0.75540, missed-high tăng 38 lên 55. Blend đó thay cả selection
  lẫn ranking; không trả lời riêng câu hỏi ranking.
- Pilot abstract sau đó dùng một calibration group mỗi outer. Không lấy số
  incumbent 0.63768 hoặc prediction từ pilot đó ghép vào phép đo 4-inner này.
- Code hiện tại đã tách P(read) và expected gain khi xuất priority. Phương pháp
  mới ở đây là thay nguồn ranking trong khi giữ nguyên nguồn selection, không
  gọi việc viết lại công thức priority là một kiến trúc mới.

Lượt này chỉ tái sử dụng prediction đã có. Không thêm learned stacker, abstract,
metadata mới, checkpoint, cutoff grid, seed, synthetic label hoặc encoder fit.
Kết quả không tự động kích hoạt một vòng train mới.

## 3. Nguồn dữ liệu và prediction chính xác

Nguồn chính là `experiments/specter2_preference/outputs` (gọi tắt là TITLE).
Nguồn incumbent probabilities là `experiments/claim_stability/outputs` (CLAIM).
Model/CSV incumbent cuối là `experiments/preference_augmentation/outputs` (INC).

| Thành phần | Artifact | Cách sử dụng |
|---|---|---|
| Dữ liệu/nhóm | `specter2_preference/run_preference.py::context()` | 2.494 nhãn train, 596 test; cùng nhóm trùng publication |
| Outer/inner membership | `specter2_v2/outputs/nested_splits.csv` | Giữ nguyên 5 outer, 4 inner trong từng outer train |
| Final inner membership | `INC/final_inner_splits.csv` | Giữ nguyên 4-fold toàn train |
| Incumbent inner/outer | `CLAIM/outer{f}_inner{k}_predictions.npz`, `CLAIM/outer{f}_validation_predictions.npz` | Dùng trường `reference`, xác minh fit/pred indices |
| Title inner | `TITLE/outer{f}_inner{k}_adapted.npz` | Chỉ arm adapted, checkpoint từ lựa chọn adapted cũ của outer đó |
| Title outer | `TITLE/outer{f}_validation_adapted.npz` | Đúng checkpoint adapted đã chọn bằng inner của outer đó |
| Lựa chọn cũ | `TITLE/outer{f}_choices.json`, `TITLE/final_choices.json` | Tái tính độc lập incumbent cutoff và adapted epoch trước khi dùng |
| Final inner probabilities | `CLAIM/final_inner{k}_predictions.npz`, `TITLE/final_inner{k}_adapted.npz` | Chọn trọng số cuối trên 4 inner folds |
| Final test probabilities | `INC/test_prediction_details.csv`, `TITLE/final_neural.npz` | Chỉ inference; dùng neural cache nếu manifest khớp adapted epoch cuối |
| Model cuối | `INC/final_classifier.cbm`, `TITLE/neural_model/` | Reload để kiểm tra cached final probabilities |
| Đối chứng đã công bố | `TITLE/fold_metrics.csv` | So lại đúng hàng family=incumbent; không dùng policy chọn nhiều model |

Nếu final neural manifest khác family/epoch cần dùng, dừng xuất candidate và ghi
blocker. Không thay bằng cache khác hoặc tự train thêm. Cache inner/outer thiếu
hoặc không khớp lineage cũng là lỗi dữ liệu đầu vào, không bỏ fold để tiếp tục.

## 4. Công thức khóa trước khi tính điểm

Với probability vector p có thứ tự nhãn 1..5:

```text
read_inc = 1 - p_inc[Label=1]
gain_inc = sum(p_inc[y] * (y-1)/4, y=1..5)
gain_title = sum(p_title[y] * (y-1)/4, y=1..5)

selected = read_inc > cutoff_inc
rank_w = (1-w)*gain_inc + w*gain_title
priority_w = 1 + 9*rank_w, nếu selected; ngược lại 0
```

`cutoff_inc` là cutoff incumbent đã chọn bằng mean 4-inner H trong primary cũ.
`epoch_title` là epoch của **adapted arm**, đã chọn bằng mean 4-inner H trong
primary cũ. Không dùng lựa chọn `neural_selection` nếu nó trỏ sang frozen.
Không dùng title cutoff trong lượt này. Không chọn lại epoch theo ranking score.

Phép affine 1+9*rank_w giữ đúng thứ tự của rank_w và đảm bảo mọi bài được selected
đều có priority dương, kể cả gain bằng 0. Không clip/round/rank-normalize điểm;
không thêm epsilon theo ID. Ties dùng ID dạng text tăng dần theo đề.

### Các nhánh cố định

| Nhánh | Trọng số title | Mục đích |
|---|---:|---|
| `incumbent_rank` | 0 | Baseline với selection/order của incumbent |
| `title_rank` | 1 | So sánh chính: thay riêng ranking |
| `half_rank` | 0.5 | Đối chứng phụ, trộn ranking cố định |
| `inner_selected_rank` | Chọn trong 0, 0.25, 0.5, 0.75, 1 | Quy trình có thể xuất candidate |

Chỉ ba trọng số cố định 0/0.5/1 được báo như các nhánh outer riêng. Hai trọng số
0.25/0.75 chỉ tham gia grid inner; không chọn trọng số thắng outer để báo làm
candidate. Có thể giữ raw predictions của mọi w để audit, nhưng bảng kết luận
phải phân biệt chúng với policy đã chọn trước.

## 5. Pha A — audit trước mọi score mới

1. Xác minh source labels không đổi và test labels không được đưa vào metric.
2. Xác minh các verification.json của nguồn; hash input, labels, splits, source
   scripts và **mọi cache sẽ đọc** vào protocol riêng của lượt này.
3. Kiểm tra ID order, class order, array shapes, fit/pred memberships, simplex,
   finite probabilities và group separation cho từng partition. Inner fit và
   inner validation đều thuộc outer train, không chứa outer validation group.
4. Replay lựa chọn incumbent cutoff và adapted epoch bằng đủ 4 inner folds;
   đối chiếu choices đã lưu với tolerance số thực 1e-12. Replay choices có thể
   đọc inner labels, nhưng không được đọc outer labels.
5. Với final, replay choices trên 4 inner folds toàn train. Kiểm tra manifest
   final neural là đúng adapted checkpoint, không suy từ tên file.
6. Khóa `experiment_spec.json` và source runner bằng hash trước khi tính score
   fixed-gate. Nếu sửa lỗi triển khai sau đó, lưu amendment và rerun toàn bộ
   artifact liên quan; không âm thầm thay recipe sau khi xem kết quả.

Điều kiện qua pha A: không có cache/membership/hash mismatch. Không retrain
encoder để chữa thiếu cache trong phạm vi lượt này.

## 6. Pha B — phép đo ranking và policy 5 outer / 4 inner

Thực hiện theo thứ tự dưới đây cho từng outer fold f=0..4:

1. Chỉ đọc inner predictions/labels để lấy cutoff_inc và epoch_title đã replay.
2. Với mỗi w trong grid, dùng **cùng incumbent gate** trên mỗi inner holdout;
   tính H riêng trên từng holdout, sau đó lấy mean 4 H. Không gộp mọi inner
   holdout thành một danh sách ranking rồi chấm một H.
3. Chọn w có mean inner H cao nhất. Nếu chênh trong 1e-12, ưu tiên w nhỏ hơn
   (w=0 ưu tiên incumbent). Lưu lựa chọn trước khi đọc outer labels.
4. Dùng incumbent và title outer probabilities đúng fit/checkpoint để tạo ba
   nhánh cố định và nhánh inner-selected. Không fit thêm model.
5. Chấm outer đúng một lần cho các nhánh đã khóa; lưu prediction theo ID và
   metric từng fold. Không dùng kết quả outer để đổi grid/cutoff/checkpoint.

Pha B luôn chạy đủ sau khi qua audit, không dùng kết quả diagnostic trên outer
để quyết định giữ/bỏ nhánh hoặc tạo grid mới. Có thể cho runner khóa choices của
cả năm folds trước khi bắt đầu đọc bất cứ outer labels nào.

So sánh chính: `title_rank - incumbent_rank`.
So sánh triển khai: `inner_selected_rank - incumbent_rank`.
`half_rank` là secondary comparison; không thay thế kết quả policy khi nó đẹp hơn.

### Bất biến bắt buộc

- Selected IDs của tất cả nhánh phải khớp incumbent **từng ID**, từng fold.
- TP/FP/FN/selected/missed-high/coverage khớp incumbent chính xác.
- `incumbent_rank` khớp thứ tự và metric baseline cũ, dù numeric priorities có
  thay affine scale. Target mean H khoảng 0.6402759969, đối chiếu actual artifact
  bằng tolerance 1e-10 thay vì chỉ kiểm tra số đã làm tròn trong báo cáo.
- Mọi delta H chỉ đến từ thay thứ tự/Q. Nếu recall đổi, đó là bug, không là kết quả.
- IDs giữ dạng text và sort tie đúng theo Evaluation.

## 7. Pha C — giải thích ranking failures

Báo cáo H/Q/R mỗi fold, mean H, SD giữa folds, paired delta H/Q và số folds tăng.
AUC(read) và logloss của base models chỉ là thông tin nguồn; không dùng AUC(read)
để suy ra model xếp đúng nhãn 5 trước nhãn 2. Không gán AUC xác suất cho rank_w.

Với cùng selected set, lưu vị trí trước/sau, Label, gain và discounted gain
contribution của từng ID. Tổng các contribution phải khớp Q numerator. Đối
chiếu theo Label 1..5 để biết ranking mới đưa bài nhãn cao lên hay bị các bài
nhãn thấp chiếm vị trí sớm. Có thể trình bày vài ví dụ từ diagnostic rows nhưng
không sửa prediction hoặc weight theo các ID/venue sau khi đọc labels.

Đọc `inner_search.csv` để phân biệt:

- Title rank không giúp ngay cả inner: ít bằng chứng về tín hiệu bổ sung.
- Title rank giúp inner nhưng thua outer: lựa chọn ranking thiếu ổn định.
- Title rank giúp outer nhưng inner chọn sai w: vấn đề selection; không tự sửa
  bằng cách dùng outer-optimal w.
- Inner-selected rank giúp ổn định: có bằng chứng để giữ một candidate phối hợp
  đơn giản, chưa có lý do tăng độ phức tạp bằng learned stacker.

Không kiểm định t/CI với giả định 5 folds độc lập. Fold trainings có overlap và
outer data đã được nghiên cứu nhiều lần; báo độ lớn/độ nhất quán thực tế, không
gọi kết quả là independent holdout hoặc bằng chứng đạt Private Kaggle 0.7.

## 8. Tiêu chí quyết định khóa trước kết quả mới

Quyết định chính dựa trên **inner_selected_rank**, không arm thắng outer.

- **Promising:** mean delta H >= 0.003 và tăng ít nhất 4/5 outer folds, mọi audit
  và bất biến pass. Đề xuất candidate riêng để người dùng cân nhắc; không tự upload.
- **Weak/mixed:** mean delta H > 0 nhưng không thỏa hai điều kiện trên. Giữ
  incumbent làm bản chính, lưu candidate experimental và mô tả mức tăng nhỏ.
- **Negative:** mean delta H <= 0. Dừng hướng thay ranking bằng hai tín hiệu này;
  không mở grid hoặc thêm learned stacker trong cùng lượt để cố đổi kết luận.
- **Invalid:** lỗi lineage/metric/invariant. Chưa đưa ra kết luận về model; sửa
  lỗi triển khai có log amendment rồi chạy lại phép đo liên quan.

Mốc 0.003 là ngưỡng lợi ích thực dụng cho quyết định nghiên cứu, không là ngưỡng
ý nghĩa thống kê hay quy định assignment. Nó được đặt cao hơn mức +0.00117 nhỏ
và không ổn định của blend trước, để tránh gọi lặp lại kết quả đó là đột phá.
Ngay cả promising cũng không chứng minh có thể đạt 0.7.

## 9. Pha D — final candidate và verification

Sau phép đo outer, luôn chọn lại w bằng 4 final-inner folds toàn train theo cùng
recipe đã khóa, không dùng w thắng outer. Dùng incumbent final cutoff đã replay
và title adapted final epoch đã replay. Xuất candidate experimental dù kết quả
weak/negative để tái tạo đầy đủ quy trình, nhưng decision phải ghi rõ không
promote. Nếu final w=0, ghi rõ candidate tương đương incumbent về selection/order.

Dùng cached final probabilities, reload model cũ để kiểm chứng (có thể cần GPU
cho SPECTER2 inference). Không fit thêm model. Source test có hai duplicate
publication groups giao train/test; giữ đúng dataset và ghi giới hạn, không gọi
test là group-independent. Không đọc hay suy đoán test labels.

Verifier độc lập cần kiểm tra:

1. Tính lại metric bằng implementation riêng: empty list, ties/text IDs, gain,
   weighted mean Q, recall, H. Có small synthetic cases với kết quả biết trước.
2. Replay inner weight choices cho 5 outer + final; tolerance 1e-12/tie rule
   đồng nhất. Kiểm tra choices được ghi trước outer scoring.
3. Tính lại mọi outer metrics từ saved probabilities và priority CSV; mean H là
   mean của 5 fold H, không harmonic mean của mean Q/R.
4. Assert invariants selected IDs/TP/FP/FN/highFN/R và baseline order/metric.
5. Reload CatBoost + neural adapter/head; probabilities tương ứng khớp cache
   với tolerance kế thừa verifier nguồn; recompute final priorities/order.
6. CSV đúng hai cột id,priority, 596 IDs duy nhất đủ template, priorities finite
   nonnegative, file theo template order. Incumbent SHA-256 vẫn bằng
   `8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf`.

## 10. Artifact và thứ tự triển khai

Tất cả file mới nằm trong `experiments/fixed_gate_ranking/`, giữ nguồn cũ.
Các tên dưới đây là **deliverables cần viết**, chưa phải script đã tồn tại.

| File | Chức năng |
|---|---|
| `experiment_spec.json` | Cấu hình khóa, nguồn, grid, tie rule, decision rule |
| `run_ranking.py` | Audit, inner choices, outer scoring, final priorities; resumable theo protocol |
| `verify_ranking.py` | Metric và selection/order replay độc lập, final model reload |
| `write_report.py` | Summary, paired effects, rank diagnostics, decision |
| `tests/test_metric.py` | Edge cases metric + invariance khi đổi rank, chỉ các test có ý nghĩa |
| `outputs/protocol.json` | Source/input/cache hashes, versions, recipe |
| `outputs/lineage.csv` | Từng cache, fit/pred membership/hash/checkpoint nguồn |
| `outputs/outer_choices.json`, `inner_search.csv` | Lựa chọn mỗi outer, toàn grid inner |
| `outputs/fold_metrics.csv`, `summary.csv`, `paired_effects.csv` | Ba fixed branches và inner-selected policy |
| `outputs/oof_predictions.csv`, `rank_changes.csv`, `rank_by_label.csv` | Prediction và lý do Q tăng/giảm |
| `outputs/final_search.csv`, `final_recipe.json`, `test_predictions.csv` | Recipe final cùng provenance model |
| `outputs/submission_candidate.csv` | Candidate riêng; không thay incumbent |
| `outputs/verification.json` | PASS chỉ sau khi mọi kiểm tra hoàn tất |
| `RESULTS.md`, `DECISION.md` | Báo cáo tiếng Việt và quyết định theo rule đã khóa |

Thứ tự công việc: viết runner/audit -> verifier và metric tests -> khóa protocol
-> audit -> khóa 5 outer choices từ inner -> outer scoring -> final selection/
CSV -> reload/verification -> report/decision. Verifier/report có thể sửa lỗi
trình bày hoặc tolerance với amendment; không thay recipe để cải thiện điểm.

Ngân sách: 0 encoder/head training fits, không tải model mới. Phép toán trên
cache chạy CPU; GPU chỉ dùng nếu cần reload/inference neural cuối. Trước chạy
phải kiểm tra dependency và cache đủ, rồi báo ETA theo preflight thực tế thay
vì hứa thời gian train. Không submit Kaggle trong phạm vi lượt này.

## 11. Readiness đã kiểm tra khi viết plan

- `experiment_spec.json` parse hợp lệ, grid và số new training fits đúng plan.
- 58 cache inner/outer cần dùng có sẵn, chứa đúng adapted checkpoint được chọn
  trong source choices; fit/pred indices khớp giữa neural và incumbent. Đây chỉ
  là kiểm tra availability/membership, chưa thay thế full lineage/protocol audit.
- Manifest `TITLE/neural_model/manifest.json` là arm adapted, epochs=[8], khớp
  adapted choice của final. Final source probabilities và classifier có sẵn.
- Chưa tính bất kỳ H/ranking effect mới nào trong lúc viết plan; chưa tạo runner,
  train model, xuất candidate hoặc submit Kaggle cho thí nghiệm này.
