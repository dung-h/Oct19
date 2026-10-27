# Chẩn đoán Qwen supervised pilot — 2026-10-27

## Kết luận từ bằng chứng

Qwen học được liên hệ giữa văn bản và rating, nhưng sức phân biệt còn yếu.
Phần lớn cải thiện rất lớn so frozen đến từ sửa phân phối nhãn ban đầu.
Selection và thứ tự ưu tiên đều kém incumbent. Chỉ chỉnh cutoff hoặc một
ánh xạ đơn điệu lên read probability không thể sửa lựa chọn top-budget yếu.

Chưa tìm thấy lỗi kỹ thuật giải thích thất bại chất lượng: label mapping,
last-token indexing, gradients, frozen base, checkpoint hashes và reload
parity đều qua các kiểm tra đã chạy. Các kiểm tra này không chứng minh mọi
kernel hoặc mọi khía cạnh implementation đều hoàn hảo.

## 1. Argmax nhãn 1 và chọn nhiều bài là hai phép quyết định khác nhau

Fit prior của outer1_inner0 là
[.3701, .2088, .1660, .1352, .1198]. Argmax ra 1, trong khi tổng xác suất
nhãn 2–5 là .6299, vượt cutoff .55. Một classifier chỉ biết tỷ lệ nhãn cũng
có hai hành vi đó. Vì thế 996/1001 argmax ra 1 chưa đủ để kết luận training
hỏng; nó phù hợp với phân phối dự đoán phụ thuộc prior quá nhiều.

Phân phối trung bình trained của hai fold là:

- outer1_inner0: [.3776, .1922, .1751, .1502, .1049].
- outer1_inner1: [.3377, .2118, .1674, .1699, .1132].

Argmax chỉ dùng audit 5-class. Priority vẫn dùng read và expected gain.

## 2. Đối chứng prior làm thay đổi cách hiểu learned_signal

Mean theo đúng hai partition; các reference chỉ fit từ fit labels.

| Arm | 5-class CE | Read AUC | Read Brier | Native H |
|---|---:|---:|---:|---:|
| Frozen Qwen | 12.1467 | .5434 | .3648 | .5162 |
| Fit class prior, không có nội dung | 1.5234 | .5000 | .2333 | .5348 |
| Fit prior theo abstract present/missing | 1.5043 | .5406 | .2238 | .5483 |
| Trained Qwen | 1.4695 | .6522 | .2138 | .5644 |
| Incumbent | 1.2889 | .8075 | .1711 | .6476 |

Gain CE thật so class prior là .0538, nhỏ hơn nhiều so khoảng giảm 10.6771
so frozen. Learned_signal=true trong gate cũ là đúng theo định nghĩa đã khóa,
nhưng không đủ để nói đã học sở thích mạnh. Missingness reference là phân
tích hậu nghiệm, không phải challenger đã đăng ký trước; dùng add-one cố
định, không chọn smoothing hoặc cutoff bằng held-out.

Phân rã CE chính xác theo từng fold:

CE_5class = CE_read + fraction_read * CE_rating_given_read.

| Arm | Mean CE_read | Mean CE_rating_given_read |
|---|---:|---:|
| Fit class prior | .6595 | 1.3709 |
| Trained Qwen | .6170 | 1.3531 |
| Incumbent | .5122 | 1.2332 |

Khoảng 78.9% gain CE so prior nằm ở phần read. Khả năng phân biệt rating
trong nhóm 2–5 chỉ cải thiện ít. Đây là phân rã metric, không phải phép đo
nhân quả từng optimizer update.

## 3. Discrimination yếu là vấn đề lớn hơn một lệch cutoff đơn giản

Mean read AUC trained .6522, incumbent .8075. Trong nhóm đã có abstract,
read AUC trained chỉ .6083/.6379, thấp hơn AUC toàn fold .6407/.6638.
38 held-out papers thiếu abstract có 33 rating 1, nên availability là một
proxy dễ học. Missingness reference AUC .5406 không giải thích hết tín hiệu
của Qwen; model vẫn học thêm thông tin từ văn bản.

Model cũng có một phần calibration hợp lý:

- Fold 0: 289 bài trong bin read .65–.80 có mean prediction .7069 và read
  rate thật .7059. Global mean read .6224, true rate .6494.
- Fold 1: 353 bài trong cùng bin có mean prediction .7187 và read rate thật
  .6912. Global mean read .6623, true rate .6112.

Như vậy không có một lệch intercept cùng chiều giải thích cả hai fold.
Một khối bài lớn nhận xác suất gần nhau quanh prior; ngay cả khi average
calibration tốt, model vẫn khó tách các bài đáng đọc khỏi các bài không đọc.

Matched-budget giữ 648 bài như incumbent nhưng có FP194 thay vì FP135,
TP454 thay vì TP513; missed4/5 là 33/15 thay vì 8/3. Chỉ đổi cutoff hoặc áp
strictly monotonic map lên read không thay thứ tự top-k. Calibration có thể
đổi operating point native, nhưng chưa có bằng chứng nó khắc phục read
ordering hoặc high-rating misses này.

## 4. Ordinal ranking cũng yếu

Trong nhóm rating 2–5, Spearman(expected gain, rating):

- Qwen .2791/.1675; incumbent .5263/.4631.
- AUC phân biệt rating4–5 với rating2–3: Qwen .6609/.6069;
  incumbent .7956/.7763.

PC đầu chiếm 94.2%/92.2% biến thiên vector p của Qwen, incumbent khoảng
77.0%/77.2%. Mixture rating2–5 có độ biến thiên nhỏ hơn incumbent nhiều.
Đây là dấu hiệu output chủ yếu thay đổi theo một chiều. Gain/read correlation
cao tự nó không chứng minh lỗi: incumbent cũng có correlation cao. Bằng
chứng trực tiếp là ordinal metrics và ranking-only H .6253 < .6476.

## 5. Fit-only probes cho thấy model hoạt động nhưng học còn nông

Mỗi adapter: 128 fit indices lấy ngẫu nhiên không dùng nhãn, seed20261028.
Không có optimizer step; đây là kiểm tra fit, không phải OOF mới.

| Partition | Fit trained CE / prior CE | Fit read AUC | Held-out CE / AUC |
|---|---:|---:|---:|
| outer1_inner0 | 1.4433 / 1.5049 | .6816 | 1.4710 / .6407 |
| outer1_inner1 | 1.4318 / 1.5053 | .6911 | 1.4681 / .6638 |

Fit probabilities ghép đúng labels tốt hơn 200 phép hoán vị labels cố định
(median shuffled CE1.5527/1.5481; AUC .4987/.5055). Ngay khi chỉ giữ các fit
papers có abstract, read AUC vẫn .6699/.6589. Model thực sự dùng nội dung,
không chỉ missingness.

Trên 32 fit anchors mỗi adapter, bỏ abstract làm mean total-variation distance
của vector p tăng .2590/.2257; bỏ title chỉ .0653/.0500. Bỏ abstract cũng đổi
availability marker và đưa prompt ra ngoài phân phối thường gặp, nên không
được quy toàn bộ hiệu ứng đó cho semantic content của abstract.

112/112 tensor LoRA có gradient hữu hạn khác0; base không có gradient.
Adapter hashes đúng, LoRA B khác0 ở đủ 56 q/v modules; parity với 5 fit
probes đã lưu của mỗi adapter có sai số0.0.

Fit score cũng chưa mạnh và khoảng cách fit/held-out ở các probe này vừa
phải. Kết quả phù hợp với underlearning hoặc giới hạn tín hiệu/input;
không đủ để kết luận nguyên nhân riêng là thiếu epoch, LoRA rank hay head.
Các fit samples nhỏ và khác class proportions, nên không suy ra mức overfit
chính xác từ chênh lệch trên.

Trong training cũ, updates1–20 còn chủ yếu sửa prior: loss trung bình updates
1–5 là11.47/12.74, updates6–20 là4.94/4.49; từ update21 loss quanh1.5.
LM head cố định và q/v LoRA phải sửa digit prior rất lệch của base trước khi
học phân biệt tốt. Đây là cơ chế khả dĩ, cần đối chứng để xác nhận.

## 6. Các nghi vấn kỹ thuật đã kiểm tra

- Answer tokens16–20 mapping đúng1–5; CE dùng rating-1; logits ở assistant
  start. Installed Qwen2 source dùng last-token slice đúng khi logits_to_keep=1.
- Warning sliding-window: checkpoint đặt use_sliding_window=false; warning
  của Transformers chỉ kiểm tra sliding_window có giá trị. Window131072 cũng
  dài hơn nhiều max prompt677. Chưa thấy bằng chứng warning này gây failure.
- Raw row-sum errors tối đa khoảng1.23e-7. Sklearn warning đến từ sai số
  float32 được lưu sang float64; diagnostic CE chuẩn hóa hàng trước tính.
  Việc chuẩn hóa cũng có thể đổi tie cực nhỏ trong AUC, khoảng1e-5; metric
  pilot gốc trong source summary vẫn là kết quả authoritative.
- Fit probe đầu OOM trong PEFT khi upcast embedding/head sangFP32 rồi loader
  đưa lạiBF16. Giữ fit_probe_v1 và stderr. Với allocator expandable segments,
  smoke mới đạt, sau đó hai adapter đều đạt saved-probe parity0.0. Đây là
  mitigation bộ nhớ đã xác minh, không phải bằng chứng quality tăng.
- Runtime lock cũ không ghi allocator env; diagnostic protocol lock bổ sung
  allocator_environment và nguồn smoke. Không sửa fingerprint cũ.

## 7. Cách sửa cần kiểm chứng tiếp

Ưu tiên kiểm chứng việc học nội dung vượt prior, trước khi tuning policy.
Một đối chứng tập trung là giữ Qwen7B/revision, input và CE5class, rồi dùng
group-disjoint split nằm hoàn toàn trong fit để so:

1. q/v LoRA hiện tại với frozen digit projection.
2. Frozen Qwen +5 output-bias parameters để đo phần chỉ sửa digit prior.
3. Cùng q/v LoRA +5 bias parameters; bias khởi tạo/fit bằng inner-fit-only
   frozen logits, sau đó học cùng LoRA. Lưu raw5 logits vì softmax float32
   có thể mất độ phân giải ở xác suất rất nhỏ.

Đây là giả thuyết về initialization/parameterization, chưa chứng minh sẽ tăng
H. Khóa protocol và cùng ngân sách trước chạy; không chọn epoch/prompt/loss
bằng hai held-out sets đã xem. Nếu arm3 không hơn arm2 ngay trên fit-internal
out-of-group data, cần kiểm tra representation/input hoặc objective tiếp,
không tự động tăng epoch. Nếu discrimination đủ tốt mới mở calibration với
OOF sạch. Author/fusion cần protocol và cross-fitted lineage riêng.

Đã hoàn tất chẩn đoán thực nghiệm và mitigation OOM. Chưa có corrective
quality run được chứng minh tốt hơn, chưa thay incumbent hoặc submission.

## Artifacts và phạm vi

Source pilot: ../qwen_remote_v1/outputs/pilot_torch212_cu130.
Diagnostic output: outputs/diagnosis_v1.

- analysis.json, diagnostic_metrics.csv, read_reliability.csv, subgroups.csv.
- references_v2/reference_metrics.csv và loss_decomposition.csv.
- fit_probe_v1/status.json: failed OOM, giữ nguyên.
- runtime_expandable/smoke/COMPLETE.json: TECHNICAL_PASS, parity0.0.
- fit_probe_v2_expandable/summary.json, probabilities/sensitivity JSONL,
  membership và adapter_norms CSV: DIAGNOSTIC_COMPLETE.
- summary_v1/findings.json, derived_fit_metrics.csv.

Original source/config/input/checkpoint fingerprints được giữ nguyên. Hai
held-out folds vẫn exploratory. Không tune cutoff, refit2494, predict test,
hoặc upload Kaggle trong chẩn đoán này.
