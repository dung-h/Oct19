# Plan: head phân loại 5 nhãn cho Qwen

Trạng thái: **plan, chưa implement hoặc chạy quality jobs**. Khi implement,
chuyển các lựa chọn dưới đây thành config/protocol, kiểm tra rồi khóa hashes
trước lần fit đầu. Đây là diagnostic tiếp theo của `qwen_learning_check_v1`,
không phải protocol confirmation hay final submission.

## 1. Câu hỏi và lý do

Liệu một head 5 nhãn có regularization, được học trên frozen features trước
rồi học cùng LoRA, có khai thác nội dung tốt hơn digit projection cố định?

Bằng chứng từ check trước trên cùng validation 298 bài:

- Digit-head LoRA epoch3: CE 1.4551, read AUC .6319, H .5567.
- Frozen hidden + linear head C=1: train-probe CE .0241 nhưng validation CE
  3.6659; AUC .6371. Có overfit mạnh ở head cụ thể này.
- TF-IDF cùng input: CE 1.3712, AUC .7099. Nội dung vẫn có tín hiệu.
- Thêm epoch cải thiện read ordering, chưa cải thiện rõ CE 5 nhãn.

Chưa thể quy nguyên nhân riêng cho digit head, capacity hoặc số epoch. Plan
này kiểm tra **gói head + regularization + cách khởi động**, rồi đo thêm ích
lợi của LoRA khi dùng cùng head. Kết quả tốt hơn digit arm không tự chứng
minh rằng chỉ riêng việc bỏ digit prior là nguyên nhân.

## 2. Phạm vi dữ liệu và giới hạn suy luận

- Dùng đúng pool 1,494 bài từ fit `outer1_inner0`; giữ membership cũ:
  train 1,196, validation 298, train probes 128. Giữ nguyên publication groups.
- Copy và hash `../qwen_learning_check_v1/outputs/check_v1/membership.json`,
  frozen features, token audits và provenance; verify với input manifest gốc.
  Nếu cache không khớp, tạo extraction mới trong namespace mới.
- Hyperparameter selection chỉ dùng group CV bên trong 1,196 train bài.
  Không đọc validation labels trong fitting/selection hoặc giữa các epochs.
- Validation 298 đã được xem; các dữ liệu này cũng xuất hiện trong nghiên cứu
  trước. Đây vẫn là **exploratory**, không gọi là holdout mới hoặc CV sạch
  khỏi lịch sử lựa chọn phương pháp. Group separation ngăn fit trực tiếp qua
  fold; nó không xóa việc đã xem kết quả trước đây.
- Chưa mở các held-out sets cũ để chọn head. Không train final 2,494, score
  test 596, hoặc thay incumbent. Các metric trong plan chỉ so cùng membership.

## 3. Các arm và đối chứng

| Arm | Backbone và output | Fitting | Vai trò |
|---|---|---|---|
| A: digit-LoRA | Qwen + q/v LoRA, digit projection cố định | 3 epochs như check trước | Reference hiện tại |
| B: frozen-head | Frozen Qwen + affine head 3,584→5 | Regularized CE, chọn penalty bằng train-only group CV | Đo head/regularization không cập nhật encoder |
| C: joint-head | Fresh q/v LoRA + đúng head B | Khởi động từ head B, rồi 3 epochs joint | Đo ích lợi adaptation ngoài frozen-head |
| Prior / TF-IDF | Controls từ check trước | Giữ nguyên artifacts | Mốc no-content và text-only |

Chỉ có **một joint quality fit GPU** cho C. B cần CPU, tận dụng frozen features.
A có thể tái dùng probabilities epoch3 của check trước nếu provenance,
membership, token hashes, base/runtime và training recipe khớp. Ghi rõ reuse;
không gọi là một replicate mới. Nếu cần đổi runtime hoặc recipe để implement
C, re-smoke và chạy lại A theo recipe tương ứng trong namespace mới trước
khi báo so sánh quality. Không âm thầm trộn khác runtime hoặc precision.

## 4. Head và giai đoạn CPU

Feature h là final hidden state ở assistant-start, đúng vị trí và phép cast
đã tạo cache cũ. Không thay prompt/pooling/token caps cùng thí nghiệm này.

```text
x = (float32(h) - train_mean) / train_scale
z = W x + b                 # 5 logits, W shape [5, 3584]
p = softmax(z)              # class order 1, 2, 3, 4, 5
L = mean CE(z, y-1) + lambda/2 * sum(W**2)
```

- StandardScaler fit chỉ trên fitting rows; variance bằng 0 dùng scale 1,
  theo đúng convention của scaler. Scaler là phép biến đổi cố định.
- Head affine có 17,925 parameters; FP32, không hidden layer, activation
  hoặc dropout riêng. Không class weights, oversampling hoặc label smoothing.
- Khởi tạo CPU W=0, b=log(prior add-one) từ labels của fitting rows.
  Sau đó tối ưu objective bằng L-BFGS, tối đa 2,000 iterations. Yêu cầu
  convergence và objective/gradient hữu hạn; numerical failure là technical
  failure, không gán penalty đó quality kém để bỏ qua.
- Khóa grid lambda `[0.001, 0.01, 0.1, 1.0, 10.0]`; **3-fold GroupKFold**,
  memberships lưu trước khi fit. Fail preflight nếu fitting fold thiếu class.
- Mỗi fold fit lại scaler, prior và head chỉ trên fold-train. Chọn lambda
  bằng CE trên các out-of-group predictions nối theo row order; tie trong
  1e-6 chọn lambda lớn hơn. Không dùng H/AUC để chọn penalty.
- Refit B trên toàn bộ 1,196 train với lambda đã chọn. Lưu scaler, W, b,
  fold OOF probabilities và selection trace. Không fit lại trên validation.
- Nếu dùng sklearn C, phải quy đổi chính xác cho objective trên:
  `lambda = 1/(C * n_fit)`; không dùng cùng C qua fold sizes rồi coi lambda
  không đổi. Test objective/gradient tương đương trước khi chọn implementation.

Không gate GPU chỉ vì B không vượt A: frozen features yếu vẫn có thể được
LoRA cải thiện. CPU stage chỉ được dừng do technical failure; quality của B
trên validation chỉ mở cùng kết quả cuối của C.

## 5. Joint training C

- Fresh base `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`; không nạp adapter đã nhìn các
  labels validation hiện tại.
- QLoRA NF4 double quant/BF16, SDPA; q_proj/v_proj, r8, alpha16,
  LoRA dropout .05. Microbatch 1, accumulation 16, clip global norm 1.
- Khởi tạo LoRA như A; head bắt đầu từ B đã fit. Như vậy GPU stage không
  phải học từ head nhiễu và không bị chặn gradient bởi W=0 ban đầu.
- Giữ scaler B cố định. Tính h theo graph có gradient về LoRA; không detach,
  không dùng cached features trong joint forward. Head và loss FP32; verify
  feature/precision parity với cache trước khi cho phép training.
- AdamW: LoRA lr 1e-4, weight decay .01 như A. Head lr 1e-4,
  optimizer weight decay 0; head dùng explicit L2 penalty với **lambda đã
  chọn cho B**, bias không phạt. AdamW decay không được coi là cùng objective
  với penalty CPU. Khóa hai parameter groups, không tune LR sau khi xem val.
- Loss CPU và GPU đều là mean CE + lambda/2 ||W||². Accumulation phải tính
  đúng một penalty mỗi optimizer update; nhóm cuối chia theo số mẫu thực.
- Seed 20261030; shuffle seed+epoch như check trước. Khởi tạo head dùng
  generator riêng hoặc bảo toàn RNG để không vô tình đổi LoRA init/dropout
  RNG. Ghi hashes adapter ban đầu và row order cho paired comparison.
- Đúng 3 epochs, 225 optimizer updates; linear schedule toàn trajectory,
  warmup ceil(5% updates), giống A. Không early stop; epoch3 là endpoint chính.
- Lưu checkpoint/predictions mỗi epoch để vẽ learning curve. Validation labels
  chỉ được join sau khi B và toàn bộ C hoàn tất; không chọn epoch tốt nhất.

B được tối ưu bằng L-BFGS còn C dùng AdamW và thêm adapter parameters. So
sánh C−B đo ích lợi thực dụng của joint stage theo recipe này; không phải
một phép tách nhân quả tuyệt đối riêng từng thành phần optimizer.

## 6. Technical gate và implementation

Tạo package riêng tại `experiments/qwen_head_v1`; không sửa file đã lock của
các runs cũ. CLI dự kiến: prepare → cpu-head → smoke → train → evaluate.
README/config/PROTOCOL và nguồn model wrapper riêng phải có trước execution.

Các checks bắt buộc trước quality run:

1. Membership/group exclusion và input/token hashes đúng. Thay validation
   labels không thay scaler, lambda, initialization hoặc fitted B.
2. CPU/GPU standardized features, logits và probabilities của B khớp trên
   fit-only anchors; max probability error <=1e-4. Nếu khác, sửa source và
   namespace trước khi training, không nới tolerance theo kết quả.
3. Gradient từ head đi tới đúng LoRA modules. Chỉ head/LoRA trainable;
   base weights không gradient và optimizer không chứa base parameters.
4. Smoke 2 updates trên fit-only, đủ năm nhãn: CE/penalty/norm hữu hạn,
   head và LoRA thực sự đổi. Zero-init W riêng chỉ dùng cho CPU initialization.
5. Save/reload **adapter + head + scaler + class order + config** tái tạo
   probabilities với max error <=1e-4. Adapter PEFT riêng không đủ để reload C.
6. Head loss/penalty có gradient đúng; accumulation cả nhóm đủ và nhóm cuối
   khớp objective tham chiếu. Evaluator từ chối thiếu epochs hoặc sai row order.
7. Predictions đủ năm lớp, hữu hạn, chuẩn hóa. Base revision, prompt, code,
   runtime/allocator và mọi artifact source được fingerprint trước fit.

Không cần phát sinh full-vocabulary logits cho C nếu lấy được final hidden
state trực tiếp qua backbone; phải chứng minh feature parity và gradient
đúng qua PEFT wrapper, không bypass adapter vô tình.

Runtime/source đổi → namespace mới và smoke mới. Không overwrite failed
artifacts, sửa fingerprint hoặc resume optimizer giữa chừng. Completed cache
chỉ được reuse sau khi khớp manifest. Training interruption → giữ job failed
và restart fresh C dưới namespace mới, ghi rõ chi phí.

## 7. Evaluation và tiêu chí quyết định

Primary endpoint: epoch3 C so A trên đúng validation 298. B so A và C so B
là các contrasts định trước. Báo Prior/TF-IDF như context cùng split.

- CE 5 nhãn, read AUC/Brier; CE_read và CE_rating_given_read để biết head
  cải thiện phân biệt đọc/không đọc hay thứ tự trong nhóm 2–5.
- `read=1-p1`, `gain=.25*p2+.5*p3+.75*p4+p5`. Native priority bằng
  `1+9*gain` nếu read>.55, còn lại 0. Không đổi cutoff hoặc cân bằng classes.
- H/Q/R; TP/FP/FN/TN; missed4/missed5 riêng; confusion 5×5, precision/recall
  từng nhãn; train probes và output dispersion để phát hiện overfit/collapse.
- Selection-only/ranking-only/matched-budget dùng **A làm anchor cùng split**:
  challenger chọn + A xếp; A chọn + challenger xếp; top read cùng số A chọn
  + A xếp. Gọi đúng tên anchor, không nhầm thành so với incumbent full277.
- Paired publication-group bootstrap 1,000 lần, seed cố định 20261101 cho
  delta CE/AUC/H. Khoảng 2.5–97.5% chỉ mô tả uncertainty exploratory.

Gate thực dụng đăng ký trước, áp riêng B và C so A:

1. CE giảm >=.03, read AUC tăng >=.02.
2. Native H không giảm; missed4 và missed5 đều không tăng, xét riêng.
3. Không technical failure hoặc artifact mismatch.

Để gọi joint adaptation có thêm giá trị so frozen-head: C giảm CE >=.01
so B và AUC không giảm. Đây là ngưỡng practical, không phải significance.

Quyết định cuối:

- C đạt gate so A và có ích thêm so B → đáng confirmation nhiều groups/seeds
  theo protocol riêng; chưa thay incumbent.
- B đạt gate, C không có ích thêm → ưu tiên frozen-head rẻ hơn cho confirmation.
- CE tốt hơn nhưng AUC/H không tốt → cải thiện confidence chưa giải quyết
  discrimination/selection; không gọi đó là quality fix cho cuộc thi.
- C tốt hơn A nhưng không đạt gate → ghi mức cải thiện và uncertainty;
  không tự mở LR/epoch/grid mới trên validation này.
- Cả B/C thất bại → chưa hỗ trợ head recipe này; không kết luận mọi cách dùng
  Qwen hoặc mọi model lớn đều thất bại.

## 8. Ngân sách, artifacts và report

- CPU: 5 penalties × 3 folds = 15 fits, cộng một refit B; ghi thời gian thực.
- GPU: một smoke C, một joint C 225 updates. A rerun chỉ khi điều kiện reuse
  không đạt; không chạy song song mặc định để tránh cạnh tranh VRAM.
- Mốc đo cũ: 225 updates mất 881s training (~14.7 phút), scoring 112s,
  frozen extraction 138s. Đây là reference, không phải ETA bảo đảm cho C.
  Đo lại sau 10 updates; tách load/train/score/reload, peak allocated và NVML.
  Thời gian implement/verification không nằm trong số runtime GPU này.
- Namespace dự kiến `outputs/head_v1_<runtime>`; outputs gitignored.
- Lưu source_manifest, membership, CV memberships/metrics/selection,
  standardized-feature parity, initialization hashes, smoke, training.jsonl,
  head/scaler/adapter từng epoch, probabilities, metrics, transitions,
  confusion, bootstrap, decision.json, COMPLETE/FAILED với exit status.
- Report tiếng Việt trả lời ba câu: head có generalize hơn không; LoRA có
  thêm giá trị trên head đó không; evidence có đủ để đầu tư confirmation không.
  Không so các con số này trực tiếp với public .64349 hoặc pilot mean .64758.

Deliverable của bước hiện tại chỉ là plan này. Implementation và execution
là bước tiếp; khi được giao triển khai, thực hiện theo thứ tự và gate trên.
