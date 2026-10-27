# Kết quả kiểm tra cách học Qwen — 2026-10-27

Đã chạy đủ đối chứng và ba epoch theo protocol khóa trước. Pipeline tmux
`qwencheck` kết thúc exit0; smoke và adapter epoch3 đều đạt save/reload
parity0.0. Kết quả nghiêng về hạn chế của cách học/generalization hiện tại,
chưa có bằng chứng cần chuyển sang LLM lớn hơn. Train thêm từ epoch1 đến
epoch3 cải thiện read ranking, nhưng gần như không cải thiện 5-class logloss.

## Phạm vi và tính hợp lệ

- Chỉ dùng 1,494 fit documents của outer1_inner0: inner-train1,196 và
  validation298, publication groups không giao nhau. Old held-out502 bị loại.
- LoRA khởi tạo mới từ đúng Qwen2.5-7B-Instruct revision
  a09a35458c702b33eeacc393d103063234e8bc28. Không dùng adapter pilot đã nhìn
  nhãn của validation nội bộ này.
- Chỉ inner-train được dùng fit scaler, vocabulary, classifier, output bias.
  Nhãn validation chỉ được join sau khi mọi arm và ba epoch hoàn tất.
- Mỗi arm được đo thêm trên cùng128 train probes chọn trước, không dùng nhãn.
- Cutoff.55, input title/abstract có cùng token caps, giữ đủ nhãn1–5.
- Đây là một fit-internal split exploratory. Không so metric trực tiếp với
  old pilot folds, incumbent cache hoặc public score. Single three-epoch LR
  trajectory cũng không phải exact replay của schedule1epoch cũ.

## Kết quả validation298

CE/Brier thấp hơn tốt hơn; AUC/H cao hơn tốt hơn.

| Arm | 5-class CE | Read AUC | Read Brier | H |
|---|---:|---:|---:|---:|
| Inner-train class prior | 1.4872 | .5000 | .2402 | .4906 |
| Frozen digit logits | 13.2013 | .4275 | .3959 | .4832 |
| Frozen digit logits +5 biases | 2.0818 | .4293 | .3061 | .4162 |
| Frozen hidden state +linear head | 3.6659 | .6371 | .3177 | .5259 |
| Same head, shuffled train labels | 4.0548 | .5566 | .3488 | .4541 |
| TF-IDF +linear classifier | 1.3712 | .7099 | .2115 | .5839 |
| LoRA epoch1 | 1.4594 | .5865 | .2342 | .5309 |
| LoRA epoch2 | 1.4489 | .6265 | .2337 | .5487 |
| LoRA epoch3 | 1.4551 | .6319 | .2361 | .5567 |

Các head CPU đều dùng hyperparameter cố định, không grid search. Linear head
Qwen có3,584 features nhưng chỉ1,196 train examples và C=1; đây là một control
cụ thể, không đại diện mọi cách dùng Qwen hidden states.

## Có nên train thêm?

| Epoch | Train-probe CE | Train-probe read AUC | Validation CE | Validation read AUC | Validation H |
|---|---:|---:|---:|---:|---:|
| 1 | 1.5105 | .7321 | 1.4594 | .5865 | .5309 |
| 2 | 1.4539 | .7952 | 1.4489 | .6265 | .5487 |
| 3 | 1.4110 | .8091 | 1.4551 | .6319 | .5567 |

Epoch3−epoch1:

- Read AUC +.04536; paired publication-group bootstrap2.5–97.5% interval
  [+ .02053, + .07175]. Có bằng chứng descriptive về read ranking tốt hơn.
- CE −.00433; interval[−.03559,+.02467], gần như flat. Không đạt tiêu chí đã
  khóa CE giảm>=.03 và AUC không giảm: extra_epochs_promising=false.
- Read Brier.2342→.2361 cũng không cải thiện trong comparison này.
- Native H tăng.02578 tại cutoff cố định; TP139→147, FP76→73, FN41→33,
  TN42→45; missed4/5 từ5/4 xuống3/2. Đây là point estimate trên một split,
  chưa phải bằng chứng promotion.
- Train-probe CE giảm.09953 nhưng validation CE gần như không giảm. Tăng
  exposure giúp fit/read separation, còn generalization của phân phối5class
  vẫn yếu. Probe và validation khác class proportions, nên không suy diễn
  mức overfit chính xác từ raw train/val CE gap.

Đã có lợi ích khi đi từ1 lên2–3epoch trong trajectory này. Chưa có bằng chứng
để tiếp tục mở5–10epoch hoặc coi thiếu epoch là nguyên nhân chính. Giữ epoch3
như một kết quả diagnostic; không chọn checkpoint mới để thay incumbent.

## Có phải Qwen yếu hoặc output head chưa hiệu quả?

Bias-only không giải quyết được: validation CE2.0818, AUC.4293. Frozen digit
logits không chỉ lệch một intercept toàn cục; độ biến thiên theo document
cũng không phù hợp rating. LoRA đã học vượt control này.

Đổi sang frozen hidden state +linear head hiện tại cũng không chữa được:
train-probe CE.0241/AUC1.0 nhưng validation CE3.6659/AUC.6371. Head này
overfit và dự đoán quá tự tin. So epoch3, AUC chỉ hơn.00530, interval
[−.06555,+.07581], còn CE tệ hơn2.21081. hidden_head_advantage=false.
Điều đó bác bỏ việc coi head tuyến tính mặc định là sửa lỗi đã chứng minh;
không chứng minh toàn bộ representation/capacity Qwen không đủ.

Hidden-head so shuffled-label head có AUC point delta+.08051 nhưng interval
[−.00918,+.17158] đi qua0. Không gọi đây là bằng chứng chắc chắn về khả năng
generalize của frozen representation trên một split nhỏ.

TF-IDF dùng đúng title/abstract capped đạt AUC.7099, CE1.3712, H.5839:
point estimates tốt hơn LoRA3 (AUC+.07808, CE−.08384) và hidden head.
TF-IDF−hidden-head AUC interval[+.00802,+.13312]. Cùng đầu vào vẫn có tín
hiệu preference dùng được, nên thất bại không thể quy hoàn toàn cho input
không có thông tin. Cách representation/readout/regularization hiện tại khai
thác tín hiệu kém hơn control text rẻ này; model size lớn tự nó chưa giải quyết.

## Kết luận thực dụng

1. Model có học thật: read ordering và selection cải thiện theo epoch;
   training kỹ thuật hoạt động và adapter thay đổi.
2. Đơn thuần train lâu hơn chưa sửa được generalization5class. Lợi ích
   validation CE rất nhỏ dù fit CE tiếp tục giảm. Không mở thêm epoch grid.
3. Chưa nên mua chi phí một LLM lớn hơn chỉ từ các kết quả này. Control
   TF-IDF mạnh hơn cho thấy phương pháp học/sample efficiency đáng ưu tiên.
4. Nếu tiếp tục Qwen, bước hợp lý là một protocol riêng về regularization
   và readout/ordinal parameterization, chọn trong inner-train group CV sạch.
   Head frozen C=1 đã overfit; bias-only cũng thất bại, nên không gọi hai
   thay đổi đó là quality fix. Giữ TF-IDF làm control cùng input.
5. Representation và thiếu metadata ngoài nội dung vẫn chưa tách hết. Chỉ
   đổi model hoặc thêm author cần đối chứng và lineage riêng; chưa có
   corrective run được chứng minh hơn incumbent.

## Runtime, timings và artifact verification

RTX5070Ti, driver595.91.07, Python3.11.16, torch2.12.0+cu130,
bnb0.48.0, BF16/NF4; allocator expandable_segments:True. Các phiên bản còn
lại, code/config/source hashes nằm trong runtime_lock.json.

- Smoke2updates: TECHNICAL_PASS, parity0.0.
- Frozen extraction:1,494×3,584 hidden features; loading10.32s,
  forward/extraction138.29s. CPU controls3.69s.
- LoRA:225updates/3epochs, mọi loss/norm hữu hạn; train881.24s,
  scoring111.81s, loading9.71s, reload8.29s. Khoảng3.9s/update;
  peak CUDA allocation8.21GiB. Tổng VRAM steady-state quan sát trong các lần
  monitor training khoảng6.6–7.1GB; peak allocation có cả transient loading.
- Adapter epoch1/2/3 hashes đều khớp; final parity0.0. Frozen base không
  có gradient, adapter thay đổi. Probability sums sai số tối đa1.20e-7.
- Membership/group exclusion, feature order/shape/hash, capped TF-IDF text,
  scaler train-only và class order1–5 đã xác minh. Original pilot source và
  cả hai adapter cũ giữ nguyên hashes.
- Hai CPU integration checks đã pass trước run. Không lặp tests khi không
  có source change/failure. Không có technical failure trong check_v1.

Artifacts ở outputs/check_v1, bị gitignore:

- [metrics.csv](outputs/check_v1/metrics.csv), [decision.json](outputs/check_v1/decision.json).
- [postrun_verification.json](outputs/check_v1/postrun_verification.json).
- [membership.json](outputs/check_v1/membership.json), [runtime_lock.json](outputs/check_v1/runtime_lock.json).
- [training/COMPLETE.json](outputs/check_v1/training/COMPLETE.json), adapters,
  per-epoch predictions và training.jsonl.
- features.npz, feature_texts.jsonl và các CPU model checkpoints.
- Console log: outputs/check_v1.log; tmux qwencheck pane dead, exit0.

Không train final2494, predict test, upload Kaggle hoặc thay incumbent/submission.
