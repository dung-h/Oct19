# SPECTER2 học trực tiếp nhãn sở thích

Kế hoạch cố định trước khi xem điểm outer của thí nghiệm này. Mục tiêu là kiểm tra
việc cập nhật biểu diễn theo nhãn, không hứa đạt H > 0.7.

- Dữ liệu/nhãn gốc và grouping đã dùng trước; 2494 train, 596 test. Chỉ title
  sạch + SEP, CLS L2, tối đa 128 tokens. Không đưa nhãn, ID, venue vào token.
- Hai nhánh matched: `frozen` và `adapted`. Cùng SPECTER2 base revision
  `3447645e1def9117997203454fa4495937bfbd83`, pretrained regression adapter
  revision `a9d105def7a081c8f7c5a88ab4145646901c6003`. Adapted chỉ cập nhật
  adapter pretrained; base transformer luôn đóng băng. Đây không phải full fine-tune.
- Encoder eval mode ở cả hai nhánh để giữ dropout cố định; autograd vẫn bật cho
  adapter. Head giống nhau: L2 CLS * 10 -> Linear768/64 -> ReLU -> Dropout0.1 ->
  Linear64/5. Khởi tạo head và thứ tự minibatch giống nhau, seed20261021.
- Loss: CE 5 lớp + 0.5 BCE đọc/không đọc, với read logit =
  logsumexp(logits2..5)-logit1. Không cân lại lớp/resample, không hard pairs,
  không sinh nhãn. P(read)=1-P(label1); priority=10*E[(label-1)/4].
- AdamW head lr0.001, adapter lr0.00005, weight_decay0.01, batch32, clip1,
  fp16 AMP + GradScaler. Mỗi hàng training đúng một lần mỗi epoch.
- Sửa lỗi triển khai trước bất kỳ outer score nào: GradScaler bắt đầu scale1024,
  tự bỏ qua update tràn số và giảm scale như cơ chế AMP chuẩn. Log số update
  hữu hạn và số overflow. Lượt dừng đầu tiên được lưu ở `outputs/aborted_amp_preflight`;
  chạy formal lại cả hai nhánh. Không đổi LR/loss/checkpoints theo kết quả.
- Checkpoints1,2,4,8 epochs; ngưỡng P(read)>0..0.90 bước0.05. Cùng ngân sách
  chọn checkpoint/ngưỡng cho hai nhánh. Không thêm LR/epoch/seed sau khi xem outer.
- Đúng 5 outer / 4 inner groups đã lưu. Mỗi inner fit 8 epochs và lưu xác suất
  ở bốn checkpoints; chọn epoch/ngưỡng bằng mean inner H. Outer refit đúng số
  epoch đã chọn, không early-stop theo outer. Không gọi outer là dữ liệu chưa từng xem.
- Đối chứng incumbent là CatBoost `both_d4` đã lưu (cùng membership), chọn lại
  ngưỡng trên inner. Báo riêng mỗi family, policy chọn trong hai neural arms,
  và policy chọn cả incumbent. Không thay điểm policy bằng arm thắng outer.
- Chẩn đoán phụ: tại epoch đã chọn, chọn ngưỡng có inner recall gần0.80 nhất;
  báo FP/FN/H/recall outer để xem có giảm FP tại recall tương đương hay không.
- Báo H/Q/R, AUC, multiclass logloss, FP/FN, missed labels4/5, train-vs-val AUC;
  paired differences mỗi fold. Mục tiêu0.7 được báo như kết quả, không là lý do
  đổi validation/nhãn hay chọn seed.
- Final: cùng4 grouped folds toàn train seed20261118 như incumbent, chọn lại
  epoch/ngưỡng/arm trong inner. Fit neural challenger tốt nhất trên toàn train,
  lưu adapter/head và reload để xác minh dự đoán. Nếu global selection chọn
  incumbent, recommended CSV giữ ứng viên incumbent; challenger lưu riêng.
- Kiểm tra group separation, hash dữ liệu/splits/protocol, adapter-only gradients
  và updates, loss hữu hạn, probability simplex, metric và mọi lựa chọn inner,
  596 ID duy nhất, reload cùng thứ tự. Không gửi Kaggle trong thí nghiệm này.

Tài liệu: [SPECTER2](https://huggingface.co/allenai/specter2),
[AdapterHub training](https://docs.adapterhub.ml/training.html).
Giới hạn: inner/outer cùng dataset đã được nghiên cứu nhiều lần; một kết quả tốt
vẫn cần xác nhận trên Kaggle. Hai nhánh chỉ kiểm tra cấu hình adapter cụ thể này.
