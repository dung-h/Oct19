# Kiểm tra bổ sung: phối hợp neural với incumbent

Ghi kế hoạch trước khi tính bất kỳ blend score nào. Đã xem kết quả ba outer folds
của thí nghiệm chính; đây là nghiên cứu bổ sung trên cùng dataset, không phải một
thí nghiệm preregistered từ đầu hoặc đánh giá độc lập. Kết quả primary không đổi.

- Giữ nguyên mọi neural fit, epoch/arm do primary inner selection chọn. Không
  thay LR, checkpoint, seed hay khai thác outer labels để đổi model.
- Với mỗi outer, lấy `neural_selection` đã chọn từ inner (chưa dùng outer labels),
  lấy probability của neural arm/epoch đó trên từng inner holdout và outer.
- Blend distribution: `(1-w)*incumbent + w*neural`, w trong[0,.25,.5,.75,1].
  Ngưỡng đọc trong[0..0.90] bước0.05; priority10*expected_gain. Chọn w và cutoff
  bằng mean4 inner H, ties ưu tiên w nhỏ rồi cutoff nhỏ.
- Đánh giá outer bằng weight/cutoff đã chọn trong inner. Báo riêng H/Q/R/AUC,
  FP/FN/high_FN, và delta từng fold so với incumbent và primary policy. Không
  chọn weight bằng điểm outer, không chỉ báo weight thắng outer.
- Final chọn lại bằng4 inner toàn train đã có; dùng neural challenger đã refit
  và incumbent classifier cũ để tạo `submission_blend_candidate.csv`. Không
  thay CSV primary hoặc upload Kaggle. Lưu công thức, nguồn model và kiểm tra
  lại raw predictions/model reload, metric, mọi inner choices và596 ID.
- Không hứa H>0.7. Nếu blend không cải thiện thì báo thất bại, không mở rộng
  weight grid hay thêm phép kết hợp sau khi xem score.
