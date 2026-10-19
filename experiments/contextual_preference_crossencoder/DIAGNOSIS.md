# Chẩn đoán sau khi kiểm tra probability và probe trên train

## Kết luận

Lượt CV gốc đã được dừng vì control `query_only` collapse. Không dùng chênh lệch
H giữa contextual và control để kết luận chất lượng semantic của phương pháp.
Giữ nguyên source/protocol và prediction cũ để audit; bản sửa cần thư mục mới.

## Bằng chứng từ artifact

Outer fold 0 có 496 mẫu:

| Chỉ số | Query-only | Contextual | Incumbent cùng fold |
|---|---:|---:|---:|
| H | .534445 | .596055 | .643408 |
| AUC read | .500000 | .748238 | .801395 |
| FP | 188 | 127 | 79 |
| FN | 0 | 13 | 50 |
| Vector probability khác nhau | 1 | 257 | chưa audit |
| Số mẫu có chung một vector lớn nhất | 496 | 239 | chưa audit |

Query-only có cùng probability `[.310481, .200723, .188006, .163652, .137138]`
cho toàn bộ 496 paper. Ba trong bốn inner fits của fold 0 cũng chỉ có một vector
prediction. Vì vậy control lỗi từ bên trong selection, không phải chỉ outer bị lệch.
Contextual cũng có 239/496 prediction trùng hoàn toàn; cần audit head của nó.

So sánh trước đó với H=.640276 dùng mean incumbent năm folds; so sánh đúng fold 0
là .643408. Các score trên không chứng minh contextual tốt hơn một control khỏe mạnh.

## Nguyên nhân đã tái hiện

Trong implementation cũ `specter2_preference/modeling.py`, `embed()` trả CLS đã
L2-normalize trước khi head nhân 10. Trong implementation mới `_forward()` trả
CLS thô, nhưng head vẫn nhân 10. Đây là thay đổi scale khoảng 25 lần.

Probe dùng outer0/inner0 training partition, seed và optimizer hiện tại; 128 bước
batch4 (512 training examples), không dùng nhãn held-out để lựa chọn cấu hình.
Hai lượt khởi tạo lại encoder/head và chỉ thay normalization trước head:

| Probe sau 128 bước | Raw CLS | L2-normalized CLS |
|---|---:|---:|
| Norm CLS thô | ~24.9 | ~24.9 |
| Mẫu probe | 32 | 32 |
| Mẫu có toàn bộ hidden ReLU bằng 0 | 32 | 0 |
| Vector probability khác nhau | 1 | 32 |
| Std probability read | 0 | .129333 |
| Training steps không có gradient adapter khác 0 | 35/128 | 0/128 |

Chuỗi cơ chế được tái hiện: scale đầu vào head quá lớn trong cấu hình hiện tại →
hidden ReLU chết → output chỉ còn bias → paper nào cũng được probability giống
nhau → gradient không truyền trở lại adapter. Có adapter drift >0 trên toàn fit
không loại trừ collapse xảy ra ở các bước sau.

Khẳng định trước đó rằng `encoder.train()` làm adapter mất gradient chưa được
chứng minh; chuyển sang eval vẫn tái hiện collapse. Preflight một bước và loss
giảm đều không đủ để xác nhận model học cách phân biệt paper.

## Hạn chế input đã xác nhận, tác động chưa được đo

Input audit: 2455/3090 paper (~79.4%) bị cắt ở 126 tokens mỗi phía. Đây là mất
nội dung rõ ràng, nhưng chưa có ablation chứng minh nó giải thích bao nhiêu điểm H.
Pair construction bỏ `token_type_ids`, không thể hiện segment B bằng embedding
segment riêng; SPECTER2 gốc dùng paper title/abstract cho embedding, không phải
checkpoint được huấn luyện sẵn để chấm preference của cặp paper.

Tham khảo: https://huggingface.co/allenai/specter2

## Kế hoạch sửa có thứ tự

1. **Sửa scale và kiểm chứng train trong thí nghiệm mới.** Giữ các điều kiện khác
   để cô lập tác động normalization; thêm L2 normalize trước cả hai head. Log
   activation, số batch mất gradient, độ phân tán probability và lưu checkpoint.
   Dùng AMP với GradScaler/unscale trước clipping, đối chiếu một batch FP32 để
   phát hiện lỗi số học. Không tự coi việc loss giảm là điều kiện pass.
2. **Smoke test nhiều bước trên train trước CV.** Kiểm tra raw/normalized matched
   probe, frozen encoder có đổi output theo nội dung, model học được tập train
   nhỏ, prediction phụ thuộc nội dung query, save/reload tái tạo prediction.
   Nếu control còn collapse thì dừng, chưa chạy nested-CV.
3. **Sửa input bằng ablation riêng.** Sau khi model khỏe, đo token coverage và
   xem phần bị cắt của FP/FN; so sánh ngân sách dài hơn hoặc phần đầu+cuối abstract
   trên inner folds. Kiểm tra format title/abstract và segment của pair, đồng thời
   ghi rõ SPECTER2 đang được chuyển sang nhiệm vụ mới. Chọn cấu hình bằng inner.
4. **Kiểm tra model có dùng semantic của reference.** So sánh contextual thật với
   reference chỉ có label/không text và reference text bị hoán đổi trong pool fit,
   giữ query và label-reference cố định. Tách tác động pairwise loss khỏi việc
   thêm context; không gán toàn bộ chênh lệch cho semantic cross-attention.
5. **Đánh giá lại và tích hợp nếu có tín hiệu.** Chạy group-safe nested CV cho bản
   khỏe, so với incumbent đúng từng fold. Báo TP/FP/FN theo label, AUC, Q/R/H và
   chuyển dịch lỗi. Chỉ thử bổ sung semantic vào metadata/incumbent bằng inner-OOF
   nếu nó phân biệt paper tốt và sửa được lỗi bổ sung; giữ ngưỡng promotion cũ.

Outer labels đã được xem nhiều lần: mọi kết luận tiếp theo đều là nghiên cứu
exploratory, không phải ước lượng từ một holdout hoàn toàn chưa từng được xem.
