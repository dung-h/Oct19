# Chẩn đoán 13 mẫu hidden bất hoạt

Đã replay partition `outer3_inner3_query_only` bằng source đã hash, seed,
optimizer, token input và 2 epochs không đổi. Không đổi guard và không tiếp tục
nested CV. Script: `diagnose_head.py`; artifact: `outputs/head_diagnostic/`.

## Nguyên nhân trực tiếp

Ở checkpoint cuối, phép linear đầu của head cho 96 giá trị âm trên từng mẫu
trong nhóm 13. ReLU đưa toàn bộ hidden về 0. Logits chỉ còn bias của output layer,
nên các văn bản khác nhau nhận cùng probability. Gradient loss về embedding của
những mẫu đó bằng 0 ở trạng thái này. Gradient adapter của cả minibatch vẫn có
thể khác 0 nhờ các mẫu khác; kiểm tra theo batch không phát hiện được mọi mẫu chết.

Điều này không chứng minh trạng thái vĩnh viễn không thể phục hồi, hoặc mọi
ReLU unit của cả model đều chết.

## Đã loại trừ và khoanh vùng

- 13 token inputs khác nhau; không input nào trùng với input held-out khác.
- Cả 13 có abstract. Bốn paper không bị cắt token; chín paper bị cắt. Vì vậy
  truncation không phải điều kiện cần để failure xảy ra.
- FP32 cũng cho 13/13 hidden bất hoạt. Maximum preactivation từng mẫu nằm trong
  khoảng -.166 đến -.00637: không phải riêng sai số làm tròn FP16 sát 0.
- CLS đã normalized; failure vẫn xuất hiện sau nhiều bước. Mẫu train đầu tiên
  có hidden bất hoạt được ghi nhận ở optimizer step 360/750.
- Save/reload adapter và head tái tạo probability trên toàn held-out với max
  absolute difference bằng 0.

Các phép hoán đổi sau train chỉ dùng để chẩn đoán activation, không làm ứng viên:

| Embedding | Trọng số/bias head đầu | Mẫu hidden bằng 0 / 498 |
|---|---|---:|
| Trước train | Trước train | 0 |
| Sau train | Sau train | 13 |
| Sau train | Trước train | 0 |
| Trước train | Sau train | 0 |
| Sau train | Trọng số sau train, bias trước train | 14 |
| Sau train | Trọng số trước train, bias sau train | 0 |

Cơ chế liên quan tới tương tác giữa embedding sau adapter tuning và trọng số
head đã học; không có bằng chứng quy nguyên nhân chỉ cho bias hay chỉ cho
encoder. Đổi bias về khởi tạo không giải quyết. Learning rate, batch size và
loss có thể ảnh hưởng optimization, nhưng chưa có ablation xác định yếu tố nào.

## 13 mẫu và tác động lên đầu ra

Membership gốc xác nhận **498** held-out rows, không phải 500 như báo trước.
Giá trị .974 được runner tính bằng mean batch active fractions, không phải
active rows / total rows; count đúng là 13/498. Guard vẫn fail ở cả hai cách tính.

| Label | Số mẫu |
|---|---:|
| 1 | 1 |
| 2 | 1 |
| 3 | 1 |
| 4 | 4 |
| 5 | 6 |

Venue: KR=5, ICLP=4, LPNMR=4. Các title chủ yếu liên quan ASP/logic programming,
nhưng mẫu chọn từ failure nên không suy rộng thành lỗi ở toàn bộ chủ đề này.

Probability chung xấp xỉ `[.192734, .201478, .206293, .194932, .204563]`.
P(read)=.807265, expected gain=.504278. True mean gain nhóm này=.75; nhóm có
10 paper Label4/5, nhưng model không phân biệt được thứ tự giữa các label.
Tại cutoff .45 hoặc .50 cả 13 vẫn được nhận, bao gồm một FP. Vì vậy không được
nói rằng 12 paper đáng đọc này bị loại bởi gate: failure quan sát ở đây là
mất phân biệt priority và gradient trực tiếp, không phải trực tiếp rớt selection.

Ví dụ Label5: `Formal Verification of Answer Set Programs Containing Advanced
Language Constructs`, `Reasoning About Actual Causality in Answer Set Programming`,
`Neural-Probabilistic Answer Set Programming`. Chúng có cùng prediction với bài
Label1 `Enhancing DLV for Large-Scale Reasoning`.

Chi tiết từng ID/title/input length/activation/probability:
`outputs/head_diagnostic/sample_audit.csv`.

## Hướng sửa đề xuất, chưa triển khai

Tạo pilot riêng dùng LeakyReLU với slope cố định thay cho ReLU trong hai head,
giữ input/reference/loss/split không đổi. LeakyReLU giữ đạo hàm trên nhánh âm;
đây là can thiệp trực tiếp vào cơ chế đang thấy. Kiểm tra sensitivity/gradient
theo mẫu, output dispersion và checkpoint reload, rồi mới xét CV.

Không dùng labels của 13 held-out samples để chọn slope, learning rate hoặc
rule riêng cho ASP. Không hứa sửa activation sẽ làm H tăng; mean H thấp hơn
incumbent trên ba outer folds còn có selection/ranking tradeoff riêng.

Nguồn định nghĩa activation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
