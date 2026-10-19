# Kiểm tra lượt V2 ngày 2026-10-18

Run chưa hoàn tất: outer0/1/2 đã xong; outer3 đã lưu đủ inner0/1/2 cho hai arm,
nhưng dừng tại outer3/inner3/query_only. Outer4 chưa chạy. Có 36/50 fit caches
thành công. Không có process nested CV đang chạy tại thời điểm kiểm tra.

Source evaluate.py vẫn khớp SHA256 trong evaluation_manifest:
`0c34b764763bfd62d9ea4081da06f86e0447e82e0b8ad58dedbfc8b26db027ae`.

## So sánh trên cùng ba outer folds hoàn tất

Tính lại từ NPZ và choices từng fold, xác nhận fit/pred đúng split; incumbent dùng
probability cache của cùng fold, cutoff .55. Không chọn model/ngưỡng từ bảng này.

| Arm | Mean H | Mean Q | Mean R | Mean read AUC | TP | FP | FN | Missed L4 | Missed L5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Incumbent | .634003 | .525241 | .800224 | .793718 | 752 | 208 | 188 | 21 | 5 |
| Query-only | .615607 | .478711 | .866522 | .778847 | 814 | 306 | 126 | 15 | 7 |
| Contextual | .625592 | .486802 | .877305 | .789661 | 825 | 298 | 115 | 12 | 6 |

Contextual hơn query-only ở 3/3 folds nhưng thấp hơn incumbent ở 3/3 folds.
Delta mean H contextual - incumbent = -.008411. Đây là báo cáo ba folds, chưa
phải kết quả năm folds, không thay incumbent.

Chuyển quyết định contextual so với incumbent:

| Fold | FP loại đúng | TP làm mất | FN cứu được | FP mới |
|---|---:|---:|---:|---:|
| 0 | 10 | 17 | 22 | 18 |
| 1 | 0 | 6 | 42 | 54 |
| 2 | 12 | 6 | 38 | 40 |
| Tổng | 22 | 29 | 102 | 112 |

Model cứu được 102 FN nhưng làm mất 29 TP và thêm 112 FP; FP giảm đúng 22.
Ròng TP +73, FP +90. Q giảm trong cả ba folds; tăng recall chưa bù được Q.
Counts không thay thế H, và không chứng minh nguyên nhân sâu là tokenizer hay
representation.

## Tái hiện partition bị dừng

Chạy lại nguyên `fit_predict` với cùng source, seed, nhóm, 2 epochs và optimizer;
đọc `info` tại exception để không nới guard hay thay code đã hash. Loss hai epoch
tái hiện lần lượt 1.8089477431 và 1.6488398472.

- Adapter có gradient 750/750 batch, adapter L2 drift 2.332715.
- Active ReLU-row fraction train=.965333, prediction=.974, thấp hơn guard .99.
- 498 held-out query tạo 485 vector probability khác nhau; nhóm trùng lớn nhất 13.
- Read std=.257898, predictions hữu hạn.

Đây là bất hoạt toàn bộ hidden ReLU trên một phần mẫu (13/498 held-out), không
phải collapse toàn bộ như V1. Guard đã phát hiện bất hoạt cục bộ. Chưa xác minh
các vector trùng còn lại có do input giống nhau hay không. Không tự nới guard để
hoàn tất CV. Failure audit có trong `outputs/outer3_inner3_query_only_failure_audit.json`.
Count 498 đã được xác nhận bằng membership trong lượt audit chi tiết; con số 500
trước đó suy ra sai từ tỷ lệ mean theo batch. Xem `HEAD_DIAGNOSIS.md` cho cơ chế,
audit từng mẫu, kiểm tra FP32 và checkpoint reload.

36 cached fits thành công đều pass các guard được lưu; min prediction active
fraction=1.0, largest duplicate group tối đa 6. Đây là audit artifact và replay
partition lỗi, không phải verifier độc lập đầy đủ: chưa replay checkpoint hay
kiểm tra sensitivity với việc hoán đổi reference.

## Bước cần làm trước khi tiếp tục

Phân tích 13 mẫu hidden bất hoạt, kiểm tra model sensitivity/input và head bias.
Chốt phép sửa head trong một protocol mới nếu cần; không thay guard sau khi xem
outer scores để biến run lỗi thành pass. Giữ kết quả ba folds ở trạng thái tạm.
