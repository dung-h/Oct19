# Kết quả thử nghiệm SPECTER2

SPECTER2 + Ridge là cấu hình được chọn từ development CV, trước khi xem holdout. Trong lần chia này, nó cũng đạt điểm holdout cao nhất trong sáu họ mô hình đã thử.

Lượt [failure analysis tiếp theo](FAILURE_ANALYSIS.md) phát hiện thêm abstract có mã LaTeX, trích quá section và một trường hợp ghép luận án thay cho bài hội nghị. Các kết quả dưới đây là baseline trên đúng snapshot input trong manifest, chưa phải kết quả sau khi sửa các vấn đề này.

| Mô hình | Điểm development CV | Điểm holdout | Relevance Q | Recall R |
|---|---:|---:|---:|---:|
| SPECTER2 + Ridge | 0.61873 | **0.61929** | 0.48477 | 0.85714 |
| SPECTER2 + SVR | 0.61744 | 0.61149 | 0.48135 | 0.83810 |
| TF-IDF + Ridge | 0.61132 | 0.59564 | 0.48020 | 0.78413 |
| TF-IDF + metadata + Ridge | 0.61049 | 0.58626 | 0.48023 | 0.75238 |
| SPECTER2 + metadata + Ridge | 0.59220 | 0.56719 | 0.43117 | 0.82857 |
| Metadata + Ridge | 0.56318 | 0.54894 | 0.37922 | 0.99365 |
| Recommend all, priority bằng nhau | — | 0.52740 | 0.35814 | 1.00000 |

Các họ hồi quy dùng bộ tham số và ngưỡng tốt nhất riêng, được chọn bằng development CV. Metadata gồm venue, year và authors; cả hai baseline TF-IDF và SPECTER2 đều dùng cùng title/abstract sau xử lý. Recommend-all hòa điểm theo ID dạng text đúng quy tắc của đề.

## Cấu hình được chọn

- Frozen `allenai/specter2_base` + `allenai/specter2_regression`; CLS embedding 768 chiều, L2-normalize trước Ridge.
- Ridge `alpha=0.1`, fit trên nhãn 1–5. Transformer không được fine-tune bằng nhãn assignment.
- Chọn bài khi dự đoán rating `p > 2.0`; đặt `priority = 2.5 * (p - 1)`, còn lại 0.
- 1.994 dòng development; 500 dòng holdout; seed outer=42, inner=43; inner CV gồm 4 folds.
- Nhóm duplicate không bị chia qua train/validation: 2.494 record tạo thành 2.479 nhóm. Grouping chưa bảo đảm nhận diện mọi phiên bản gần trùng.
- Hyperparameter và ngưỡng chốt trước khi chấm holdout; không chọn lại theo holdout.

Trên holdout, model được chọn đề xuất 365/500 mục và thu hồi 270/315 mục có nhãn 2–5. MAE rating là 0.97057. SVR có MAE thấp hơn nhưng điểm recommendation thấp hơn, nên việc chọn model dựa trên metric của đề có ý nghĩa.

SPECTER2 + Ridge hơn TF-IDF + Ridge khoảng **0.02365 điểm** trong lần chia này. Thêm metadata làm điểm giảm trong cấu hình đã thử; kết quả này chưa chứng minh metadata luôn vô ích. Chỉ có một outer holdout và một grid nhỏ, chưa có kết luận về ý nghĩa thống kê hay điểm Kaggle.

## Dữ liệu và tài nguyên

Embedding đủ 3.090 record. 80 mục thiếu abstract và 35 mục có ghi chú identity/version review dùng title; 2.975 mục còn lại dùng title + abstract. Có 18 input vượt 512 tokens và bị truncate. Không bỏ các record tuyển tập khỏi tập huấn luyện/test hoặc gán nhãn cho chúng bằng tay.

Inference chạy cục bộ trên RTX 4060 Laptop 8 GB, khoảng 20 giây cho bước encode sau khi load model/tokenize. Thời gian này không bao gồm tải weights, khởi tạo model và validation. Model revisions, checksum input và package versions nằm trong [embedding_manifest.json](outputs/embedding_manifest.json).

File [submission_specter2.csv](outputs/submission_specter2.csv) có đủ 596 test ID theo thứ tự template: 387 mục được đề xuất, 209 mục priority=0. Priority lớn nhất khoảng 7.7601. Model đã refit trên toàn bộ 2.494 nhãn gốc để tạo file này. `submission_best_dev.csv` có cùng dự đoán vì model chung tốt nhất trên development cũng là SPECTER2 + Ridge. Đây là dự đoán cục bộ, chưa được submit lên Kaggle.

## Kiểm tra

Năm kiểm tra cho metric đã qua, bao gồm loại bài, recall denominator, hòa điểm theo ID dạng text và tính nhất quán giữa priority xuất ra với score. Các score holdout còn được tính lại độc lập từ các priority đã lưu. Kiểm tra nhóm không giao nhau qua các split và cấu trúc 596 ID trong submission cũng đã qua. Pipeline đã lưu được load lại và tái tạo priority trong sai số số học float32.

Xem [README.md](README.md) để chạy lại và hiểu các giả định, đặc biệt sự khác nhau giữa mô tả metric ở phần overview ngắn và phần Evaluation chi tiết.
