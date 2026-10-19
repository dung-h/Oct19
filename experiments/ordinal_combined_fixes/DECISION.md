# Kết luận kiểm tra

Giữ multiclass hiện tại. Không có cấu hình mới nào vượt H=0.640276 trên trung bình năm outer folds. Đây là so sánh thăm dò trên dữ liệu đã xem nhiều lần, không phải điểm Kaggle hay bằng chứng khái quát hóa độc lập.

| Cấu hình | H |
|---|---:|
| Multiclass hiện tại | 0.640276 |
| Ordinal gốc, thí nghiệm trước | 0.635347 |
| Ordinal khóa mặc định | 0.638415 |
| Ordinal chỉ bỏ láng giềng | 0.639459 |
| Ordinal kết hợp khóa mặc định + bỏ láng giềng | 0.637296 |
| Kết hợp + regularization mạnh | 0.637555 |
| Ordinal chuẩn hóa cột đếm | 0.638909 |
| Multiclass chuẩn hóa cột đếm | 0.636930 |
| Quy trình tự chọn trong tám cấu hình bằng inner CV | 0.636459 |

Các sửa đổi riêng lẻ không cộng dồn: kết hợp kém hơn khóa mặc định riêng 0.001119 H và kém hơn bỏ láng giềng riêng 0.002163 H. Regularization thêm chỉ tăng 0.000259 so với cấu hình kết hợp.

## Những gì đã xác định được

1. **Giảm độ khớp train chưa đủ để tăng H.** Cấu hình kết hợp + regularization có AUC train=0.836590 và validation=0.800731, gap=0.035859; ordinal gốc có gap=0.165646. H vẫn thấp hơn multiclass. Vì vậy không thể xem việc giảm gap là tiêu chí thành công cuối cùng.
2. **Lệch đơn vị đếm có thật, nhưng phép sửa này không tạo cải thiện lớn.** Trung bình năm fold, 96.07% giá trị abstract/all support và 95.35% retrieval/venue support ngoài khoảng train; sau chuẩn hóa về tỷ lệ, các con số này bằng 0%. Ordinal chỉ tăng 0.000495 H so với ordinal đã khóa mặc định, còn multiclass giảm 0.003346 H. Điều này chưa loại trừ tác động của quy mô tập tham chiếu lên các thống kê láng giềng khác.
3. **Điểm mất chủ yếu nằm ở quyết định giữ/bỏ bài.** Với cấu hình kết hợp, hoán đổi gate/ranker và chia phần tương tác cho thấy delta H=-0.002980 gồm gate=-0.002791 và rank=-0.000189. Đây là phân rã dự đoán, không phải chứng minh nguyên nhân sâu bên trong embedding.
4. **Lỗi cụ thể của cấu hình kết hợp:** cứu được 53 bài nhãn >1 mà multiclass bỏ, nhưng lại bỏ thêm 83 bài multiclass giữ; với nhãn 4/5 là cứu 6, bỏ thêm 19. Tổng FN tăng 313→343, số bài nhãn 4/5 bị bỏ tăng 38→51. AUC tăng nhẹ cũng không bù được tổn thất theo H.

## Quyết định thực tế

Không tiếp tục mặc định rằng bỏ thêm feature hoặc tăng regularization sẽ cải thiện. Các phép thử đã giảm gap và sửa thang đo nhưng chưa tạo ưu thế theo H. Chưa có đủ bằng chứng quy lỗi cho embedding hoặc kết luận toàn bộ preprocessing kém.

Nếu tiếp tục nghiên cứu, tập trung vào độ ổn định của quyết định giữ/bỏ bài: dùng các dự đoán ngoài fold và các ca đổi quyết định đã lưu để xác định nhóm bài bị mất, sau đó đặt một can thiệp cụ thể có đối chứng. Không coi đổi ngưỡng theo nhãn outer là giải pháp triển khai, không hứa hẹn vượt 0.7 từ các sửa đổi này.

Inner CV cuối chọn cấu hình kết hợp + regularization vì inner H=0.639034, chỉ hơn multiclass 0.638687 khoảng 0.000347. Candidate đó được lưu riêng trong thư mục thí nghiệm để tái lập đúng quy trình; điểm outer của toàn bộ quy trình chọn là 0.636459. Không thay submission chính và không upload Kaggle.

Chi tiết: [RESULTS.md](RESULTS.md), [đối chứng](outputs/contrasts.csv), [lỗi thay đổi](outputs/gate_changes.csv), [kiểm tra độc lập](outputs/verification.json). Incumbent giữ ở `../preference_augmentation/outputs/submission_candidate.csv`.
