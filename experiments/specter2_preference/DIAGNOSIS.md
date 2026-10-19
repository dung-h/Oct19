# Chẩn đoán sau thí nghiệm

Adapted so với incumbent cố định: delta H **-0.00709**. Toàn policy chọn trong
inner có H **0.63277**; không coi family thắng
outer là điểm của policy.

## Lỗi đọc/không đọc

| arm                 | reference   |   fixed |   introduced |   net_errors_reduced |   errors_before |   errors_after |
|:--------------------|:------------|--------:|-------------:|---------------------:|----------------:|---------------:|
| adapted             | frozen      |     242 |          169 |                   73 |             758 |            685 |
| adapted             | incumbent   |     215 |          249 |                  -34 |             651 |            685 |
| selection_procedure | incumbent   |     176 |          213 |                  -37 |             651 |            688 |

Đây là thay đổi quyết định nhị phân, không thay thế H (còn phụ thuộc nhãn2..5 và
thứ tự). Chi tiết từng ID ở `failure_transitions.csv`, theo venue/label ở
`failure_breakdown.csv`; các trường hợp đã bàn trước ở `known_cases.csv`.

## Giới hạn lịch huấn luyện

Inner CV chọn checkpoint cuối8epochs cho adapted ở **5/5 outer folds**.
Bảng dưới chọn cutoff tốt nhất trong inner cho từng epoch, rồi lấy trung bình
qua outer training partitions. Đây là dữ liệu chẩn đoán có overlap, không là
validation độc lập hay lý do thay checkpoint theo outer.

| family    |   epoch |   mean_inner_H |   mean_inner_R |
|:----------|--------:|---------------:|---------------:|
| adapted   |       1 |        0.58438 |        0.89854 |
| adapted   |       2 |        0.61239 |        0.83989 |
| adapted   |       4 |        0.63215 |        0.77543 |
| adapted   |       8 |        0.63979 |        0.77775 |
| frozen    |       1 |        0.56890 |        0.97901 |
| frozen    |       2 |        0.57371 |        0.92278 |
| frozen    |       4 |        0.58827 |        0.86225 |
| frozen    |       8 |        0.61004 |        0.87641 |
| incumbent |       0 |        0.63624 |        0.80119 |

Nếu checkpoint cuối được chọn thường xuyên thì kết quả chỉ kiểm tra lịch1/2/4/8;
không chứng minh encoder đã hội tụ hoặc training lâu hơn vô ích. Đồng thời không
có cơ sở ngoại suy rằng tăng epochs sẽ tự đạt0.7. Chưa thử full transformer fine-tune.

So sánh FP ở ngưỡng inner recall gần0.8 phải xem cùng outer recall thực tế trong
`recall80_summary.csv`; hai nhánh không bị ép có cùng outer recall. Không gán chênh
lệch H hoàn toàn cho việc giảm FP.
