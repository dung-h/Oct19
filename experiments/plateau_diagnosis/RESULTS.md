# Chẩn đoán khi tối ưu bị chững — kết quả

## Điều đã xác định được

**Trong phạm vi số mẫu đã thử, thêm các nhãn train thật giúp CatBoost phân biệt đọc/không đọc tốt hơn và giảm sai số rating. Điều này phù hợp với hạn chế về lượng ví dụ sở thích, chưa chứng minh tăng dữ liệu ngoài phạm vi này sẽ tiếp tục cho cùng mức lợi ích.**

- CatBoost, 25% → 100% outer train: validation AUC **+0.04791**, MAE **-0.09471**, AUC tăng ở **5/5 folds**.
- 75% → 100%: validation AUC **+0.00424**, MAE **-0.01321**, AUC tăng ở **4/5 folds**.
- Ở full size, train AUC cao hơn validation **0.07470** với CatBoost và **0.05065** với Ridge. Đây là khoảng cách tổng quát hóa; có thể gồm overfitting của head và khác biệt kích thước nguồn tham chiếu. Không suy ra một nguyên nhân duy nhất từ khoảng cách này.
- Khảo sát 20 cặp cho thấy văn bản có những khác biệt phương pháp/đóng góp quan sát được, nhưng **chưa biết lý do thật của nhãn sở thích trong cả 20 cặp**. Hai vấn đề này phải tách biệt.

## Learning curves

Giữ nguyên năm outer folds. Trong từng outer train, tạo nested subsets ở 25/50/75/100% theo nhóm, hai seed lấy mẫu. Có 35 bộ fit khác nhau, mỗi bộ fit hai model cố định. Full-size fit chỉ tính một lần; seed lấy mẫu được trung bình trong từng outer fold trước khi trung bình năm folds. Không tìm hyperparameter hay cutoff mới.

|   fraction | model    | partition   |      n_fit |     AUC |     MAE |       H |   logloss |
|-----------:|:---------|:------------|-----------:|--------:|--------:|--------:|----------:|
|    0.25000 | catboost | train       |  499.60000 | 0.96140 | 0.74929 | 0.70438 |   0.84957 |
|    0.25000 | catboost | validation  |  499.60000 | 0.75155 | 0.97087 | 0.61390 |   1.36787 |
|    0.25000 | ridge    | train       |  499.60000 | 0.86985 | 0.75973 | 0.67546 | nan       |
|    0.25000 | ridge    | validation  |  499.60000 | 0.75863 | 0.95740 | 0.61701 | nan       |
|    0.50000 | catboost | train       |  998.10000 | 0.91205 | 0.78656 | 0.67531 |   0.96624 |
|    0.50000 | catboost | validation  |  998.10000 | 0.77604 | 0.91530 | 0.62706 |   1.32131 |
|    0.50000 | ridge    | train       |  998.10000 | 0.85850 | 0.78426 | 0.66197 | nan       |
|    0.50000 | ridge    | validation  |  998.10000 | 0.77774 | 0.92147 | 0.62644 | nan       |
|    0.75000 | catboost | train       | 1497.20000 | 0.89042 | 0.79566 | 0.66397 |   1.01999 |
|    0.75000 | catboost | validation  | 1497.20000 | 0.79521 | 0.88938 | 0.63538 |   1.29095 |
|    0.75000 | ridge    | train       | 1497.20000 | 0.85035 | 0.79962 | 0.65450 | nan       |
|    0.75000 | ridge    | validation  | 1497.20000 | 0.78888 | 0.90640 | 0.63109 | nan       |
|    1.00000 | catboost | train       | 1995.20000 | 0.87416 | 0.80625 | 0.65664 |   1.06195 |
|    1.00000 | catboost | validation  | 1995.20000 | 0.79946 | 0.87617 | 0.63752 |   1.28296 |
|    1.00000 | ridge    | train       | 1995.20000 | 0.84357 | 0.80711 | 0.64871 | nan       |
|    1.00000 | ridge    | validation  | 1995.20000 | 0.79291 | 0.89619 | 0.63340 | nan       |

![Learning curves](outputs/learning_curves.png)

[PDF để xuất](outputs/learning_curves.pdf). Dùng AUC/MAE/log-loss để đọc khoảng cách học và tổng quát hóa. Log-loss chỉ áp dụng cho CatBoost; Ridge không cung cấp phân phối xác suất năm nhãn. H có cutoff chẩn đoán cố định: Ridge gain>0.25; CatBoost P(read)>0.50, thứ tự theo expected gain. H này không phải điểm của submission đã tune ngưỡng; train H cũng thay đổi theo độ dài danh sách và không được so với 1 như một mức tối ưu mặc định.

| model    |   delta_AUC_25_100 |   delta_AUC_75_100 |   delta_MAE_25_100 |   delta_MAE_75_100 |   delta_H_25_100 |   delta_H_75_100 |   AUC_gap_full |   MAE_gap_full |   folds_AUC_improve_25_100 |   folds_AUC_improve_75_100 |
|:---------|-------------------:|-------------------:|-------------------:|-------------------:|-----------------:|-----------------:|---------------:|---------------:|---------------------------:|---------------------------:|
| catboost |            0.04791 |            0.00424 |           -0.09471 |           -0.01321 |          0.02362 |          0.00214 |        0.07470 |        0.06992 |                          5 |                          4 |
| ridge    |            0.03429 |            0.00403 |           -0.06121 |           -0.01021 |          0.01639 |          0.00230 |        0.05065 |        0.08907 |                          5 |                          5 |

Chỉ có một model seed cố định; hai seed ở đây thay **cách lấy subset**, không phải hai benchmark độc lập. Các train folds dùng chung nhiều dữ liệu. Các chênh lệch là mô tả paired folds, không có khẳng định ý nghĩa thống kê từ giả định độc lập. Kiến trúc đã được chọn trong nghiên cứu trước trên cùng dữ liệu, nên đây vẫn là chẩn đoán exploratory.

Tăng số mẫu đồng thời tăng số ví dụ huấn luyện, độ phủ history và dữ liệu để fit vocab/SVD. Vì thế learning curve đo lợi ích của thêm mẫu cho **toàn pipeline**, chưa tách riêng tác động của số nhãn cho classifier với tác động của nguồn láng giềng/phần biểu diễn.

## Khảo sát nội dung

Mẫu 20 cặp, bốn cặp mỗi venue, được chọn bằng similarity retrieval cao nhất giữa một bài nhãn 1 và một bài nhãn 4/5; không chọn theo model error. Các bài không trùng nhau trong mẫu và khác duplicate group trong mỗi cặp. Các ví dụ cũ được xuất riêng, không cộng vào mẫu 20 nếu không tự xuất hiện theo quy tắc. ID 217/355 tự xuất hiện dưới quy tắc này.

- Có đủ usable abstract cho **40/40** bài; **34/40** bài có ít nhất một candidate sentence theo discourse cues. Cues chỉ là cách tìm đoạn văn để xem, không phải phân tích ngữ nghĩa đã được xác minh và chưa được đưa vào model.
- Người đọc thấy khác biệt cụ thể về nội dung ở 20/20 cặp đã chọn; tất cả có hai đoạn bằng chứng nguyên văn và offsets. Đây không phải 20 quyết định rằng nhãn 1/4/5 đúng hay sai, hoặc rằng khác biệt vừa tìm được chính là lý do gán nhãn.
- **2/20** cặp có toàn bộ category flags giống nhau: P05 (epistemic semantics) và P16 (infinite-domain CSP). Các vector/title features khác vẫn có thể giữ thông tin; không suy ra toàn bộ model đầu vào giống nhau.
- Theo nhận xét nội dung, 8/20 cặp được bộ cờ thể hiện một phần khác biệt đã mô tả, 12/20 không thể hiện tường minh khác biệt đó. Đây là rubric đọc của một người trên mẫu khó, không phải accuracy tổng quát của extractor.
- P07: `student` xuất hiện trong thông tin Ph.D. của tác giả, nhưng kích hoạt cờ domain education. P08: `tutorial` có rõ trong văn bản nhưng taxonomy document type chưa nhận diện. Literal phrase match đã được lưu đúng; việc diễn giải nó thành ngữ nghĩa tài liệu vẫn có thể sai hoặc thiếu.

[Đọc đủ 20 nhận xét và bằng chứng](outputs/CONTENT_REVIEW.md), [bảng CSV có offsets](outputs/reviewed_pairs.csv), [toàn văn các cặp để kiểm tra](outputs/PAIR_PACKET.md).

## Áp dụng vào hướng đi tiếp theo

1. **Ưu tiên kiểm soát variance và ổn định việc chọn model.** Giữ Ridge và CatBoost hiện tại làm mốc. Learning curve cho biết thêm bao nhiêu lợi ích khi tăng mẫu trong phạm vi có sẵn; không có cơ sở để gọi H≈0.64 là trần bài toán hoặc lập tức tăng model capacity.
2. **Nếu làm extraction tiếp, phải giữ nội dung của mệnh đề cụ thể thay vì thêm nhiều cờ chủ đề.** Ví dụ cần phân biệt “epistemic atoms ở rule heads” với “world views ba giá trị”, hoặc CSP reduction với local/global height-1 identities. Đánh giá chất lượng trích xuất bằng bằng chứng trước khi train lại. Chẩn đoán này chỉ xác định thông tin nào có thể đọc được, chưa kiểm chứng rằng extraction mới tăng H.
3. **Còn thiếu thông tin về quy trình gán nhãn.** Cần biết người gán nhãn đã thấy title, abstract, full text hay thông tin nào khác, và labels có cùng một tiêu chí/thời điểm hay không. Không suy đoán nhãn sai hoặc tự sửa labels. Không tạo thêm nhãn sở thích giả để giải quyết tình trạng ít mẫu.
4. **Đánh giá ứng viên hiện có theo lợi ích và độ ổn định**, không loại chỉ vì chưa tăng 0.02 H. Mốc 0.02 trước đây là mục tiêu nghiên cứu tự đặt, không phải yêu cầu assignment. Chưa có dữ liệu ngoài nghiên cứu này để khẳng định thứ hạng Kaggle hoặc khả năng cải thiện khi thêm dữ liệu tương lai.

## Kiểm chứng và tái lập

Đã kiểm tra nhãn gốc, 35 subset partitions, tính lồng nhau và tách duplicate groups, 105 cross-fitting partitions; tính lại 140 metric rows từ raw predictions. Dự đoán full size khớp thí nghiệm cũ tới 1e-10. Kiểm tra 40 đoạn bằng chứng thủ công và 73 candidate cue spans. Ba unit tests về subset label isolation, cutoff và offsets đều qua. Không tạo submission/model family mới.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/plateau_diagnosis/run_plan.py
```

Đọc [PLAN.md](PLAN.md), [outputs/verification.json](outputs/verification.json), [outputs/curve_summary.csv](outputs/curve_summary.csv), [outputs/endpoint_fold_deltas.csv](outputs/endpoint_fold_deltas.csv). Nguồn phương pháp: [scikit-learn learning curves](https://scikit-learn.org/stable/modules/learning_curve.html), [Cawley–Talbot về model-selection overfitting](https://jmlr.org/papers/v11/cawley10a.html), [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml).
