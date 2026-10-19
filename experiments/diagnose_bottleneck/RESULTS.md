# Thí nghiệm xác định nút thắt

## Kết luận từ các phép thử đã chạy

**Chưa có bằng chứng embedding hiện tại là nguyên nhân chính, cũng chưa có bằng chứng đổi head phi tuyến hay bổ sung metadata sẽ giải quyết phần lớn lỗi.** Kết luận này chỉ áp dụng các mô hình và cách biểu diễn được đo ở đây; không chứng minh encoder/head hiện tại là tối ưu.

- Đổi SPECTER2 regression sang retrieval với cùng Ridge: H 0.61949 → 0.61567, giảm ở cả 5 folds. MiniLM + Ridge 0.60582, cũng giảm ở cả 5 folds. Các phép đổi embedding generic này không sửa nút thắt.
- Giữ embedding và thay Ridge bằng RBF: H 0.62006, chỉ tăng trung bình 0.00057 và tốt hơn 2/5 folds. Train MAE giảm rõ hơn validation MAE; AUC chọn/bỏ gần như không tăng. Mô hình linh hoạt hơn chủ yếu khớp train tốt hơn, chưa cải thiện khả năng phân biệt sở thích ngoài train một cách lớn.
- Thêm cả author/venue/year: H 0.61724; chỉ thêm venue 0.62004. Chưa thấy thiếu metadata là nguyên nhân chi phối dưới phép mã hóa/ghép đã thử. Metadata có thể cứu vài bài riêng mà tổng thể không tốt hơn.
- Center vector theo mean train rồi L2: H 0.61510. Cosine cao của embedding regression chưa đủ để gọi là lỗi; đổi hình học đơn giản này không giúp.
- Có **280 mục cùng sai chọn/bỏ ở cả 8 họ text**: 222 FP và 58 FN. Đây là nhóm khó chung để điều tra tiếp, không phải bằng chứng nhãn gốc sai.

Phải điều chỉnh giả thuyết trước đó: không có cơ sở quy nguyên nhân chính cho việc head còn tuyến tính hoặc thiếu author/venue/year. **Giả thuyết còn lại cần kiểm chứng** là pretrained features và tín hiệu hiện có chưa phân biệt đủ sở thích cá nhân; cũng có thể còn giới hạn từ mục tiêu học, chất lượng input hoặc lượng dữ liệu. Thí nghiệm này chưa tách được các khả năng đó. Muốn kiểm chứng tiếp cần so sánh fine-tuning theo nhãn sở thích với frozen encoder dưới validation phù hợp, hoặc kiểm tra thêm chất lượng/tín hiệu dữ liệu; không thể khẳng định nguyên nhân chỉ từ việc nhiều model cùng sai.

## Thiết kế và giới hạn

Đã chạy **17 họ / 138 cấu hình** được khai báo trong `outputs/protocol.json` trước đánh giá. Dùng đúng 5 outer folds và 4 inner folds từ v2. Mỗi outer train tự chọn tham số/ngưỡng bằng inner CV. Không dùng nhãn outer validation để chọn cấu hình trong từng họ. Toàn bộ nhãn giữ nguyên từ ZIP; nhóm trùng title/identifier không giao qua train/validation.

Đối chiếu theo cặp giữ cùng các yếu tố còn lại: đổi representation với cùng Ridge hoặc RBF; đổi head trên cùng representation; thêm metadata vào SPECTER2 regression; thêm phép chuẩn hóa center-train-mean + L2. Hyperparameter grid của cùng head giống nhau giữa representations.

Head tuyến tính: Ridge alpha `[.001,.01,.1,1,10,100]`. Head phi tuyến: kernel ridge RBF alpha `[.01,.1,1]`, gamma multiplier `[.25,1,4]` trên median squared distance **tính chỉ trong training fold**, target trừ mean train rồi cộng lại. Vocab author/venue/TF-IDF, scaler năm và mean vector đều fit trong từng training fold. Không fit TF-IDF hoặc chuẩn hóa thống kê bằng cả dataset.

Các score là **trung bình H của 5 outer folds**, FP/FN là cộng 2.494 OOF. Model trong một họ có thể chọn tham số khác nhau mỗi fold. Đây không phải điểm đã xác nhận cho một model cuối fit toàn bộ train. Các fold dùng training sets chồng lấn; không gọi fold wins hoặc SD là kiểm định ý nghĩa thống kê.

Dataset và OOF cũ đã được xem khi xây dựng thí nghiệm. Nested CV hạn chế overfit lúc tune, không loại bỏ hoàn toàn ảnh hưởng của quá trình nghiên cứu thích nghi trên cùng dữ liệu.

## Đổi embedding và đổi head

| family | score_mean | score_std | FP | FN | read_auc | validation_mae |
| --- | --- | --- | --- | --- | --- | --- |
| minilm_rbf | 0.60862 | 0.01344 | 463 | 327 | 0.73438 | 0.97828 |
| minilm_ridge | 0.60582 | 0.01480 | 471 | 321 | 0.72676 | 1.00214 |
| regression_rbf | 0.62006 | 0.00986 | 417 | 297 | 0.76149 | 0.96144 |
| regression_ridge | 0.61949 | 0.01598 | 414 | 293 | 0.76197 | 0.97318 |
| retrieval_rbf | 0.61797 | 0.01572 | 454 | 278 | 0.75452 | 0.95805 |
| retrieval_ridge | 0.61567 | 0.01804 | 451 | 276 | 0.74750 | 0.97973 |
| tfidf_rbf | 0.61503 | 0.01090 | 467 | 272 | 0.74726 | 1.01304 |
| tfidf_ridge | 0.61549 | 0.01150 | 522 | 203 | 0.74792 | 0.98199 |

SPECTER2 retrieval dùng cùng base và cùng CLS/max512 như regression, chỉ đổi adapter; đây là phép so sánh gần kiểm soát nhất về embedding. MiniLM đổi cả encoder/tokenizer/pooling, là lựa chọn biểu diễn khác chứ không chỉ khác adapter. MiniLM dùng mean pooling theo attention mask, max512 để cùng ngân sách token (khác default256 của sentence-transformers); do tokenizer khác, số token và phần văn bản bị cắt không hoàn toàn trùng nhau.

Nguồn triển khai: [SPECTER2 model card](https://huggingface.co/allenai/specter2) mô tả adapter retrieval/regression và CLS; [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) mô tả mean pooling/default256. SHA model và input có trong protocol/embedding manifests; không fine-tune encoder trong thí nghiệm này.

## Các chênh lệch có kiểm soát

Delta = model thay đổi trừ model đối chiếu. H/AUC cao hơn tốt hơn, MAE thấp hơn tốt hơn. `folds_improved` chỉ mô tả số fold có H tăng.

| contrast | delta_H | delta_SD | folds_improved | delta_AUC | delta_MAE |
| --- | --- | --- | --- | --- | --- |
| head_regression | 0.00057 | 0.00694 | 2 | -0.00047 | -0.01174 |
| embedding_retrieval_linear | -0.00382 | 0.00307 | 0 | -0.01446 | 0.00654 |
| embedding_minilm_linear | -0.01367 | 0.01000 | 0 | -0.03521 | 0.02895 |
| conditioning | -0.00440 | 0.00656 | 1 | -0.00094 | -0.00669 |
| add_metadata | -0.00225 | 0.00800 | 2 | -0.00043 | -0.00191 |
| add_venue | 0.00054 | 0.00446 | 2 | 0.00347 | -0.02060 |
| add_authors | -0.00901 | 0.00732 | 0 | -0.01096 | 0.00741 |
| add_year | -0.00330 | 0.00553 | 2 | -0.00348 | 0.00101 |

Toàn bộ các cặp và delta từng fold nằm trong `outputs/paired_contrasts.csv`; bảng này không lựa chọn thêm họ model để báo một điểm tổng thắng cuộc.

## Metadata riêng và khi kết hợp text

| family | score_mean | FP | FN | read_auc | validation_mae |
| --- | --- | --- | --- | --- | --- |
| authors_ridge | 0.55375 | 889 | 16 | 0.59227 | 1.17126 |
| metadata_ridge | 0.55864 | 887 | 21 | 0.63581 | 1.11443 |
| regression_plus_authors_ridge | 0.61048 | 395 | 338 | 0.75101 | 0.98060 |
| regression_plus_metadata_ridge | 0.61724 | 458 | 249 | 0.76154 | 0.97127 |
| regression_plus_venue_ridge | 0.62004 | 436 | 265 | 0.76544 | 0.95258 |
| regression_plus_year_ridge | 0.61619 | 424 | 299 | 0.75849 | 0.97419 |
| regression_ridge | 0.61949 | 414 | 293 | 0.76197 | 0.97318 |
| venue_ridge | 0.55377 | 903 | 0 | 0.62191 | 1.12118 |
| year_ridge | 0.52730 | 903 | 0 | 0.54503 | 1.20924 |

Author là tên đầy đủ chuẩn hóa Unicode/case/whitespace rồi tách dấu phẩy/chấm phẩy, multi-hot + L2; không dùng nhãn tác giả được tính từ cả dataset. Venue one-hot; year chuẩn hóa bằng train, nhân0.25. Metadata tổng hợp được L2; khi ghép embedding, trọng số `[.25,1]` được chọn bên trong. Một phép ghép đơn giản không cải thiện không chứng minh metadata không có ích dưới mọi mô hình.

Author còn hạn chế coverage: cộng outer validation có 1638/2.494 mục có ít nhất một author đã gặp trong outer train, 192 mục không có author và 664 mục chỉ có author chưa gặp. Không gán điểm thấp cho author chưa gặp; bộ mã hóa bỏ qua các tên ngoài vocab train.

## Dấu hiệu quá khớp và hình học

| family | train_mae | validation_mae |
| --- | --- | --- |
| minilm_rbf | 0.50457 | 0.97828 |
| minilm_ridge | 0.96690 | 1.00214 |
| regression_centered_ridge | 0.87342 | 0.96649 |
| regression_plus_metadata_ridge | 0.75898 | 0.97127 |
| regression_rbf | 0.82466 | 0.96144 |
| regression_ridge | 0.92909 | 0.97318 |
| retrieval_rbf | 0.57347 | 0.95805 |
| retrieval_ridge | 0.90035 | 0.97973 |
| tfidf_rbf | 0.49065 | 1.01304 |
| tfidf_ridge | 0.53463 | 0.98199 |

Train MAE thấp hơn validation MAE là chênh lệch khớp mẫu/khái quát hóa, không tự chứng minh tất cả lỗi là overfit. Lựa chọn head tối ưu theo H chứ không theo MAE, nên hai thước đo có thể không cùng tăng.

| space | cosine_p05 | cosine_median | cosine_p95 |
| --- | --- | --- | --- |
| regression | 0.94657 | 0.97742 | 0.98904 |
| retrieval | 0.79531 | 0.85909 | 0.90364 |
| minilm | 0.06249 | 0.28847 | 0.50114 |

Cosine cao giữa các vector chưa chứng minh embedding hỏng: Ridge có intercept nên đã loại ảnh hưởng mean khi fit; RBF được scale theo khoảng cách train thay vì dùng gamma mặc định không phù hợp. Phép center+L2 được đo riêng để xem thay hình học có giúp hay không, không dựa vào histogram cosine để kết luận.

## Ví dụ và kết quả kiểm tra

Xem `outputs/example_comparison.csv` cho ID 217,184,519,1344,3062,1883,1674: rating và chọn/bỏ riêng cho mỗi biểu diễn/head. `error_changes.csv` đếm số lỗi sửa được và phát sinh so với SPECTER2 regression+Ridge trong cùng thí nghiệm; `errors_by_venue.csv` tách nhóm venue.

Anchor alpha=0.1 (chọn ngưỡng inner, cùng splits) đạt H trung bình **0.61949**, dùng để đối chiếu phép ablation sạch v2 (~0.61958). Lượt này chuẩn hóa dense features bằng float64 và dùng Ridge LSQR tol=1e-6 thay vì default1e-4 trước đó; có sai khác số học nhỏ. Các phép so sánh trong bảng đều dùng cùng thiết lập mới. Baseline Ridge trong bảng chính tìm alpha trên grid rộng hơn v2 (cả 5 fold vẫn chọn alpha=0.1); không coi tác động mở grid hoặc sai khác số học là tác động thay embedding/head.

![Kết quả từng fold](outputs/diagnostic_comparison.png)

Chạy kiểm tra: `python experiments/diagnose_bottleneck/verify_results.py`. File `outputs/verification.json` chỉ xuất sau khi kiểm tra hash input/splits, nhãn, group separation, tái chọn toàn bộ cấu hình bằng inner predictions và tính lại từng outer score từ OOF priority thành công.

## Chạy lại

```bash
python experiments/diagnose_bottleneck/run_diagnostics.py all
python -m unittest discover -s experiments/diagnose_bottleneck -p 'test_*.py' -v
python experiments/diagnose_bottleneck/verify_results.py
python experiments/diagnose_bottleneck/analyze_results.py
```

Môi trường hiện tại: `/tmp/assignment-specter2-venv/bin/python`, dependencies theo `requirements.txt`. Cache embeddings và prediction từng config/fold cho phép kiểm tra mà không encode/fit lại. Không tạo submission hoặc thay model v2.
