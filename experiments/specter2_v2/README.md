# Bản sửa dữ liệu và mô hình v2

V2 sửa các lỗi nguồn/abstract đã xác nhận, thử classifier chọn bài và kết hợp SPECTER2 với TF-IDF, rồi đánh giá bằng grouped nested cross-validation. Dữ liệu, embedding và model baseline ở `experiments/specter2` được giữ nguyên để đối chiếu.

## Kết quả

Các con số dưới đây là trung bình **5 outer folds** trên toàn bộ 2.494 record có nhãn. Mỗi outer fold dùng **4 inner folds** để chọn cấu hình và ngưỡng; các nhóm trùng title/identifier không được chia qua train/validation. Đây không phải cùng phép đo với holdout 500 mục đạt 0.6193 trước đó.

| Quy trình / họ mô hình | Score trung bình | SD giữa folds | FP cộng 5 folds | FN cộng 5 folds |
|---|---:|---:|---:|---:|
| Baseline SPECTER2 trên input cũ, tune lại trong từng outer train | 0.61777 | 0.01531 | 428 | 289 |
| SPECTER2 với input đã sửa, cùng alpha=0.1 | 0.61958 | 0.01590 | 413 | 294 |
| SPECTER2 sạch + mô hình hai tầng | 0.62043 | 0.01697 | 424 | 275 |
| Họ hybrid SPECTER2 sạch + TF-IDF | **0.62242** | 0.01100 | 399 | 307 |
| Họ hybrid hai tầng | 0.61946 | 0.01838 | 444 | 275 |
| Toàn bộ quy trình tự chọn cấu hình v2 trong inner CV | **0.61987** | 0.01410 | 397 | 321 |

FP/FN ở đây là tổng trên 2.494 dự đoán outer OOF, không so trực tiếp với 95/45 lỗi của holdout 500 mục cũ. SD là độ phân tán giữa năm folds, không phải khoảng tin cậy hoặc bằng chứng ý nghĩa thống kê.

Họ hybrid là một lựa chọn đã định nghĩa trước, nhưng mỗi outer fold có thể chọn trọng số khác nhau bằng inner CV. Vì vậy 0.62242 là đánh giá của **họ hybrid với lựa chọn trọng số bên trong**, không phải điểm đã đo của riêng cấu hình 50/50 cuối cùng. Con số thận trọng để đánh giá toàn bộ thuật toán chọn cả họ mô hình lẫn tham số là 0.61987, tăng khoảng **0.00210** so với baseline 0.61777 trên cùng outer splits. Mức tăng nhỏ và chưa bảo đảm tăng điểm Kaggle.

Mô hình hai tầng giảm nhẹ cả FP và FN so với baseline trong phép đo này. Hybrid giảm FP nhưng hy sinh recall. Các lỗi sở thích chưa được giải quyết hết.

## Dữ liệu được sửa như thế nào?

`outputs/abstracts_cleaned.csv` là bản dữ liệu dùng cho v2, gồm đủ 3.090 ID và các cột `abstract_usable_v2`, `quality_flags_v2`, `cleaning_actions_v2`. Bản gốc `abstracts_collected.csv` không bị ghi đè.

- **13 abstract:** bỏ `documentclass`, `usepackage`, `setlength` và wrapper `begin/end{document}`; giữ nội dung toán và các mệnh đề. Input ID 1674 giảm từ 1.655 xuống 394 tokens, ID 1883 từ 1.311 xuống 341.
- **ID 175:** dừng ở đề mục `Bug Report`, giữ đúng đoạn Abstract thay vì kéo vào phần thân bài. Parser PDF dùng về sau đã có thêm ranh giới cho các đề mục không đánh số.
- **ID 446:** thay abstract luận án cùng tên bằng abstract của [đúng bài ICLP 2016, DOI 10.4230/OASIcs.ICLP.2016.16](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.ICLP.2016.16). Kiểm tra title và năm từ metadata publisher trước khi chấp nhận.
- **ID 392:** kiểm tra [đúng bản ICLP 2018](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.ICLP.2018.7). Abstract chính thức chỉ là một câu dẫn sang công trình khác. V2 giữ câu này làm provenance nhưng dùng title cho model, thay vì abstract journal về sau.
- **100 nguồn Crossref:** đã kiểm tra metadata title, type, năm, DOI. Hai vấn đề publication type/version ở trên được xử lý. Có 16 cảnh báo năm khác nhau trong trường hợp DOI dataset trùng chính DOI nguồn; những trường hợp này được ghi cảnh báo, không tự đổi năm hay tuyên bố nguồn sai chỉ từ một trường metadata.
- 35 mục có identity-review note vẫn dùng title theo chính sách baseline; không sửa nhãn hay đoán nhãn theo nội dung.

Có **16 model input thay đổi**. Chỉ encode lại 16 mục đó; 3.074 vector khác được dùng nguyên từ baseline để tránh đưa sai khác số học do batching vào phép so sánh. Số input vượt 512 tokens giảm **18 → 9**. Có 2.974 abstract được phép dùng và 116 mục dùng title. Coverage có văn bản và số abstract đủ điều kiện dùng là hai con số khác nhau.

Lượt sửa chưa phải kiểm định thủ công toàn bộ 3.010 abstract đã collect. Nguồn ngoài Crossref vẫn có thể có lỗi chưa phát hiện. Các thay đổi và cảnh báo nằm trong [data_change_ledger.csv](outputs/data_change_ledger.csv); audit nguồn nằm trong [crossref_source_audit.csv](outputs/crossref_source_audit.csv). Quy tắc sửa dữ liệu không sử dụng `Label`.

## Các mô hình đã thử

1. SPECTER2 + Ridge trên input cũ: baseline tham chiếu, alpha `[0.1, 1, 10, 100]`.
2. SPECTER2 sạch + Ridge: alpha `[0.03, 0.1, 0.3, 1]`.
3. TF-IDF sạch + Ridge: alpha `[0.3, 1, 3]`.
4. Hybrid hai điểm hồi quy: SPECTER2 Ridge alpha=0.1, TF-IDF Ridge alpha=1; trọng số SPECTER2 `[0.25, 0.5, 0.75]`.
5. Hai tầng trên từng biểu diễn: Logistic Regression học `P(Label > 1)`, Ridge trên các mẫu dương học `E[(Label-1)/4 | Label > 1]`; expected gain là tích hai giá trị. Chọn bài theo ngưỡng xác suất và xếp theo expected gain.
6. Hybrid hai tầng: kết hợp hai xác suất và expected gain bằng các trọng số trên.

Tổng cộng 21 cấu hình nhỏ được ghi vào `protocol.json` trước vòng outer CV. Ngưỡng được chọn bằng điểm trung bình inner folds theo harmonic mean Q/R trong Evaluation chi tiết của đề. Không dùng ID/DOI hoặc nhãn làm feature. TF-IDF được fit lại bên trong mỗi training fold. Embedding pretrained đóng băng, không fine-tune encoder trong lượt này.

Raw/clean Ridge có grid khác nhau trong phần tìm mô hình, nên không dùng chênh lệch hai hàng đó để quy toàn bộ hiệu quả cho việc sửa dữ liệu. File [controlled_data_ablation.csv](outputs/controlled_data_ablation.csv) giữ cùng alpha=0.1, cùng outer training sets và cách chọn ngưỡng trong inner folds: input cũ 0.61777, input sửa 0.61958. Đây là phép kiểm tra tác động dữ liệu riêng, không tham gia chọn lại model cuối.

## Model và submission cuối

Sau nested evaluation, quy trình chọn trên 4-fold inner OOF của toàn bộ train quyết định:

- **50% SPECTER2 Ridge alpha=0.1 + 50% TF-IDF Ridge alpha=1.0**.
- Điểm inner CV 0.62330; cấu hình tốt thứ hai 0.62301, nên không coi chênh lệch này là bằng chứng vượt trội lớn.
- Expected rating `p = 0.5*p_specter + 0.5*p_tfidf`.
- Chọn khi `(p-1)/4 > 0.275`, tương đương `p > 2.1`.
- Priority khi chọn là `10*(p-1)/4`, còn lại là 0. Không clip để tránh tạo thêm hòa điểm.

Hai hồi quy được refit trên toàn bộ 2.494 nhãn gốc. [submission_v2.csv](outputs/submission_v2.csv) có đúng 596 ID theo thứ tự template, **347 được đề xuất, 249 priority=0**. [model_v2.joblib](outputs/model_v2.joblib) chứa model, TF-IDF vectorizer, cấu hình và provenance. Chưa submit lên Kaggle.

Lựa chọn cuối dựa trên inner CV toàn bộ train, không chọn họ mô hình theo bảng điểm outer vừa nhìn thấy. Dù vậy, thiết kế v2 xuất phát từ failure analysis của dữ liệu này; nested CV hạn chế việc chọn tham số quá khớp nhưng không biến đây thành một benchmark hoàn toàn chưa từng được quan sát.

## Chạy lại

Từ thư mục Assignment với Python 3.10 và môi trường đã cài dependencies:

```bash
python -m pip install -r experiments/specter2_v2/requirements.txt
python experiments/specter2_v2/prepare_data.py
python experiments/specter2_v2/run_v2.py all
python -m unittest discover -s experiments/specter2_v2 -p 'test_*.py' -v
python experiments/specter2_v2/verify_outputs.py
```

Lần thực thi hiện tại dùng `/tmp/assignment-specter2-venv/bin/python`. Cần baseline `experiments/specter2/outputs/embeddings.npz` và manifest đi kèm để đối chiếu và tái sử dụng vector. Nếu chưa có, chạy `experiments/specter2/run_experiment.py embed` trước. Source metadata/HTML được cache trong `source_cache`; các lần chạy lại không phải tra lại 100 DOI. Nếu input thay đổi, chương trình từ chối dùng embedding cache có checksum khác.

Đã qua **7 kiểm tra đơn vị** và các kiểm tra artifact: checksum dữ liệu cũ, sửa nguồn/ranh giới/LaTeX, không giao nhóm qua outer/inner folds, giữ nguyên toàn bộ nhãn, tính lại từng điểm từ priority OOF, vector không đổi được giữ nguyên, load lại model và tái tạo lựa chọn/thứ tự, CSV đủ 596 ID hợp lệ.

## Các file để xem

- [RESIDUAL_FAILURES.md](RESIDUAL_FAILURES.md): giải thích mức tăng nhỏ, chuyển đổi lỗi baseline/v2 và các lỗi còn lại theo bài, nhãn, venue và abstract.
- [residual_failure_rows.csv](outputs/residual_failure_rows.csv): 2.494 dự đoán outer OOF kèm trạng thái đúng/sai cũ và mới; chạy lại bằng `python experiments/specter2_v2/analyze_residuals.py`.
- [nested_summary.csv](outputs/nested_summary.csv): so sánh các họ trên cùng nested CV.
- [nested_fold_metrics.csv](outputs/nested_fold_metrics.csv): điểm từng fold, cấu hình và ngưỡng được chọn bên trong.
- [nested_oof_predictions.csv](outputs/nested_oof_predictions.csv): dự đoán từng mục khi nằm ngoài training fold.
- [nested_splits.csv](outputs/nested_splits.csv): membership outer/inner để kiểm tra leakage.
- [final_inner_search.csv](outputs/final_inner_search.csv): quyết định cấu hình cuối từ train OOF.
- [test_prediction_details.csv](outputs/test_prediction_details.csv): expected gain, ngưỡng và priority test.
- [data_manifest.json](outputs/data_manifest.json), [embedding_manifest.json](outputs/embedding_manifest.json): hashes và nguồn của dữ liệu/vector.
