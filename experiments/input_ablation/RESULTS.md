# Kết quả thực hiện kế hoạch title/abstract → metadata → adapter → model cuối

## Kết quả toàn bộ quy trình

| method | H | Q | R | FP | FN | missed_high |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline joint SPECTER2 + Ridge | 0.61949 | 0.49972 | 0.81630 | 414 | 293 | 40 |
| Full staged selection workflow | 0.62896 | 0.51718 | 0.80756 | 371 | 307 | 43 |

Chênh lệch H so với baseline: **+0.00947**. Số quyết định cũ được sửa **241**, lỗi mới phát sinh **212**; tổng lỗi chọn/bỏ **707 → 678**. Điểm phụ thuộc mức nhãn và thứ tự nên tổng lỗi không thay thế H. Đánh giá đúng cả quy trình: mỗi outer train chọn nguồn/model bằng điểm inner của chính outer train, rồi mới tính điểm outer. Không lấy họ có outer score cao nhất làm điểm của model cuối.

Đây là 2.494 dự đoán ngoài tập fit trên cùng grouped outer folds đã dùng trước đây. Dataset/outer results đã được quan sát để thiết kế nghiên cứu nên không gọi đây là benchmark hoàn toàn mới hoặc khẳng định tăng điểm Kaggle.

Điểm metadata-only policy0.63249 và điểm full policy ở bảng trên là hai đại lượng khác nhau. Full policy còn cho phép chọn các nhánh neural bằng inner evaluation; việc mở rộng lựa chọn có thể chọn nhầm cấu hình gần điểm nhau và làm outer score giảm. Neural dùng một calibration group còn mô hình truyền thống dùng mean4inner, nên các ước lượng lựa chọn có độ nhiễu khác nhau. Điểm toàn bộ workflow đã tính cả ảnh hưởng này; không dùng score của họ tốt nhất để che phần giảm do model selection.

## 1. Title/abstract ablation

| family | H | H_std | Q | R | FP | FN | missed_high | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection_procedure | 0.62929 | 0.01816 | 0.51214 | 0.81791 | 384 | 290 | 38 | 0.78644 |
| specter_joint | 0.61949 | 0.01598 | 0.49972 | 0.81630 | 414 | 293 | 40 | 0.76197 |
| specter_separate | 0.62782 | 0.01993 | 0.51519 | 0.80493 | 372 | 310 | 44 | 0.78721 |
| specter_title | 0.63010 | 0.01709 | 0.52104 | 0.80203 | 364 | 317 | 38 | 0.79048 |
| tfidf_joint | 0.61549 | 0.01150 | 0.47575 | 0.87278 | 522 | 203 | 23 | 0.74792 |
| tfidf_separate | 0.62768 | 0.01378 | 0.52772 | 0.77693 | 346 | 356 | 49 | 0.77912 |
| tfidf_title | 0.62281 | 0.01526 | 0.49545 | 0.83924 | 452 | 256 | 38 | 0.77182 |

Chạy đúng sáu tổ hợp đã hứa, 72 cấu hình alpha/trọng số; mỗi họ tự chọn tham số/cutoff trong 4 inner folds. Title/abstract riêng được mã hóa thành hai khối rồi L2; trọng số abstract `[.25,.5,1,2]`. Abstract thiếu/withheld thành vector0. Joint SPECTER2 vectors dùng nguyên cache v2; title-only/abstract-only encode bằng cùng base/adapter SHA. TF-IDF fit vocab riêng trong mỗi training partition.

Title-only SPECTER2 đạt H 0.63010 so với joint 0.61949, tăng ở cả 5 folds; AUC tăng khoảng 0.0285. Phép đối chứng giữ cùng alpha=0.1 cũng cho title-only H khoảng 0.63012, joint 0.61949; không chỉ là hiệu ứng mở grid alpha. Baseline joint được tái tạo khớp prediction/priority cũ tới1e-10.

Kết luận giới hạn: **cách dùng abstract trong các pipeline đang thử làm giảm khả năng dự đoán nhãn so với title-only hoặc tách khối**. Không suy ra abstract vô ích trong mọi mô hình, không suy ra người gán nhãn chỉ đọc title. Title-only giảm FP nhưng có thể giảm recall. Quy trình tự chọn input trong inner đạt 0.62929; không đánh đồng nó với 0.63010 của họ title-only.

## 2. Audit và phép thử metadata

Kiểm tra đăng ký DOI và mục lục CEUR không sử dụng Label: xác minh author cho **1605/3.090** mục, có **239** khác biệt bộ họ tác giả (56 bổ sung cho trường gốc rỗng, 183 thay đổi danh sách không rỗng), **23** thay đổi năm trong phạm vi1 năm. Có 23 ca năm khác nhiều hơn được gắn cờ, 35 ca không đạt kiểm tra danh tính nguồn. Tất cả có ledger, nguồn và bản gốc. Chưa xác minh toàn bộ dataset: DOI API thành công 1379/1813 DOI duy nhất; các mục chưa xác minh giữ metadata gốc. Metadata source/audit có thể còn thiếu hoặc chưa hoàn chỉnh.

| family | H | H_std | Q | R | FP | FN | missed_high | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_all | 0.62658 | 0.01806 | 0.50407 | 0.82807 | 403 | 273 | 34 | 0.78152 |
| clean_authors | 0.61975 | 0.01506 | 0.49489 | 0.83473 | 445 | 263 | 33 | 0.76973 |
| clean_year | 0.63087 | 0.01867 | 0.51935 | 0.80462 | 358 | 311 | 40 | 0.78679 |
| input_policy | 0.62929 | 0.01816 | 0.51214 | 0.81791 | 384 | 290 | 38 | 0.78644 |
| original_all | 0.62685 | 0.01742 | 0.50319 | 0.83127 | 408 | 268 | 35 | 0.78099 |
| selection_procedure | 0.63249 | 0.01979 | 0.53122 | 0.78310 | 323 | 346 | 45 | 0.78896 |
| venue | 0.63216 | 0.02023 | 0.53045 | 0.78372 | 326 | 345 | 43 | 0.79012 |

Original/clean author dùng cùng surname encoding để tránh so sánh tên đầy đủ với họ rút gọn. Vocab author, venue onehot và scaler năm đều fit trong train fold. Text input của mỗi outer fold được chọn chỉ bằng stage1 inner CV. Bộ giải Ridge metadata dùng centered linear-kernel eigensolve để tái sử dụng factorization giữa các alpha; cùng mục tiêu Ridge với intercept, đã kiểm tra đối chiếu primal Ridge. Cache partial từ bộ giải chậm được lưu riêng, không trộn vào kết quả.

Metadata sạch không tạo cải thiện tổng H rõ so với metadata gốc trong cách ghép này. Venue giúp ở một số folds nhưng chủ yếu đổi đánh đổi Q/R. Global metadata policy H 0.63249, FP323/FN346, bỏ45 bài nhãn4–5. Vì recall giảm quá0.01 và missed-high tăng, quy trình tự chuyển sang bước adapter theo gate đã công bố, dù H tăng.

## 3. Đối chứng representation học theo sở thích

Đã chạy đủ ba đối chứng trên 5 outer folds. Lịch train cố định 3 epochs, dùng inner_fold0 làm calibration để chọn threshold; không search epoch/lr trên outer. Nhãn outer không đi vào mining hoặc optimizer. Seed thứ hai fit lại các outer train, giữ cutoff của seed chính, là kiểm tra độ nhạy khởi tạo chứ không phải một lần tune độc lập.

| family | H | H_std | Q | R | FP | FN | missed_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adapter_hard | 0.60656 | 0.01329 | 0.48681 | 0.81179 | 489 | 299 | 55 |
| adapter_random | 0.62089 | 0.01683 | 0.49808 | 0.84117 | 463 | 255 | 43 |
| frozen | 0.62122 | 0.01716 | 0.48401 | 0.87156 | 514 | 206 | 25 |

Kiểm tra seed thứ hai (cutoff giữ cố định từ seed chính):

| family | H | H_std | FP | FN |
| --- | --- | --- | --- | --- |
| adapter_hard | 0.60964 | 0.01234 | 525 | 242 |
| adapter_random | 0.62021 | 0.02020 | 472 | 249 |
| frozen | 0.62131 | 0.01208 | 522 | 195 |

Nếu bước adapter được chạy: ba nhánh dùng MiniLM cùng SHA, max256, cùng khởi tạo bottleneck adapter seq_bn reduction16, cùng MLP/loss. Nhánh frozen đóng băng cả adapter; hai nhánh còn lại chỉ train113.040 tham số adapter và head, giữ base cố định. Input mode/weight lấy từ stage1 inner của từng outer; do backbone/budget token khác SPECTER2 nên suy luận nhân quả về cập nhật representation chỉ áp dụng giữa ba nhánh neural có đối chứng.

Loss gồm BCE đọc/không đọc, MSE gain trên mẫu dương và pairwise softplus. Hard negatives nằm trong top5 bài nhãn1 gần nhất thuộc cùng venue nếu có, tìm chỉ trong training partition. Không lấy danh sách 280 lỗi outer đã xem để gán trọng số fit. Preflight có chạy một số inner folds thừa; trước khi xem bất kỳ neural outer score nào đã chốt một calibration group cho mỗi outer, archive các preflight thừa và không dùng để chọn model.

Đây là một lịch huấn luyện nhỏ, cố định; kết quả âm không loại trừ mọi phương án fine-tune hoặc fine-tune SPECTER2. Đọc kết quả seed để đánh giá độ nhạy, không chọn seed cao điểm hơn để báo cáo.

## 4. Chọn model cuối và submission

Chọn lại input trên 4 grouped inner folds của toàn bộ train, seed20261117; chọn metadata trên các folds đó. Nếu có neural, cutoff/nhánh được đối chiếu trên calibration group0 của cùng final inner split. Kết quả outer không chọn trực tiếp model cuối.

- Nguồn được chọn: **metadata**.
- Text config: `{"family": "specter_title", "representation": "specter", "mode": "title", "weight": null, "alpha": 0.01, "key": "c037", "threshold": 0.25, "inner_score": 0.6341637794361416}`.
- Cấu hình cuối: `{"key": "m044", "family": "venue", "kind": "venue", "weight": 1.0, "alpha": 0.01, "threshold": 0.25, "inner_score": 0.6363946914943209}`.
- CSV đủ596 ID theo template; **370** priority>0, **226** priority=0.
- Artifact nằm tại [final_outputs](final_outputs), gồm model/recipe, dự đoán test, kết quả full-train inner search và OOF của toàn bộ workflow.
- Chưa submit lên Kaggle. Nhãn gốc không bị sửa; raw abstracts, baseline và model v2 được giữ để đối chiếu.

![So sánh từng fold](final_outputs/workflow_comparison.png)

## Lỗi còn lại của workflow

| venue | n | FP | FN |
| --- | --- | --- | --- |
| cav | 657 | 99 | 104 |
| iclp | 456 | 73 | 20 |
| kr | 661 | 99 | 49 |
| lics | 604 | 88 | 128 |
| lpnmr | 116 | 12 | 6 |

| Label | n | FP | FN |
| --- | --- | --- | --- |
| 1 | 903 | 371 | 0 |
| 2 | 514 | 0 | 169 |
| 3 | 438 | 0 | 95 |
| 4 | 367 | 0 | 34 |
| 5 | 272 | 0 | 9 |

Từng ID và chuyển đổi lỗi cũ/mới: [failure_transitions.csv](final_outputs/failure_transitions.csv). Chất lượng phải được xem cùng H, recall và mức nhãn; không coi cải thiện của một vài ca đã biết là bằng chứng tổng quát hóa.

## Tái tạo và kiểm tra

```bash
python experiments/input_ablation/run_ablation.py all
python experiments/input_ablation/audit_metadata.py
python experiments/input_ablation/run_metadata.py
# Khi stage_decision yêu cầu adapter:
python experiments/preference_adapter/run_adapter.py
python experiments/input_ablation/finalize_workflow.py
python experiments/input_ablation/verify_and_summarize.py
python experiments/input_ablation/verify_metadata.py
python experiments/input_ablation/verify_final.py
python experiments/input_ablation/write_report.py
```

Môi trường: `/tmp/assignment-specter2-venv/bin/python`; requirements kế thừa `../diagnose_bottleneck/requirements.txt`. Mỗi stage có protocol, cache prediction, hash dữ liệu và nguồn. Các verification files chỉ được xuất sau khi tính lại quyết định/score thành công. `audit_metadata.py` có giới hạn thời gian lấy DOI để nguồn chưa phản hồi được ghi chưa xác minh; cache làm các lượt sau có thể tăng coverage. Một snapshot audit khác cần thư mục thí nghiệm mới vì protocol kiểm tra hash.
