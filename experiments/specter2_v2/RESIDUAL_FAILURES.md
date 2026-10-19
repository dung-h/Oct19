# Vì sao v2 tăng ít, và các lỗi còn ở đâu?

Phân tích dùng 2.494 dự đoán **outer OOF** đã lưu: mỗi record được dự đoán khi không nằm trong tập train của model đó. Không refit, đổi ngưỡng hay chọn mô hình dựa trên các lỗi dưới đây. V2 ở đây là toàn bộ `selection_procedure`, không phải một model cuối duy nhất. So sánh với baseline trên đúng các outer folds; không so với holdout 500 mục cũ.

## Mức tăng nhỏ phản ánh một đánh đổi thực sự

Điểm trung bình 5 folds: **0.61777 → 0.61987**, tăng **0.00210**, khoảng **0.34% tương đối**. SD giữa folds của v2 là 0.01410; mức tăng này chưa chứng minh cải thiện đáng tin cậy ngoài tập dữ liệu này. SD không phải confidence interval.

- FP: **428 → 397** (giảm 31); FN: **289 → 321** (tăng 32).
- Sửa được **80** quyết định: 55 FP thành TN và 25 FN thành TP.
- Phát sinh **81** quyết định sai mới: 24 TN thành FP và 57 TP thành FN.
- Có **637** quyết định vẫn sai. Tổng lỗi chọn/bỏ: **717 → 718**.
- Q trung bình tăng 0.49667 → 0.50717; R giảm 0.81883 → 0.79822. Metric còn phụ thuộc mức nhãn và thứ tự, nên tổng lỗi không giảm mà điểm vẫn có thể tăng nhẹ.

| old_status | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| TP | 1245 | 0 | 0 | 57 |
| TN | 0 | 451 | 24 | 0 |
| FP | 0 | 55 | 373 | 0 |
| FN | 25 | 0 | 0 | 264 |

## Lỗi dữ liệu được sửa chỉ chiếm phần nhỏ

Chỉ 16/3.090 input thay đổi, gồm **15/2.494 mẫu train (~0.60%)**. Mô hình vẫn dùng encoder SPECTER2 đóng băng và các mô hình tuyến tính; lượt này không fine-tune encoder theo sở thích người dùng. Các mẫu được sửa ảnh hưởng cả model được fit và những dự đoán khác, nhưng không cung cấp một lượng lớn tín hiệu sở thích mới.

Phép ablation giữ cùng Ridge alpha=0.1, splits và cách tune ngưỡng: 0.61777 → 0.61958; FP 428 → 413, FN 289 → 294. Không quy chênh lệch toàn bộ v2 cho riêng việc làm sạch.

ID 1674 được chọn lại đúng; ID 1883 tăng expected rating rõ rệt nhưng vẫn bị bỏ. Đây là bằng chứng sửa input có ích ở một số ca nhưng không đủ giải quyết quyết định cuối. Các thay đổi dự đoán này gồm cả ảnh hưởng của thay model/ngưỡng, không phải hiệu ứng nhân quả riêng của từng sửa abstract. Chi tiết: `outputs/residual_repaired_inputs.csv`.

## Các lỗi còn lại

### 1. Chọn sai 397 bài nhãn 1, có bài đứng rất cao

Rank dưới đây tính **riêng trong outer fold** (khoảng 500 ứng viên/fold), không ghép các fold thành một ranking. Expected rating là 1+4*predicted gain, không phải nhãn đã quan sát của model.

| id | title | venue | Label | outer_fold | v2_status | old_rating | v2_rating | v2_rank_in_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 217 | Epistemic Logic Programs: a Novel Perspective and Some Extensions. | iclp | 1 | 1 | FP | 3.855 | 4.324 | 2 |
| 622 | Explaining Decisions in ML Models: A Parameterized Complexity Analysis. | kr | 1 | 2 | FP | 3.772 | 3.718 | 11 |
| 372 | Advances in Big Data Bio Analytics. | iclp | 1 | 2 | FP | 3.844 | 3.623 | 17 |
| 184 | ABA Learning via ASP. | iclp | 1 | 3 | FP | 3.552 | 3.633 | 19 |
| 320 | Towards Structure Learning under the Credal Semantics. | iclp | 1 | 0 | FP | 3.628 | 3.611 | 20 |
| 144 | Commonsense Reasoning-Aided Autonomous Vehicle Systems. | iclp | 1 | 4 | FP | 3.737 | 3.586 | 20 |

Ví dụ ID 217 nhãn 1 nhưng đứng thứ 2 trong fold. Việc chỉ dịch ngưỡng chọn thông thường không giải quyết được một bài đã bị chấm cao như vậy. Cần cải thiện tín hiệu hoặc cách học sở thích, thay vì chỉ tăng coverage abstract.

### 2. Bỏ sót 321 bài, gồm 40 bài nhãn 4–5

| Label | n | selected | FN | mean_rating | std_rating |
| --- | --- | --- | --- | --- | --- |
| 1 | 903 | 397 | 0 | 2.056 | 0.55 |
| 2 | 514 | 339 | 175 | 2.379 | 0.542 |
| 3 | 438 | 332 | 106 | 2.525 | 0.547 |
| 4 | 367 | 335 | 32 | 2.829 | 0.547 |
| 5 | 272 | 264 | 8 | 3.202 | 0.528 |

Nhãn 1 có mean prediction khoảng 2.06, nhãn 5 khoảng 3.20. Dự đoán giữa các mức vẫn chồng lấn và co về giữa; đây là triệu chứng quan sát được của mô hình hiện tại, không chứng minh nhãn gốc sai.

| id | title | venue | Label | outer_fold | v2_status | old_rating | v2_rating | v2_rank_in_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1045 | Artificial Intelligence in Medicine: Knowledge Representation and Transparent and Explainable Systems - AIME 2019 International Workshops, KR4HC/ProHealth and TEAAM, Poznan, Poland, June 26-29, 2019, Revised Selected Papers | kr | 5 | 1 | FN | 1.82 | 1.744 | <NA> |
| 519 | Non-deterministic Action Reversibility: Complexity Results. | kr | 5 | 2 | FN | 1.739 | 1.864 | <NA> |
| 682 | How to Manage a Budget with ATL+. | kr | 5 | 3 | FN | 1.924 | 1.927 | <NA> |
| 2369 | Separating Markov's Principles. | lics | 5 | 0 | FN | 1.855 | 1.929 | <NA> |
| 707 | Knowledge Compilation and More with SharpSAT-TD. | kr | 5 | 1 | FN | 2.031 | 1.963 | <NA> |
| 2383 | Soundness of reset workflow nets. | lics | 5 | 0 | FN | 2.129 | 1.995 | <NA> |
| 704 | Practical Abstraction for Model Checking of Multi-Agent Systems. | kr | 5 | 0 | FN | 2.007 | 2.063 | <NA> |
| 664 | On Conflict-free Labellings - Realizability, Construction and Patterns of Redundancy. | kr | 5 | 1 | FN | 2.068 | 2.089 | <NA> |
| 3062 | Upper Bounds on the Quantifier Depth for Graph Differentiation in First Order Logic. | lics | 4 | 4 | FN | 1.329 | 1.482 | <NA> |
| 2516 | Reasonable Space for the λ-Calculus, Logarithmically. | lics | 4 | 2 | FN | 1.274 | 1.502 | <NA> |
| 1344 | QMaxSATpb: A Certified MaxSAT Solver. | lpnmr | 4 | 1 | FN | 1.718 | 1.503 | <NA> |
| 2995 | Uniform, integral and efficient proofs for the determinant identities. | lics | 4 | 1 | FN | 1.533 | 1.531 | <NA> |

### 3. Bỏ sót tập trung ở CAV/LICS; ICLP/KR còn chọn nhiều bài nhãn 1

| venue | n | positives | FP | FN | FPR_percent | FNR_percent |
| --- | --- | --- | --- | --- | --- | --- |
| cav | 657 | 357 | 100 | 124 | 33.3 | 34.7 |
| iclp | 456 | 339 | 88 | 6 | 75.2 | 1.8 |
| kr | 661 | 484 | 117 | 35 | 66.1 | 7.2 |
| lics | 604 | 316 | 79 | 153 | 27.4 | 48.4 |
| lpnmr | 116 | 95 | 13 | 3 | 61.9 | 3.2 |

**277/321 FN (~86.3%)** ở CAV và LICS. FPR tính trên số nhãn 1 từng venue; FNR tính trên số nhãn >1 từng venue. Không suy ra cứ thuộc một venue là nên chọn hoặc bỏ: mỗi venue vẫn chứa cả hai loại nhãn. Có thể dùng thông tin này để thiết kế mô hình bổ sung đặc trưng/interaction rồi kiểm tra lại, không gán thủ công nhãn theo venue.

### 4. Thiếu abstract vẫn tồn tại nhưng không giải thích phần lớn lỗi

| abstract_group | n | positives | FP | FN |
| --- | --- | --- | --- | --- |
| title_only | 98 | 13 | 11 | 0 |
| usable_abstract | 2396 | 1578 | 386 | 321 |

`usable_abstract` nghĩa là vượt các quy tắc kiểm tra đang có, không phải tất cả đã được đọc đối chiếu thủ công. Còn 80 mục thiếu văn bản, 35 mục cần xác minh danh tính nguồn và 1 abstract chỉ dẫn sang công trình khác trên cả train+test; 9 input vẫn dài hơn 512 tokens. Có thể còn lỗi nguồn ngoài các ca đã xác nhận. Bảng trên kiểm tra trực tiếp các lỗi OOF, tránh quy tất cả lỗi cho coverage.

## Cần sửa gì tiếp theo?

Ưu tiên **học phân biệt sở thích ở các chủ đề gần nhau** và kiểm soát đánh đổi chọn/bỏ. V2 hai tầng đạt FP=424, FN=275 và điểm 0.62043: bỏ sót ít hơn quy trình đang chọn, nhưng FP cao hơn; chưa có họ nào giải quyết được cả hai ở mức lớn. Không thể chỉ hạ ngưỡng rồi coi là đã sửa vì sẽ đưa thêm bài nhãn 1 vào.

Các thí nghiệm kế tiếp có cơ sở: bổ sung author/venue/year cùng text dưới regularization; thử học ordinal hoặc ranking phù hợp các mức nhãn; so sánh encoder retrieval và cân nhắc fine-tuning sau khi có validation nghiêm ngặt. Đây là các giả thuyết cần đo, không phải bảo đảm tăng điểm. Giữ một tập đánh giá mới hoặc dùng đánh giá ngoài phù hợp vì tập OOF này đã được xem để thiết kế các thay đổi.

Chênh lệch inner CV của hai cấu hình đầu chỉ khoảng 0.00029; lựa chọn cấu hình còn không ổn định. Không gọi họ hybrid đạt 0.62242 là điểm của riêng model 50/50 cuối cùng, hoặc đổi sang họ thắng outer CV rồi báo cùng điểm như một đánh giá độc lập.

## Artifacts

- `outputs/residual_failure_rows.csv`: đủ 2.494 mục, trạng thái cũ/mới, transition, rating và rank trong fold.
- `outputs/residual_examples.csv`: các ví dụ sai nặng ở trên.
- `outputs/residual_transitions.csv`, `residual_by_venue.csv`, `residual_by_label.csv`, `residual_by_abstract_group.csv`.
- `outputs/residual_repaired_inputs.csv`: kết quả 15 input train đã sửa.
- `outputs/residual_summary.json`: các số đếm kiểm tra được.

Chạy lại: `python experiments/specter2_v2/analyze_residuals.py`.
