# Phân tích failure của SPECTER2 + Ridge

Phân tích này dùng 500 dòng holdout và các dự đoán đã lưu của baseline. Không huấn luyện lại, đổi ngưỡng, thay abstract hoặc sửa submission trong lượt phân tích này. ID trong các ví dụ là ID gốc của tập train, không phải nhãn test được suy đoán bằng tay.

**Điểm yếu lớn nhất hiện tại là chọn nhầm bài nhãn 1. Bỏ sót bài đáng đọc tập trung ở CAV/LICS. Ngoài model, đã xác nhận thêm vấn đề chất lượng abstract trong dữ liệu đầu vào.**

Các sửa chữa và thử nghiệm tiếp theo đã được thực hiện trong [bản v2](../specter2_v2/README.md). Báo cáo này giữ số liệu baseline tại thời điểm chẩn đoán; v2 có dataset và đánh giá nested CV riêng.

![Failure diagnostics](outputs/failure_analysis/failure_diagnostics.png)

## 1. Chọn bài: recall khá, nhận diện bài không muốn đọc còn yếu

| Nhãn thật | Bị loại | Được đề xuất | Tổng |
|---|---:|---:|---:|
| 1: không muốn đọc | 90 | **95 chọn nhầm** | 185 |
| 2–5: đáng đọc | **45 bỏ sót** | 270 | 315 |

Model chọn 365/500 mục, trong khi có 315 mục nhãn 2–5. Precision là 73.97%, recall 85.71%. Có 95/185 = **51.35% bài nhãn 1 vẫn được đề xuất**. Đây là lỗi về sở thích đọc của người được mô hình hóa, không phải kết luận những bài đó kém chất lượng khoa học.

Trong 45 mục bị loại nhầm: 27 nhãn 2, 14 nhãn 3, 3 nhãn 4, và 1 nhãn 5. Model giữ được 62/65 mục nhãn 4 và 53/54 mục nhãn 5; vấn đề bỏ sót chủ yếu nằm ở mức ưu tiên vừa/thấp.

22/45 mục bỏ sót có dự đoán trong khoảng (1.8, 2.0], gần ngưỡng 2.0. Nhưng 11 bài nhãn 1 lại được dự đoán trên 3.0, nên đổi một ngưỡng chung không giải quyết được toàn bộ: có cả sai thứ tự giữa hai nhóm. Ngưỡng hiện tại được chọn từ development CV, không phải holdout này.

## 2. Điểm dự đoán bị kéo về giữa, các nhóm chồng lấn

| Label thật | Số mục | Dự đoán trung bình | Độ lệch chuẩn | Được chọn |
|---|---:|---:|---:|---:|
| 1 | 185 | 2.09 | 0.56 | 95 |
| 2 | 94 | 2.36 | 0.55 | 67 |
| 3 | 102 | 2.58 | 0.55 | 88 |
| 4 | 65 | 2.84 | 0.53 | 62 |
| 5 | 54 | 3.11 | 0.60 | 53 |

Dự đoán trung bình tăng đúng chiều, nhưng khoảng cách giữa hai nhãn kề nhau chỉ khoảng 0.2–0.3, nhỏ hơn độ phân tán trong từng nhóm. Có bài nhãn 1 được 3.91, trong khi một bài nhãn 5 chỉ được 1.84.

Đây phù hợp với hiện tượng hồi quy về trung bình ở một model hồi quy có regularization và tín hiệu sở thích chưa đủ rõ. Nó chưa chứng minh chỉ cần ép dự đoán trải đều từ 1 đến 5: một phép biến đổi đơn điệu sẽ giữ nguyên thứ tự, nên không sửa được các cặp xếp ngược. Bản thân dự đoán rating trung bình cũng có thể là hợp lý khi model không chắc chắn; điều cần cải thiện là khả năng phân biệt sở thích và thứ tự.

Ridge học mục tiêu bình phương sai số rating; metric cuối cùng lại quan tâm chọn bài, gain và thứ tự. Đây là một khoảng cách giữa mục tiêu học và mục tiêu đánh giá. Trong thử nghiệm trước, SVR có MAE tốt hơn Ridge nhưng điểm recommendation thấp hơn, nên MAE không đủ để chọn giải pháp.

## 3. Các failure cụ thể

| ID | Bài | Label thật | Dự đoán | Kết quả |
|---|---|---:|---:|---|
| 184 | ABA Learning via ASP | 1 | 3.91 | Xếp thứ **9**, chọn nhầm |
| 108 | From Logic Programming to Programming in Logica… | 1 | 3.82 | Xếp thứ **13**, chọn nhầm |
| 217 | Epistemic Logic Programs: a Novel Perspective and Some Extensions | 1 | 3.77 | Xếp thứ **16**, chọn nhầm |
| 519 | Non-deterministic Action Reversibility: Complexity Results | 5 | 1.84 | Bị loại |
| 3062 | Upper Bounds on the Quantifier Depth for Graph Differentiation in First Order Logic | 4 | 1.43 | Bị loại |
| 1344 | QMaxSATpb: A Certified MaxSAT Solver | 4 | 1.75 | Bị loại; TF-IDF chọn được |
| 2383 | Soundness of reset workflow nets | 5 | 2.05 | Được chọn nhưng thứ **354/365** |
| 682 | How to Manage a Budget with ATL+ | 5 | 2.02 | Được chọn nhưng thứ **360/365** |

Top 20 có 3 bài nhãn 1; top 50 có 5 bài nhãn 1. Đồng thời có các bài nhãn 5 xuống gần cuối danh sách, vì vậy phần ranking cũng còn lỗi dù recall nhãn 5 cao.

Đối chiếu bài trong development gợi ý model đang học được vùng chủ đề rộng, nhưng chưa phân biệt tốt sở thích ở mức từng bài. Ví dụ, ID 184 gần các bài về học heuristic cho ASP (nhãn 5), symbolic reinforcement learning (4), và counterfactual explanations bằng ASP (3). Với ID 217, vùng lân cận có cả bài nhãn 2 lẫn 5 về epistemic logic. Chủ đề gần nhau chưa quyết định mức muốn đọc.

Đây là **diễn giải từ láng giềng**, không phải bằng chứng nhân quả về quyết định của Ridge; Ridge không trực tiếp lấy trung bình các láng giềng đó. Láng giềng chỉ được lấy từ development, bằng cosine sau khi trừ vector trung bình development để giảm thành phần chung. Cosine thô của embedding regression trong dữ liệu này thường rất cao ngay giữa các bài khác nhau, nên không dùng ngưỡng 0.99 để kết luận hai bài trùng nội dung. Chi tiết ở [development_neighbors.csv](outputs/failure_analysis/development_neighbors.csv).

## 4. Lỗi tập trung theo venue

| Venue | Mục đáng đọc | Bỏ sót | Recall | Chọn nhầm nhãn 1 / tổng nhãn 1 |
|---|---:|---:|---:|---:|
| CAV | 83 | 21 | 74.7% | 25/54 |
| ICLP | 61 | 1 | 98.4% | 20/28 |
| KR | 94 | 1 | 98.9% | 25/38 |
| LICS | 58 | 21 | **63.8%** | 23/62 |
| LPNMR | 19 | 1 | 94.7% | 2/3 |

**42/45 mục bị bỏ sót thuộc CAV hoặc LICS.** Ngược lại, ở ICLP và KR model giữ gần hết bài đáng đọc nhưng cũng chọn nhiều bài nhãn 1. LPNMR chỉ có 22 mục holdout và 3 mục nhãn 1 nên tỷ lệ ở nhóm này rất thiếu ổn định.

Một giả thuyết phù hợp là model khái quát từ các vùng chủ đề có prior sở thích cao/thấp, rồi bỏ lỡ ngoại lệ cá nhân. SPECTER2-only không nhận trường venue trực tiếp, nên đây là sự liên hệ giữa lỗi và venue/chủ đề, không chứng minh model dùng venue làm một rule.

Biến thể thêm venue/year/authors làm FP tăng từ 95 lên 116 và FN từ 45 lên 54; riêng LICS, số bỏ sót tăng từ 21 lên 32. Tuy nhiên cả cách biểu diễn, trọng số feature block và regularization đều tham gia vào biến thể này. Chưa thể quy kết một mình authors hay venue là nguyên nhân, hoặc kết luận metadata luôn vô ích.

## 5. Có lỗi dữ liệu đầu vào thật, không chỉ lỗi model

### Coverage abstract chưa phải chất lượng abstract

Trong holdout có 479 mục dùng abstract, 15 mục thiếu và 6 mục bị giữ lại để review. Cả 45 false negative đều nằm trong nhóm có abstract. Nhóm thiếu/review chỉ có hai mục đáng đọc, nên không thể dùng recall của nhóm này để kết luận title-only tốt ngang title+abstract. Điều có thể kết luận là việc tiếp tục tìm thêm các abstract đang thiếu chưa đánh trúng phần lớn false negative hiện tại.

Nhưng một abstract đã được collect vẫn có thể sai hoặc nhiễu. Đối chiếu failure cho thấy:

- **ID 446 ghép sai loại ấn phẩm:** input đang chứa abstract tiếng Pháp của luận án cùng tên, DOI `10.70675/6bc651c8ze78bz45dez9b34z8fe4450cb7ac`; [Crossref xác định record đó là dissertation](https://api.crossref.org/works/10.70675/6bc651c8ze78bz45dez9b34z8fe4450cb7ac). Record dataset là ICLP 2016; [publisher có abstract tiếng Anh của đúng bài, DOI 10.4230/OASIcs.ICLP.2016.16](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.ICLP.2016.16). Đây là lỗi ghép publication-version/type chưa được ghi trong 35 identity notes trước đó.
- **ID 175 trích quá phần Abstract:** [PDF chính thức](https://ceur-ws.org/Vol-3437/short4PLP.pdf) có abstract ngắn rồi đến đề mục `Bug Report`, nhưng văn bản đã collect kéo dài vào phần thân bài. ID này vẫn được chọn đúng ở mức nhị phân, nên có lỗi dữ liệu không đồng nghĩa mỗi lần đều gây quyết định sai.
- **13/3.090 abstract có mã dựng LaTeX** như `documentclass`, `usepackage`, `setlength` lặp lại; gồm 12 train và 1 test. Những đoạn này chưa được loại ở bước clean HTML/whitespace ban đầu.

Hai ví dụ false negative có tác động lên token budget rõ ràng:

| ID | Bài | Tokens trước | Tokens sau khi chỉ bỏ mã dựng LaTeX | Label / dự đoán |
|---|---|---:|---:|---|
| 1674 | Rounding Meets Approximate Model Counting | 1.655 | 394 | 3 / 1.97, bị loại |
| 1883 | Nonlinear Craig Interpolant Generation | 1.311 | 341 | 3 / 1.58, bị loại |

SPECTER2 chỉ nhận 512 tokens. Sau loại boilerplate, cả hai abstract nằm trong giới hạn mà vẫn giữ biểu thức toán. Đây là bằng chứng token bị lãng phí; **chưa có thí nghiệm encode lại để chứng minh mức tăng điểm**. Trong holdout tổng cộng 7 input bị truncate, có 5 bài đáng đọc và bỏ sót 3; mẫu này nhỏ, chỉ gợi ý một hướng kiểm tra.

Do đó số 3.010/3.090 được báo trước đây là coverage lấy được văn bản; nó chưa tương đương 3.010 abstract sạch đã được đối chiếu toàn bộ. Audit trước tập trung vào 80 mục thiếu và các mismatch được phát hiện, chưa kiểm định thủ công toàn bộ phần đã collect. Chi tiết lỗi nguồn: [verified_source_issues.csv](outputs/failure_analysis/verified_source_issues.csv).

### Nhãn không hoàn toàn nhất quán với cùng input

Có hai cặp train có cùng title+abstract đầu vào nhưng khác nhãn:

- ID 110 và 122: *Autonomous Task Completion Based on Goal-directed Answer Set Programming*, cùng DOI, nhãn 3 và 5.
- ID 273 và 1341: *Gradient-Based Supported Model Computation in Vector Spaces*, cùng Semantic Scholar ID, nhãn 4 và 3.

Model chỉ nhìn title+abstract sẽ nhận cùng biểu diễn cho hai input giống nhau, nên không thể tái hiện chính xác đồng thời cả hai nhãn. Đây là giới hạn dữ liệu/feature quan sát được; bốn record không đủ để giải thích toàn bộ 140 lỗi chọn/loại. Không tự chọn một nhãn là đúng hoặc sửa nhãn. Các nhóm duplicate đã được giữ cùng fold.

## 6. Phần lỗi nào có tiềm năng cải thiện lớn hơn?

Bảng sau dùng nhãn thật để phân rã lỗi. Đây là **oracle diagnostic**, không phải kết quả model khả thi hoặc điểm tăng đã đạt:

| Kịch bản giả định trên cùng holdout | Score |
|---|---:|
| Model hiện tại | 0.6193 |
| Giữ nguyên 365 bài đang chọn, sắp lại hoàn hảo theo nhãn thật | 0.6501 |
| Xóa đúng 95 bài nhãn 1 đang chọn, giữ thứ tự phần còn lại | **0.7260** |
| Chọn đúng toàn bộ 315 bài nhãn 2–5, xếp theo dự đoán hiện tại | 0.7512 |
| Chọn toàn bộ nhãn 2–5, sắp theo nhãn thật | 0.7734 |

Ở tập này, nhận diện đúng các mục không muốn đọc có tiềm năng lớn hơn chỉ rerank danh sách đang chọn. Không cộng các mức tăng oracle với nhau; chúng là các kịch bản khác nhau và dùng thông tin model chưa biết.

## 7. Các thí nghiệm tiếp theo nên kiểm chứng

1. **Làm sạch và xác minh abstract trước:** bỏ preamble LaTeX, sửa ranh giới section, yêu cầu match title+year+venue/publication type thay vì title đơn thuần. Sau đó encode lại một phiên bản input mới và so sánh bằng validation hợp lệ. Không sửa nhãn.
2. **Tách học quyết định chọn khỏi học thứ tự:** classifier `Label > 1` cho xác suất muốn đọc, kết hợp hồi quy/ordinal model cho mức ưu tiên. Chọn công thức priority và threshold trên development OOF theo metric của đề. Đây là giả thuyết phù hợp với FP cao, chưa bảo đảm cải thiện.
3. **Thử kết hợp tín hiệu lexical với SPECTER2:** TF-IDF chọn đúng 14 mục SPECTER2 bỏ sót, trong khi SPECTER2 chọn đúng 37 mục TF-IDF bỏ sót; TF-IDF cũng tránh 28 FP riêng của SPECTER2 nhưng mắc 27 FP riêng khác. Có dấu hiệu bổ sung nhau, cần kiểm tra hybrid/ensemble bằng CV chứ không chọn trọng số trên holdout này.
4. **Fine-tune encoder sau khi xử lý dữ liệu và loss:** frozen embeddings chưa thích nghi với sở thích người dùng. Chỉ khoảng 2.500 nhãn và đã có mismatch/nhãn mâu thuẫn, nên trước hết cần biết các bước đơn giản hơn cải thiện đến đâu.

Holdout này hiện đã được phân tích chi tiết. Các bước phát triển tiếp theo cần nested CV hoặc đánh giá mới; không dùng chính các case đã xem để dựng rule rồi coi điểm trên cùng holdout là bằng chứng độc lập.

## Chạy lại và files

```bash
python -m pip install -r experiments/specter2/requirements-analysis.txt
python experiments/specter2/analyze_failures.py
```

- [summary.json](outputs/failure_analysis/summary.json): thống kê tổng hợp.
- [holdout_error_rows.csv](outputs/failure_analysis/holdout_error_rows.csv): 500 dòng với dự đoán, rank, loại lỗi, metadata và excerpt abstract.
- [false_positives.csv](outputs/failure_analysis/false_positives.csv): 95 mục chọn nhầm, ưu tiên thứ tự đọc cao.
- [false_negatives.csv](outputs/failure_analysis/false_negatives.csv): 45 mục bỏ sót, ưu tiên nhãn cao.
- [high_label_low_rank.csv](outputs/failure_analysis/high_label_low_rank.csv): mục nhãn 4–5 được chọn, sắp theo rank từ cuối lên.
- `by_label.csv`, `by_venue.csv`, `by_year.csv`, `by_abstract_quality.csv`, `by_truncated.csv`: lát cắt lỗi.
- [latex_boilerplate_token_diagnostic.csv](outputs/failure_analysis/latex_boilerplate_token_diagnostic.csv): 13 trường hợp nhiễu LaTeX và token budget trước/sau.

Các phát hiện về model là mô tả và giả thuyết từ một tập holdout. Những vấn đề ghép nhầm ấn phẩm và trích quá section nêu trên đã được đối chiếu nguồn trực tiếp; ảnh hưởng định lượng của việc sửa dữ liệu chưa được đo.
