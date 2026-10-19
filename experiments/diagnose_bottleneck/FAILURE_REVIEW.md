# Failure review và hướng thí nghiệm tiếp theo

## Kết luận cần dùng để quyết định

**Nút thắt quan sát được là phân biệt bài đáng đọc với bài nhãn 1, đặc biệt các bài trông rất gần chủ đề được quan tâm.** Chỉnh ngưỡng toàn cục chỉ có dư địa nhỏ với điểm dự đoán hiện tại. Chưa xác định được một nguyên nhân duy nhất ở encoder, loss hoặc dữ liệu; việc nhiều model cùng sai không chứng minh nhãn sai hay bài toán không thể học.

Hướng ưu tiên: **kiểm tra thông tin đầu vào và metadata → tập trung học phân biệt các ví dụ khó → fine-tune có đối chứng nếu các phép thử đầu vào chưa giải quyết được**. Không đủ bằng chứng để coi fine-tune là bước chắc chắn phải làm ngay, hoặc hứa nó sẽ cải thiện lớn.

Báo cáo này dùng một baseline nhất quán: `regression_ridge` trong thí nghiệm chẩn đoán, H=0.61949 trên 2.494 outer OOF, FP=414, FN=293. Không trộn với v2 `selection_procedure` có FP=397/FN=321 hay holdout 500 mục cũ. Các nhãn và input dùng huấn luyện hiện có được giữ nguyên; file audit mới chỉ ghi phát hiện.

## 1. Bóc tách lỗi bằng các giả lập biết nhãn thật

Các số dưới đây là trung bình 5 folds. Những dòng oracle **dùng nhãn validation thật chỉ để chẩn đoán**, không phải dự đoán triển khai được, không phải mức tăng kỳ vọng và không được dùng để chọn model/ngưỡng submission.

| Giả lập | H | Chênh lệch so với hiện tại |
|---|---:|---:|
| Quy trình hiện tại | 0.61949 | — |
| Chọn ngưỡng tốt nhất bằng nhãn thật, giữ nguyên thứ tự dự đoán | 0.62514 | +0.00565 |
| Xếp hạng hoàn hảo bằng nhãn thật, giữ nguyên tập bài đang chọn | 0.64883 | +0.02934 |
| Loại đúng các bài nhãn 1 đang chọn, giữ thứ tự các bài còn lại | 0.71218 | +0.09268 |
| Chọn toàn bộ bài nhãn >1, xếp theo model hiện tại | 0.74893 | +0.12944 |
| Chọn toàn bộ bài nhãn >1, xếp theo nhãn thật | 0.77324 | +0.15375 |

Hàm ý: sửa bộ phân biệt đọc/không đọc có tiềm năng tác động lớn hơn việc chỉ sắp xếp lại danh sách hiện có. Đổi một ngưỡng toàn cục không đủ vì nhiều nhãn 1 đã nằm cao và nhiều bài đáng đọc nằm thấp. Chỉ 114/414 FP và 128/293 FN nằm trong khoảng 0.2 điểm dự đoán quanh ngưỡng; phần lớn lỗi không phải chỉ sát ranh giới. Khoảng 0.2 là mô tả, không phải khoảng tin cậy.

Oracle ngưỡng chỉ giới hạn các **prefix của đúng thứ tự này**, không giới hạn model tương lai hoặc bộ chọn khác. Các mức tăng không cộng được. Dòng chọn toàn bộ dương/xếp đúng cũng không được coi là tối ưu toàn cục của H: metric có thể ưu tiên bỏ một số bài gain thấp.

## 2. Các nhóm failure cụ thể

### A. Bài bị cả tám họ văn bản cùng chấm sai

Có 280 mục sai quyết định chọn/bỏ ở cả bốn biểu diễn × hai head, gồm **222 FP và 58 FN**. 58 FN gồm 35 nhãn 2, 19 nhãn 3 và 4 nhãn 4; không có nhãn 5. Điều này không có nghĩa nhãn 5 đã được giải quyết: baseline vẫn bỏ 7 bài nhãn 5, nhưng ít nhất một họ khác cứu được mỗi bài đó.

| Venue | FP chung | FN chung |
|---|---:|---:|
| ICLP | 63 | 0 |
| KR | 84 | 0 |
| LPNMR | 12 | 0 |
| CAV | 41 | 23 |
| LICS | 22 | 35 |

Toàn bộ 58 FN chung thuộc CAV/LICS. Nhóm này đáng điều tra riêng về chủ đề và cách diễn đạt, nhưng không được gán nhãn hay loại bài theo venue.

Ví dụ OOF baseline:

| ID | Bài | Nhãn thật | Rating dự đoán | Quan sát |
|---|---|---:|---:|---|
| 217 | Epistemic Logic Programs: a Novel Perspective and Some Extensions | 1 | 3.865 | Cả 8 họ văn bản đều chọn nhầm |
| 184 | ABA Learning via ASP | 1 | 3.548 | Cả 8 họ văn bản đều chọn nhầm |
| 519 | Non-deterministic Action Reversibility: Complexity Results | 5 | 1.762 | Baseline bỏ; thêm metadata có thể cứu ca này nhưng tổng score của cách ghép đó thấp hơn |
| 1344 | QMaxSATpb: A Certified MaxSAT Solver | 4 | 1.719 | Baseline bỏ; metadata cứu được ca này trong OOF |
| 2321 | Reachability Types, Traces and Full Abstraction | 4 | 1.596 | Cả 8 họ văn bản đều bỏ |

Một cách giải thích phù hợp là model nhận ra quan hệ chủ đề nhưng chưa học đủ tiêu chí phân biệt sở thích bên trong chủ đề. Đây là **giả thuyết**, không phải lời giải thích đã biết về lý do nhà nghiên cứu gán từng nhãn.

### B. Hành vi khác nhau giữa các venue

| Venue | FP / số bài nhãn 1 | FN / số bài nhãn >1 | AUC chọn/bỏ trong venue, TB 5 folds |
|---|---:|---:|---:|
| ICLP | 85/117 = 72.6% | 7/339 = 2.1% | 0.754 |
| KR | 120/177 = 67.8% | 28/484 = 5.8% | 0.750 |
| LPNMR | 15/21 = 71.4% | 3/95 = 3.2% | 0.785 |
| CAV | 114/300 = 38.0% | 108/357 = 30.3% | 0.734 |
| LICS | 80/288 = 27.8% | 147/316 = 46.5% | 0.669 |

Baseline bỏ sót **255/293 bài ở CAV/LICS**. LICS còn yếu cả ở khả năng phân biệt trong cùng venue (AUC 0.669), nên không thể coi vấn đề chỉ là ngưỡng chung. ICLP/KR có AUC khá hơn nhưng nhiều bài nhãn 1 vượt ngưỡng. AUC và tỷ lệ lỗi đo các khía cạnh khác nhau; cần xem cả hai.

### C. Coverage abstract không phải lời giải thích chính

98 mẫu train dùng title-only có 11 FP, 0 FN. 2.396 mẫu có abstract được dùng có 403 FP và toàn bộ 293 FN. Có abstract không bảo đảm thông tin hữu ích hoặc đúng hoàn toàn, nhưng điều này bác bỏ cách giải thích đơn giản rằng phần lớn FN xuất hiện vì chưa collect abstract.

Đáng chú ý hơn: **các thí nghiệm vừa rồi chưa có ablation title-only trên toàn train**. Chúng đều dùng title + abstract cho phần lớn mẫu. Vì thế, chưa biết thêm abstract giúp được bao nhiêu, có làm loãng dấu hiệu trong title không, hoặc title/abstract nên có trọng số riêng hay không.

## 3. Phát hiện mới về metadata: author sai ở hai ca đã đối chiếu

- **ID 217:** dataset ghi `SaÚl FernÁndez GonzÁlez`; [PDF chính thức CEUR](https://ceur-ws.org/Vol-3193/paper1EELP.pdf) ghi **Stefania Costantini và Andrea Formisano**. Title và abstract thu thập khớp bài chính thức.
- **ID 372:** dataset ghi `Yong Shi`; [bản tác giả trên arXiv](https://arxiv.org/abs/1909.08254) và [bài EPTCS](https://cgi.cse.unsw.edu.au/~eptcs/paper.cgi?ICLP2019.36.pdf=) ghi **Nicos Angelopoulos và Jan Wielemaker**. Title và abstract thu thập cũng khớp nguồn.

Đây là xác nhận lỗi tác giả, **không phải xác nhận abstract sai hoặc Label sai**. Hai trường hợp được chọn để soi lỗi nên không dùng tỷ lệ 2/2 để ước lượng toàn dataset. Cần lưu `authors_original`, `authors_verified`, nguồn và trạng thái xác minh; đánh giá feature đã xác minh trước khi kết luận author không hữu ích. Việc tự sửa nhãn 1 thành nhãn cao vì “paper có vẻ liên quan” sẽ làm mất ground truth.

Ngay cả trước vấn đề này, author còn thưa: 192 mẫu OOF không có author, 664 mẫu chỉ có author chưa gặp trong outer train. Cách multi-hot exact name hiện tại không khai thác được 856/2.494 mẫu đó theo danh tính tác giả.

## 4. Các kết luận cũ cần thu hẹp

1. **17 họ/138 cấu hình không phải 138 giả thuyết độc lập.** Phần lớn chia sẻ cùng text, frozen representations và hồi quy pointwise. Tám họ text có mean pairwise error Jaccard 0.557; lỗi tương quan mạnh. Model cùng sai không đủ để kết luận dữ liệu không có tín hiệu.
2. Đổi sang retrieval/MiniLM không tốt hơn chỉ cho thấy các encoder thay thế **trong các cấu hình đã thử** không có lợi. Chưa kiểm tra khả năng encoder học theo nhãn sở thích.
3. RBF làm train MAE giảm 0.929→0.825, validation MAE chỉ giảm 0.973→0.961, H tăng 0.00057. Chỉ tăng độ linh hoạt của head chưa giúp nhiều; không suy ra neural fine-tuning chắc chắn có lợi hoặc chắc chắn vô ích.
4. Cấu trúc hai tầng đã được thử ở v2, đạt khoảng 0.62043. Vì thế “tách classification và regression” tự nó không phải ý tưởng mới có bằng chứng sẽ tạo bước nhảy. Thay đổi có ý nghĩa phải nằm ở input, representation được học, hoặc training signal cụ thể.
5. Theo [data.md](../../data.md), nhãn phản ánh sở thích **một nhà nghiên cứu**, không phải chất lượng khoa học chung. Tài liệu chưa nói rõ quy trình đánh giá hoặc người đó được xem những trường nào. Không suy luận nhãn do AI sinh ra hoặc bị nhiễu hàng loạt từ các lỗi dự đoán.

## 5. Hướng đi ưu tiên, với câu hỏi và tiêu chí quyết định

### Bước 1 — Kiểm tra đúng thông tin đầu vào trước khi tăng độ phức tạp

Chạy một ma trận nhỏ, giữ nguyên splits/head/search budget:

| Input | TF-IDF + Ridge | SPECTER2 hiện tại + Ridge |
|---|---|---|
| Chỉ title | Đối chứng mới | Đối chứng mới |
| Title + abstract hiện tại | Đối chứng đã có | Đối chứng đã có |
| Title và abstract thành hai khối riêng, trọng số tune trong inner CV | Thử trọng số title/abstract | Thử trọng số title/abstract |

Hai khối riêng tăng số feature, nên regularization cần chọn trong inner CV; không quy mọi thay đổi cho nội dung nếu capacity cũng đổi. Đánh giá abstract trên các record có nguồn đủ tin cậy; ghi coverage và so sánh cùng ID. Audit nguồn nên có cả mẫu đúng và mẫu sai, với quy tắc kiểm tra áp dụng cho tất cả, tránh chỉ tìm lỗi ở bài model dự đoán sai.

Nếu title-only tốt ngang hoặc hơn, ưu tiên title và chỉ thêm abstract khi mang lại lợi ích đo được. Nếu abstract giúp một số venue/loại bài, thử kết hợp có regularization thay vì một heuristic gán theo nhãn đã xem. Đồng thời thử metadata đã xác minh trên cùng cách đánh giá, giữ nguyên nhãn gốc.

### Bước 2 — Thiết kế bài học xoay quanh các nhầm lẫn đọc/không đọc

Mục tiêu chính là phân biệt **Label=1 với Label>1**; nhánh thứ hai học thứ tự ưu tiên trong các bài đáng đọc. Không tối ưu binary accuracy đơn thuần: tăng ngưỡng có thể giảm FP bằng cách bỏ thêm rất nhiều bài tốt.

Chọn các cặp khó **chỉ trong training partition**: một bài nhãn 1 và một bài nhãn >1 gần nhau về chủ đề/venue. Dùng pointwise binary loss kết hợp một loss ưu tiên bài dương hơn bài âm; vẫn giữ mẫu ngẫu nhiên để không lệch hoàn toàn về nhóm khó. Không kéo mọi bài nhãn 1 về cùng một cụm ngữ nghĩa, vì chúng có thể thuộc những chủ đề rất khác nhau.

Đối chứng tối thiểu:

- Frozen encoder + head với cùng mục tiêu binary/ranking.
- Encoder chỉ train adapter + chính head/loss đó.
- Chỉ sau đó so sánh sampling ngẫu nhiên với sampling các cặp khó trong train.

Như vậy mới tách được tác động của objective, cập nhật representation và hard-example sampling. Nhóm 280 lỗi đã xem chỉ dùng để báo cáo; **không đem nguyên danh sách lỗi outer OOF làm trọng số huấn luyện toàn bộ CV**, vì sẽ dùng thông tin validation trong quá trình fit.

SPECTER2 có adapter theo loại tác vụ nhưng chúng được huấn luyện trên benchmark khoa học, chưa phải nhãn sở thích ở assignment này ([model card](https://huggingface.co/allenai/specter2)). Các phương pháp như [SetFit](https://huggingface.co/docs/setfit/en/index) là một tham khảo về fine-tuning representation với ít nhãn; không phải bằng chứng sẽ thắng trên dataset này.

### Bước 3 — Chọn mô hình bằng hiệu quả thực tế, rồi mới chỉnh xác suất/ngưỡng

Theo dõi cùng lúc: H, FP/FN, FPR tại recall tương đương baseline (~0.816), AUC đọc/không đọc, số bài nhãn 4–5 bị bỏ, và lỗi theo CAV/LICS. Báo cáo từng fold và một số seed cố định cho các mô hình neural. Chỉ giảm training loss không đủ.

Khi classifier đã có khả năng phân biệt tốt hơn, mới kiểm tra calibration và chọn threshold từ inner OOF. Calibration giúp diễn giải xác suất nhưng không tự tạo thêm thông tin phân biệt ([tài liệu scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)); oracle threshold hiện tại cho thấy việc chỉnh một cutoff trên cùng ranking không phải ưu tiên lớn nhất.

Các fold này đã được xem nhiều lần. Cần chốt protocol trước vòng tiếp theo và dùng đánh giá ngoài phù hợp nếu có; đổi seed trên cùng 2.494 nhãn là kiểm tra độ nhạy, không tạo một test set hoàn toàn mới. Không chọn mô hình chỉ vì nó thắng trên vài ca nổi bật đã biết nhãn.

## Artifacts và tái tạo

- [summary.json](outputs/failure_mechanisms/summary.json): thống kê và cảnh báo diễn giải oracle.
- [oracle_diagnostics_by_fold.csv](outputs/failure_mechanisms/oracle_diagnostics_by_fold.csv): từng giả lập ở từng fold.
- [all_rows.csv](outputs/failure_mechanisms/all_rows.csv): trạng thái lỗi, số họ cùng sai, khoảng cách tới ngưỡng và rank trong fold.
- [common_280_errors.csv](outputs/failure_mechanisms/common_280_errors.csv): nhóm lỗi chung.
- [by_venue.csv](outputs/failure_mechanisms/by_venue.csv), [by_label.csv](outputs/failure_mechanisms/by_label.csv).
- [verified_metadata_discrepancies.csv](outputs/failure_mechanisms/verified_metadata_discrepancies.csv): hai sai khác author có nguồn xác nhận; không sửa dữ liệu train.

Chạy lại phân tích định lượng: `python experiments/diagnose_bottleneck/analyze_failure_mechanisms.py`.

Đây là phân tích trên prediction đã lưu; chưa thực hiện các thí nghiệm input/fine-tuning được đề xuất ở bước 1–2. Các phép kiểm tra trong script xác nhận số hàng/nhãn qua merge, FP/FN, nhóm lỗi chung và oracle không thấp hơn kết quả thực trong các trường hợp phải đơn điệu.
