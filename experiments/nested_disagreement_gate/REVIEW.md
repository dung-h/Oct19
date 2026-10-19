# Đánh giá sau run: tín hiệu tốt hơn, quyết định cuối chưa tốt hơn

Đánh giá này dùng artefact outer OOF đã có, không train thêm và không sửa protocol đã khóa.
Các counterfactual dưới đây là chẩn đoán sau khi xem outer labels; không được dùng để
chọn submission hoặc tuyên bố một model mới đã qua validation.

## Kết quả thực tế

| Đại lượng | Incumbent | Nested meta gate |
|---|---:|---:|
| Mean fold H | 0.640276 | 0.639936 |
| Mean fold Q | 0.532500 | 0.557271 |
| Mean fold R | 0.803238 | 0.758268 |
| Mean fold read AUC | 0.799457 | 0.816374 |
| Mean fold Brier (thấp hơn tốt hơn) | 0.172009 | 0.166881 |
| Accepted | 1616 | 1478 |
| FP / FN | 338 / 313 | 274 / 387 |
| Missed Label4 / Label5 | 33 / 5 | 39 / 9 |

AUC tăng ở cả 5 fold; Brier giảm ở 4/5 fold. Điều này ủng hộ việc meta score có
khả năng phân biệt read/non-read tốt hơn trên các fold đã xét. AUC không đo thứ tự
ưu tiên giữa Label2–5 và không bảo đảm H tăng. Brier là sai số probability tổng
hợp, không phải bằng chứng riêng rằng calibration đã tốt ở mọi vùng probability.

H giảm 0.000340, quá nhỏ để kết luận model kém hơn có ý nghĩa thống kê từ 5 fold
này. Tuy nhiên candidate không đáp ứng điều kiện promotion đã định trước.

## Các thay đổi mẫu

- Loại đúng 87 FP, nhưng bỏ nhầm 120 TP đang được incumbent nhận.
- Cứu 46 FN, đồng thời thêm 23 FP mới.
- Ròng: FP giảm 64, TP giảm 74, errors tăng 10.
- 120 TP bị mất gồm Label2=59, Label3=41, Label4=15, Label5=5.
- 46 TP được cứu gồm Label2=15, Label3=21, Label4=9, Label5=1.

Vì vậy thiệt hại không chỉ đến từ các paper có priority thấp: có 6 Label4 và
4 Label5 bị mất ròng. Counts là chẩn đoán hữu ích, nhưng FP+FN không phải metric
assignment và không thay thế H/Q/R.

## Hai fold gây mất lợi ích

| Outer fold | Threshold chọn bằng inner H | Delta H | Delta FP | Delta FN |
|---|---:|---:|---:|---:|
| 0 | 0.55 | +0.003545 | -2 | 0 |
| 1 | 0.60 | -0.011866 | -21 | +34 |
| 2 | 0.55 | +0.010940 | -5 | -4 |
| 3 | 0.60 | -0.006772 | -28 | +41 |
| 4 | 0.55 | +0.002456 | -8 | +3 |

Cả hai fold chọn 0.60 đều giảm H. Ở fold1, mean inner H của 0.60 chỉ hơn
0.55 khoảng 0.000950 nhưng outer recall giảm mạnh. Chưa đủ bằng chứng quy hết
failure cho threshold: đây cũng có thể là shift của source scores, calibration,
hoặc sai số chọn policy từ inner data.

## Chẩn đoán cùng ngân sách nhận

Giữ nguyên meta probability và incumbent ranking, nhưng trong mỗi outer fold
chọn top meta scores với đúng số paper mà incumbent nhận trong fold đó:

| Chẩn đoán | Mean H | FP | FN | Missed L4 | Missed L5 |
|---|---:|---:|---:|---:|---:|
| Incumbent | 0.640276 | 338 | 313 | 33 | 5 |
| Meta top scores, cùng accepted count | 0.644252 | 328 | 303 | 28 | 7 |

Đây là phép kiểm soát để tách chất lượng score khỏi thay đổi kích thước danh
sách. Nó cho thấy có lợi ích phân biệt nhỏ trong meta score; không phải một
policy triển khai được xác nhận, và vẫn làm missed Label5 tăng. Không lấy số
accepted quan sát ở outer để thiết kế policy cuối.

Vì vậy kết luận trước "phải có external feature mới cải thiện được" là quá
mạnh. Dữ liệu hiện có chưa chứng minh điều đó, cũng chưa chứng minh có thể
vượt 0.7 bằng việc thay gate.

## Hạn chế thiết kế và verifier

1. Ba pairwise logit differences là tổ hợp tuyến tính chính xác của ba source
   logits. Với logistic tuyến tính, chúng không thêm thông tin hay biểu diễn
   tương tác phi tuyến; chúng có thể thay đổi regularization hiệu dụng.
2. Target Label>1 đánh trọng số Label2 và Label5 như nhau trong binary loss.
   Assignment đồng thời cần gain theo Label và recall. Recipe tách gate/ranking
   là hợp lý để làm ablation, nhưng binary AUC tốt hơn không bảo đảm policy H
   tốt hơn hoặc giữ được high-priority papers.
3. Cross-fit meta trên một bảng source OOF chung chưa phải nesting hoàn toàn
   cho inner threshold evaluation. Khi hold out inner fold k ở meta level,
   source scores của các meta-training rows thuộc fold j có thể được tạo bởi
   source model đã học labels của k. Đây là đường phụ thuộc gián tiếp vào
   inner-held-out labels. Outer validation vẫn nằm ngoài outer train của lượt
   fit tương ứng; hạn chế này không đồng nghĩa có leakage trực tiếp outer.
4. Verifier thực sự replay được saved models/features, nhưng mục label
   perturbation hiện là tautology: nó sửa một bản sao labels rồi so features
   với chính features, không rebuild inference dưới context đã bị perturb.
   Không nên diễn giải `label_perturbation_invariant=true` là kiểm định chống
   leakage đã hoàn tất.
5. Metric replay tái dùng cùng hàm `n.metrics`, nên xác nhận nhất quán output,
   chưa phải một implementation metric độc lập. Các source/recipe cũng đã
   được xem xét trên những outer folds này ở các thử nghiệm trước; chúng
   không còn là một holdout hoàn toàn mới cho toàn bộ quá trình nghiên cứu.

Không sửa source đã hash để tránh làm sai provenance của run hiện tại. Các
hạn chế này được ghi bổ sung ở đây thay vì ghi đè báo cáo verifier cũ.

## Hướng ưu tiên

Giữ incumbent làm submission hiện tại. Giữ quyết định reject cho recipe này,
nhưng không suy rộng thành chứng minh mọi score ensemble đều vô ích.

Nếu thực hiện lượt tiếp theo, ưu tiên một thí nghiệm nhỏ về policy selection:
hold out calibration groups trước khi dựng toàn bộ source/meta training,
so threshold selection theo inner H với policy có ràng buộc recall được chốt
trước trên calibration data. Không chọn threshold từ bảng outer hậu nghiệm.
Sửa verifier perturbation và metric reference trước khi train lại. Đánh giá
high-label losses, Q/R, calibration và count transitions, đồng thời dùng một
protocol đánh giá mới đã khóa để hạn chế chọn model theo các fold cũ.

Nếu score tốt hơn mà policy vẫn không đạt, khi đó mới đầu tư vào audit nguồn
feature mới cho lỗi consensus. Nguồn graph/topic/author có thể giúp, nhưng
chưa có bằng chứng coverage hoặc predictive value để ưu tiên một đợt collect
lớn ngay lúc này.
