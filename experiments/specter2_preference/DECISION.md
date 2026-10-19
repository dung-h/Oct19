# Quyết định sau thí nghiệm SPECTER2 preference

**Chưa đạt0.7; giữ incumbent làm bản chính.** Đã chạy đủ đối chứng 5outer/4inner,
chọn epoch/ngưỡng bên trong training partitions, refit toàn train và xác minh model
reload. Model/CSV incumbent không bị ghi đè, chưa gửi Kaggle.

| Quy trình | Mean outer H | Q | R | FP | FN | Bỏ sót nhãn4/5 |
|---|---:|---:|---:|---:|---:|---:|
| SPECTER2 đóng băng + cùng neural head | 0.59794 | 0.50774 | 0.75333 | 361 | 397 | 81 |
| SPECTER2 adapter học theo nhãn | 0.63318 | 0.56622 | 0.72805 | 249 | 436 | 67 |
| CatBoost incumbent | 0.64028 | 0.53250 | 0.80324 | 338 | 313 | 38 |
| Primary policy chọn giữa các model bằng inner | 0.63277 | 0.56000 | 0.73871 | 269 | 419 | 60 |
| Phép trộn bổ sung, weight/ngưỡng chọn bằng inner | 0.64144 | 0.55937 | 0.75540 | 268 | 391 | 55 |

Fine-tune cải thiện cả5folds so với neural head đóng băng, nhưng chỉ3/5 so với
incumbent. Đây là lợi ích so với **đối chứng cùng head**, không chứng minh mọi
phương pháp dùng embedding đóng băng đều kém. AUC fine-tune tăng0.79946→0.80838
so với incumbent, đồng thời recall giảm0.80324→0.72805. Giảm89 FP kèm tăng123 FN;
chỉ giảm FP chưa đáp ứng mục tiêu H và coverage của assignment.

Phép trộn tăng0.00117 H, tốt hơn incumbent ở3/5folds, nhưng high_FN38→55.
Phép trộn được đề xuất sau khi xem một phần primary outer và outer dataset vốn
đã được nghiên cứu trước đây; mức tăng này chưa đủ cơ sở để thay bản chính hoặc
hứa cải thiện trên Private Kaggle. Không có phép kiểm định độc lập cho mức tăng.

### Follow-up: kéo dài title-only từ 8 lên 16 epochs

Thí nghiệm khóa tiếp theo cho cùng adapter/head/input không ủng hộ việc chạy lâu
hơn: epoch16 giảm H trung bình0.00148 và chỉ thắng2/5folds. Recall tăng0.05866,
FN giảm96 và bỏ sót nhãn4/5 giảm16, nhưng Q giảm0.03770; AUC cũng giảm từ0.80838
xuống0.78297. Inner selection chọn epoch8. Vì vậy dừng nhánh này, không mở rộng
sang title+abstract, và incumbent vẫn là bản chính. Đây là CV thăm dò trên các
fold đã được xem trước, không phải kết quả leaderboard độc lập. Chi tiết và
kiểm chứng ở [báo cáo 8-vs-16](../specter2_preference_longer/RESULTS.md) và
[quyết định nhánh đó](../specter2_preference_longer/DECISION.md).

Kết quả vẫn giới hạn ở adapter pretrained regression, title, LR5e-5 và lịch
1/2/4/8/16; chưa thử full transformer fine-tuning, cũng không có cơ sở ngoại suy
đạt0.7 chỉ bằng tăng epochs. [DIAGNOSIS.md](DIAGNOSIS.md) ghi rõ giới hạn và từng
lỗi.

## Artifact để đối chiếu

- [Kết quả primary](RESULTS.md), [phép trộn bổ sung](BLEND_RESULTS.md).
- [CSV incumbent đang giữ](../preference_augmentation/outputs/submission_candidate.csv).
- [CSV neural challenger](outputs/submission_neural_challenger.csv):596 ID,410 được chọn.
- [CSV blend](outputs/blend/submission_blend_candidate.csv):596 ID,393 được chọn;
  final inner chọn weight0.5, cutoff0.5.
- [Model neural đã lưu](outputs/neural_model/manifest.json):adapter894528 tham số
  được cập nhật; base transformer đóng băng, head5lớp. Snapshot nền dùng chung cache.
- [Xác minh primary](outputs/verification.json), [xác minh blend](outputs/blend/verification.json).
- [Lỗi từng bài](outputs/failure_transitions.csv), [theo venue/label](outputs/failure_breakdown.csv),
  [những trường hợp đã bàn](outputs/known_cases.csv).

Đã qua3 unit tests,48inner fits và10outer fits được audit,40metric rows được tính
lại,45inner choices được xác minh; model final là fit thứ59. Encode lại toàn bộ
3090titles khớp cache, nhãn gốc giữ nguyên, tải lại neural model cho cùng xác suất
và thứ tự CSV. Không dùng nhãn test hoặc gán sở thích bằng tay.
