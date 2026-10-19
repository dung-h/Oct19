# Đối chứng CatBoost multiclass và cumulative ordinal

Kế hoạch chốt trước khi xem outer score của nhánh ordinal. Tiếp tục công thức H
trong phần Evaluation đã chép; chưa xác nhận được mâu thuẫn với đoạn Overview60/40.
Không upload Kaggle hoặc sửa nhãn/model cũ.

- Dùng đúng2494labels gốc,596test,5outer/4inner groups và4final inner membership
  đã lưu trong `preference_augmentation/outputs`.
- Đúng277features của incumbent `both_d4`, cùng thứ tự cột. Tái sử dụng feature
  cache đã cross-fit lịch sử trong training partition; kiểm tra hash và lineage.
- Đối chứng `multiclass`: raw probabilities `both_d4` đã lưu, tái lập H0.6402759969.
- Nhánh `ordinal`:4CatBoostClassifier dự đoán P(y>=2), P(y>=3), P(y>=4), P(y>=5).
  Mỗi model depth4,300iterations,lr0.05,L2=8,Logloss,seed20261018,CPU4.
  Không class weighting/resampling/early stopping hoặc tìm hyperparameter khác.
  Đây là4binary ensembles thay một multiclass ensemble; dung lượng/compute tổng
  khác nhau, nên so sánh toàn formulation chứ không tuyên bố isolate chỉ riêng loss.
- Chiếu4xác suất mỗi bài lên dãy không tăng bằng equal-weight least-squares PAV.
  Phép chiếu không dùng nhãn hoặc dữ liệu khác; giữ nguyên tổng4tail probabilities.
  Log số dòng crossing, độ sửa P(read) và kiểm tra gain giữ nguyên.
- P(read)=q2; E[gain]=(q2+q3+q4+q5)/4. Probability5lớp:
  (1-q2,q2-q3,q3-q4,q4-q5,q5). Báo H/Q/R,FP/FN/high_FN,AUC,logloss,MAE gain,
  thêm AUC/Brier từng binary task trước/sau projection để chẩn đoán.
- Cùng policy cũ: chọn khi P(read)>threshold; priority10*E[gain]. Ngưỡng0..0.9
  bước0.05 chọn theo mean4inner H riêng cho từng family. Policy chung chọn family
  thắng inner, ties ưu tiên multiclass. Outer không dùng để chọn threshold/family.
- Final chọn lại trên4inner toàn train, fit4ordinal models trên toàn2494labels,
  lưu model và `submission_ordinal.csv`. `submission_candidate.csv` theo policy
  inner; nếu chọn multiclass thì giữ đúng CSV incumbent. Không ghi đè incumbent.
- Xác minh: tests cho target mapping/PAV/coherent probabilities; hash source,
  labels,ZIP,splits/features;90inherited history crossfits; mọi inner choices và
  outer metrics; reload4models và dựng lại test features độc lập, cùng priority/rank,
 596ID hợp lệ. Báo chuyển đổi lỗi so với incumbent và projection diagnostics.

Không thêm calibration, blend, đổi encoder hoặc tune seed sau khi xem kết quả.
Outer dataset đã được xem trong nhiều nghiên cứu; score là exploratory local CV,
không chứng minh điểm Public/Private Kaggle. H>0.7 là mục tiêu, không là lời hứa.

Tham khảo về ordinal binary subtasks và consistency:
[CORAL](https://arxiv.org/abs/1901.07884). Thí nghiệm dùng4CatBoost độc lập +PAV,
không phải triển khai neural CORAL hoặc áp dụng định lý của CORAL cho CatBoost.
