# Chẩn đoán nguyên nhân ordinal giảm H

Không đồng nhất triệu chứng (recall giảm) với nguyên nhân. Đã xem kết quả ordinal
gốc; các đối chứng mới dưới đây chốt trước khi tính outer score của chúng.
Dataset/folds đã được nghiên cứu nhiều lần, kết quả là exploratory.

1. Không fit: crossed gate/ranker2x2; Shapley accounting của deltaH; bỏ riêng PAV
   khỏi read gate (giữ ranking); giữ ngưỡng của multiclass; oracle read cutoff chỉ
   dùng nhãn thật để chẩn đoán, không tạo submission; phân rã logloss do mass0.
2. Audit input: hash feature cache giống nhau, nonfinite/constant fields, title32SVD
   variance, feature groups, history train/validation reference sizes và support
   drift. Chúng là dấu hiệu/giới hạn, không tự chứng minh là nguyên nhân.
3. Giữ277features, thay gói regularization depth4/L2=8 thành depth3/L2=32:
   ordinal mới4binary, đối chứng multiclass regularized đã cache trong claim_stability.
4. Giữdepth4/L2=8, bỏ150`neighbor_` features (còn127): ordinal mới4binary,
   multiclass `content_d4` đã cache làm đối chứng. `ridge_rating` vẫn giữ và vẫn
   cross-fit; phép thử không loại toàn bộ target-derived features.

Tất cả cùng300iterations,lr0.05,seed20261018,CPU4; PAV và priority như ordinal
gốc; ngưỡng0..0.90 bước0.05 chọn mean4inner H,5outer membership giữ nguyên.
Không dùng outer labels để chọn ngưỡng/config. Từng đối chứng chỉ đổi một gói;
không tuyên bố tách riêng effect của depth và L2 hoặc loại trừ mọi vấn đề feature.

Giữ riêng sáu families: MC_full,ORD_full,MC_regularized,ORD_regularized,
MC_without_neighbors,ORD_without_neighbors; báo thêm whole inner selection policy.
Tính paired deltas và interaction; khả năng overfit/feature limitation chỉ được
kết luận theo mức bằng chứng từ train/validation và các can thiệp thực tế.

Final:4inner membership toàn train như trước; lưu8new ordinal models và2new
multiclass controls, probabilities, candidate CSV và recipe; giữ nguyên model cũ.
Reload models kiểm tra raw predictions/priority/rank; audit old/new cache hashes,
group separation, nhãn nguồn, và metric độc lập. Không gửi Kaggle.

Tham khảo: [CatBoost tuning](https://catboost.ai/docs/en/concepts/parameter-tuning).
