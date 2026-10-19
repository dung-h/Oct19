# Đối chứng sửa feature sau audit preprocessing

## Mục tiêu và phạm vi

Đào sâu các hạn chế của incumbent CatBoost mà không quy tất cả thành nguyên nhân
tụt điểm. Lỗi đã xác nhận: bare `transformer` bị gán neural dù nói về predicate,
abstract, distribution hoặc monad transformers; standalone student cũng có thể
gán education sai ngữ cảnh. Cache input cũ chưa hash mọi field được model dùng.
Việc nén SVD32 và reference-size drift là giả thuyết cần đối chứng, chưa phải lỗi
hiệu năng đã được chứng minh.

Không sửa nhãn, không tạo preference labels, không cập nhật source/artifact đã
khóa của các thí nghiệm cũ và không upload Kaggle. Classification là phương pháp
môn học. H theo phần Evaluation chi tiết là mục tiêu chọn model; các chỉ số khác
dùng giải thích đánh đổi, không là điều kiện chấm mới.

## Các arm khóa trước khi xem kết quả mới

Tất cả dùng CatBoost MultiClass 300 trees, depth4, LR .05, L2=8, seed20261018,
categorical venue; giữ các defaults của đối chứng cùng môi trường. Không tune
thêm capacity hoặc chọn seed theo kết quả.

1. `reference`: đúng 277 features cũ. Fit lại và so raw probability với archive
   `both_d4` ở mọi partition trước khi dùng kết quả sửa chữa.
2. `counts`: chỉ bỏ sáu `references` counts và đổi sáu raw support thành tỷ lệ
   support/references. Giữ top-k counts và các feature khác.
3. `title128`: chỉ thay SVD32 title bằng SVD128, cùng vocabulary/IDF cũ fit trong
   partition. So retention/OOV và H; variance không là phần trăm thông tin nhãn.
4. `abstract64`: chỉ thêm TF-IDF abstract -> SVD64 riêng. Cùng TF-IDF word1/2,
   min_df2, max_df.98, max_features20000, sublinear TF/L2. Missing abstract thành
   vector0, dùng presence flag đã có; không thêm flag trùng lặp.
5. `rules`: cùng feature cũ, nhưng education cần wording giáo dục rõ; neural cần
   neural/deep/LLM/BERT/GPT hoặc transformer được định danh là neural/language
   model. Bare transformer không đủ bằng chứng. Thêm hai literal tutorial flags;
   tính lại unknown flags. Giữ evidence spans và ledger; không dùng Label/ID để
   viết rule. Không tự mở rộng sửa ILP vì audit chưa xác nhận ambiguity đó.
6. `matched`: giữ nguyên training frame và model của reference. Khi predict,
   dựng history bằng từng reference subset của chính ba grouped crossfits, rồi
   trung bình ba probability predictions. Nhờ vậy reference sizes tương ứng với
   feature train. Đây là can thiệp toàn history inference policy, gồm Ridge phụ,
   vocabulary references, neighbor counts/similarity/frequencies và ensembling;
   không quy mọi chênh lệch riêng cho counts hay reference size.
7. `combined`: title128 + abstract64 + rules + counts + matched. Đo trực tiếp,
   không cộng các delta của arm đơn lẻ để suy ra hiệu quả.

`matched` dùng cùng model fit với reference; các arm khác fit riêng. Khi có
duplicate groups train/test overlap, loại toàn group của query khỏi từng history
reference trước tính Ridge/prior/nearest neighbors, như đối chứng cũ.

## Đánh giá

Giữ đúng năm grouped outer folds/bốn inner folds của các thí nghiệm trước, và
final-inner membership của incumbent. Chọn cutoff0..0.90 bước0.05 theo mean4inner
H cho từng arm. `selection_procedure` chọn arm/cutoff bằng inner H, tie ưu tiên
reference theo thứ tự arm cố định. Báo riêng từng arm và toàn selection policy.

Báo mean5outer H/Q/recall/AUC, tổng FP/FN/missed-label4/5, từng fold và failure
transitions; không đồng nhất binary errors với H. Final candidate chọn lại chỉ
bằng bốn final inner folds; giữ file tách biệt với incumbent. Không promote chỉ
vì một arm có outer cao nhất. Chưa có benchmark độc lập mới: folds đã được quan
sát nhiều lần và các phát hiện audit đã dùng để thiết kế nghiên cứu này.

## Engineering và kiểm chứng

- Hash toàn raw ZIP và toàn input gồm title/abstract/venue/year/authors/doi/labels,
  grouping, splits, pretrained caches, source kế thừa và feature pack nguồn.
  Lock protocol trước mọi điểm sửa chữa; không đổi feature theo kết quả outer.
- Tái sử dụng các feature cũ đã crossfit; mọi projection mới fit chỉ trong fit
  partition, không dùng vocabulary/SVD học từ outer validation hoặc test.
- Dựng forensic ledger: thay đổi flags có bằng chứng, OOV, SVD variance,
  reference counts, missingness, dtype/column order và NaN/Inf.
- Independent verifier tính lại tất cả metric/cutoff/arm choices, kiểm tra original
  labels, duplicate isolation, cached control replay, metadata-sensitive hashing,
  zero vectors cho abstract thiếu, evidence offsets, heldout-label invariance của
  feature inference, và final model reload/rebuild.

Có thể chia precompute thành hai process độc lập, mỗi process có partition riêng
và tối đa bốn CPU threads. Không nhìn kết quả mới để thay recipe trong lúc chạy.
