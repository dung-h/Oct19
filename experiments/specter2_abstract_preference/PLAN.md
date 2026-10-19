# SPECTER2 học preference trực tiếp từ title + abstract

## Câu hỏi và thiết kế khóa trước kết quả

Thử nghiệm representation mới, không tiếp tục chỉnh SVD/rules. Ba arm:

- `title_adapted`: title + SEP, adapter và head học theo nhãn.
- `joint_frozen`: title + SEP + usable abstract, encoder/adapter đóng băng, chỉ học head.
- `joint_adapted`: cùng joint input, adapter và head học theo nhãn.

Giữ cùng SPECTER2 base/regression adapter revisions đã dùng trước, CLS768 L2,
head768->64/ReLU/dropout.1->5, input head nhân10, loss CE5+.5BCE(read).
Không đưa metadata/ID/Label vào tokenizer. Base transformer đóng băng, encoder
dropout tắt, không resampling, không synthetic labels. AdamW headLR1e-3,
adapterLR5e-5, weight_decay.01, gradient clip1, seed20261021.

Cả ba max512; title hiện không dài tới giới hạn này. Effective batch32 gồm bốn
microbatches8, tích lũy gradient theo số mẫu rồi optimizer.step một lần; batch cuối
scale đúng actual length. Head dropout/train order/optimizer recipe giống nhau.
Checkpoints1/2/4/8. Frozen vectors được encode mới bằng cùng microbatch/pooling.
Đối chứng title được fit lại, không gọi khác biệt số học/batching là hiệu ứng abstract.

## Pilot validation và tiêu chí quyết định

Giữ năm grouped outer folds cũ. Trong từng outer train, dùng đúng inherited
inner_fold0 làm calibration, fit trên các inner_fold1/2/3. Chọn epoch và cutoff
0..0.90 bước.05 bằng H trên calibration; sau đó refit toàn outer train đúng epoch
đã chọn, chấm outer một lần. Đây là pilot với một calibration group, KHÔNG phải
mean4inner nested experiment; lựa chọn epoch/cutoff dễ nhiễu hơn. Không so trực
tiếp policy score của pilot với policy4inner mà gọi là kiểm định thuần model.

Incumbent CatBoost được tham chiếu từ các cached fit/pred partitions tương ứng,
cutoff chọn lại trên chính calibration này. `selection_procedure` chọn trong
incumbent và ba arm bằng calibration H, tie ưu tiên incumbent. Báo riêng từng
arm/policy, Q/recall/AUC, FP/FN/highFN. Hai contrasts chính joint_adapted-title_adapted
và joint_adapted-joint_frozen. Không tuyên bố hiệu quả chỉ từ một vài case.

Outer folds đã xem nhiều lần: exploratory, không là independent holdout hay
Private Kaggle. Pilot không tự động mở rộng thêm kiến trúc/epoch/hyperparams theo
outer scores. Final dùng inherited final-inner_fold0 chọn epoch/cutoff/arm rồi
fit toàn2494 và xuất candidate596 IDs riêng. Incumbent gốc giữ nguyên, không
upload. Substantive course approach là classification.

## Input, provenance và kiểm chứng

Sử dụng clean usable abstract snapshot2.974/3.090, thiếu/withheld dùng title +SEP;
không điền lời tóm tắt giả. Lưu model text, token count, số cắt512 và availability
theo split/label để audit; không dùng label để chọn input.

Preflight trước khóa protocol: kiểm tra microbatch8 trên batch dài nhất có
forward/backward hữu hạn, chỉ adapter/head nhận gradient, nguyên base giữ nguyên;
benchmark tốc độ và xác nhận tokenization không bị ảnh hưởng bởi thay nhãn. Không
chấm repaired model trên outer trong preflight.

Protocol hash plan/source/parent source/raw ZIP, full fields/labels/groups,
splits, incumbent caches, model revisions và tokenizer digests. Frozen encode
không fit trên dữ liệu/nhãn heldout. Mọi fit/checkpoint log chỉ dùng fit labels;
groups tách fit/calibration/outer. Final train/test có hai overlap publication
groups như dữ liệu gốc; encoder prediction không truy cập test label. Ghi overlap
này rõ, không gọi test là group-independent. Không dùng lịch sử nhãn cho encoder.

Verifier độc lập tính lại metric, calibration choice, selected checkpoints và
policy; kiểm tra nhãn gốc, finite probabilities, gradients/adapter drift,
saved source hashes, group isolation, truncation stats, model reload, và đúng
596 IDs. Lưu learning traces/selected epoch và failure transitions cho chẩn đoán.

## Technical amendment trước mọi model score

Trial microbatch8 có title training overhead lớn. Dừng sau epoch5 của fit đầu
tiên, trước khi đọc calibration predictions/score hoặc outer score. Trial source,
protocol, input caches, preflight và training log giữ ở `preflight_micro8/`.
Chuyển đồng nhất ba arm sang microbatch16, effective batch vẫn32. Preflight bộ
nhớ/gradient và frozen encoding chạy lại; không tái dùng predictions/vector cache
khác recipe. Không dùng training loss để chọn hyperparameter hoặc đổi số epochs.

Microbatch16 qua single-batch preflight (peak allocated6.43GB) nhưng lúc train
joint có GPU memory-used7915MiB, free43MiB và epoch đầu72.7s, gây áp lực VRAM.
Dừng trong joint calibration đầu, chưa tính calibration/outer H. Trial lưu ở
`preflight_micro16/`. Quay về microbatch8 đồng nhất ba arms, effective batch32;
protocol mới và encode/fits mới. Ưu tiên memory headroom hơn overhead title;
không thay kiến trúc, loss, LR, checkpoint grid, input hoặc validation design.
