# Bước tiếp: kiểm tra Qwen learned head có giúp incumbent xếp thứ tự không

Trạng thái: plan mới, chưa implement/chạy. Gate của `head_v1` vẫn là FAIL;
plan này không đổi cutoff hoặc chọn lại checkpoint để làm run cũ đạt gate.

## Quyết định

Chốt một pilot có ngân sách giới hạn: incumbent chọn bài, Qwen learned head
chỉ sắp thứ tự các bài đã được chọn. Câu hỏi chính là **Qwen có cải thiện
ranking của incumbent không**, không còn là Qwen có khá hơn digit head cũ không.
Nếu không đạt gate dưới đây, dừng nhánh Qwen hiện tại và giữ incumbent.

Lý do thử đúng vai trò này: trên split exploratory 298 bài, C ranking-only
đạt H .5761 so A .5567 khi cùng selection. Đây là tín hiệu so với digit-head
A; chưa chứng minh có ích so incumbent. Native C giảm FP nhưng miss thêm
nhãn 4/5. Vì vậy pilot mới khóa selection của incumbent theo thiết kế và
chỉ kiểm tra khả năng ranking. Kết quả cũ không được gọi là promotion.

## Dữ liệu và đối chứng

Dùng đúng inputs, group memberships và baseline probabilities đã commit ở
`../qwen_remote_v1/inputs`; không export/scrape lại hoặc cần CatBoost runtime.

| Partition | Fit | Prediction | Incumbent H |
|---|---:|---:|---:|
| outer1_inner0 | 1494 | 502 | .6487764769062809 |
| outer1_inner1 | 1497 | 499 | .6463847722551991 |

Mean baseline H .6475806245807401. Đây là hai split đã được xem trong nghiên
cứu trước, vẫn exploratory. Không so với H .5768 trên split 298, historical
five-fold .640276 hoặc public .64349 như cùng benchmark.

Arm chính, khóa trước execution:

1. Incumbent native: probabilities cache, cutoff .55, expected gain.
2. **Incumbent selection + C expected-gain ranking**: cùng selected IDs, chỉ
   khác reading order. Đây là challenger chính duy nhất.
3. C native, fixed cutoff .55: diagnostic về selection/ranking, không dùng để
   đổi sang một policy khác sau khi đọc prediction labels.
4. Incumbent selection + B frozen-head ranking: control phụ rẻ, tận dụng B
   đã fit để warm-start C; không thêm model family hoặc GPU fits.

## Recipe được giữ từ head_v1

- Đúng Qwen2.5-7B-Instruct revision
  a09a35458c702b33eeacc393d103063234e8bc28, title/abstract caps 80/560, prompt
  không đổi; NF4/BF16/SDPA. Giữ đủ p1…p5.
- Mỗi partition khởi tạo mới từ base, độc lập với partition còn lại. Không
  tiếp tục từ adapter diagnostic 1,196 bài hoặc adapter của partition khác.
- Trong mỗi fit: group CV 3 folds chọn lambda từ [.001,.01,.1,1,10] bằng
  pooled OOF CE, tie trong 1e-6 chọn penalty lớn hơn. Scaler/prior/head fit
  lại trong từng CV train fold; refit B chỉ trên toàn fit của partition đó.
- Joint C dùng cùng head/scaler B; q/v LoRA r8/alpha16/dropout .05, AdamW
  1e-4, LoRA wd .01, head explicit L2 đã chọn, bias không phạt. Không tune LR.
- Đúng 3 epochs: mỗi partition 282 updates, tổng 564. Endpoint chính epoch3,
  không chọn epoch2 dựa trên learning curve đã xem. Giữ seed/shuffle recipe.
- Frozen features có thể dùng chung giữa partitions vì encoder chưa học nhãn;
  scaler, CV head, lambda, LoRA adapters và labels fitting không dùng chung.
  Cache chỉ reuse sau khi kiểm tra model/input/token/feature hashes.

## Implementation và technical checks trước chạy

Tạo package/output namespace riêng. Không sửa nguồn đã lock của run r3.
Tách rõ source-cache evidence khỏi runtime/postrun evidence của run mới.

- Dùng input membership và ID order để lấy đúng incumbent probabilities.
- Kiểm tra cả hai baseline H bằng evaluator trước training.
- CPU/GPU hidden và head-probability parity <=1e-4; đúng shape [batch,5].
- Smoke fit-only với đủ năm labels: LoRA/head đổi, base frozen, loss/norm
  hữu hạn và save/reload adapter + head + scaler parity <=1e-4.
- Source/runtime đổi thì output mới và smoke mới; giữ failed artifacts.
- Không resume optimizer giữa fit. Không chạy hai GPU fits đồng thời.
- Tính CE với probabilities đã chuẩn hóa hàng để tránh warning do cast
  float32→float64; lưu raw probabilities và sai số row sums để audit.
- Prediction labels chỉ join sau khi cả hai partitions đã score xong;
  không tune cutoff, checkpoint, blend weight hoặc policy bằng các labels đó.

## Đánh giá và stop rule

Priority ranking-only bằng `1+9*Qwen_gain` cho đúng selected IDs incumbent,
còn lại 0. Với `gain=.25*p2+.5*p3+.75*p4+p5`, giữ đủ năm nhãn và không ép
class proportions. Selection giữ nguyên nên TP/FP/FN/TN, R và missed4/5
phải khớp baseline từng partition; đây là invariant bắt buộc, không phải kết
quả học selection của Qwen.

Gate cho challenger chính:

- Mean H hơn incumbent >=.003 và từng partition delta H >=0.
- ID alignment, selection invariants và technical evidence đều pass.
- Báo per-fold Q/H delta và paired group bootstrap như uncertainty mô tả;
  gate không phải kiểm định significance hoặc bảo đảm public/private score.

Report cả ba comparator và B control, CE5/read AUC/Brier, confusion 5×5,
precision/recall từng nhãn, ranking changes và class-wise gain contribution.
Mean H tính theo partition; counts cộng, không nối hai ranking thành một.

Đạt gate → chỉ đáng một protocol confirmation nhiều group folds/seeds riêng
trước khi promote. Trượt gate → dừng nhánh Qwen, không mở blend/epoch/model-size
grid hoặc đổi mục tiêu theo held-out outcomes. Giữ incumbent và hoàn thiện
reproduction/report; dữ liệu về lý do sở thích hoặc lịch sử author chỉ mở như
một phase có đối chứng/lineage riêng khi có evidence, không mặc định sẽ giúp.

## Ngân sách và deliverable

Tối đa hai joint quality fits cho pilot này. Mốc cũ ~3.9s/update tương ứng
khoảng 37 phút training cho 564 updates, chưa gồm features/loading/scoring.
Dự trù compute khoảng 45–60 phút nếu stack/tốc độ tương tự; đo lại sau 10
updates. Thời gian implement/verify riêng, không nằm trong estimate GPU.

Deliverable: source/config/protocol lock, smoke, hai completed checkpoints và
prediction files, baseline replay checks, per-fold metrics, decision.json,
report tiếng Việt và hash verification. Checkpoints/logs ở ignored outputs.
Không final2494, score test, upload Kaggle hoặc thay incumbent trong pilot.
