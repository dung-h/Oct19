# Roadmap cải thiện preference prediction

Trạng thái: **lập kế hoạch trước execution** theo yêu cầu user. Plan ranking
chưa được giao agent khác và chưa chạy. Không tự động khởi chạy GPU hoặc
train final từ tài liệu này. Các run trước và incumbent được giữ nguyên.

## 1. Mục tiêu và bằng chứng hiện có

Mục tiêu là tăng H so incumbent trên cùng partitions, giữ xác suất đủ nhãn
1–5, priority 0 loại paper và priority dương xếp bằng expected gain.

- Qwen learned head đã học thêm thật trên split exploratory 298: C so digit
  head A có CE −.0667, read AUC +.0872, H +.0200, nhưng missed4/5 mỗi loại
  tăng ba. Gate của thí nghiệm đó vẫn FAIL.
- C ranking-only có H .5761 so A .5567 ở cùng selection. Đây mới là evidence
  cho việc thử vai trò ranking, chưa chứng minh giúp incumbent.
- TF-IDF cùng split có CE thấp hơn và H cao hơn C, nhưng read AUC thấp hơn C.
  Không xem một model lớn là điều kiện đủ để học sở thích cá nhân.
- Các hướng cũ gồm ordinal CatBoost, contextual references, semantic roles,
  feature repairs và author-history rộng đã có kết quả; không giới thiệu lại
  chúng như phương pháp mới. Crossencoder V3 đã sửa activation collapse, chạy
  50/50 fits nhưng vẫn thua incumbent cả 5 outer folds.
- Nghiên cứu learning curve cũ cho thấy thêm nhãn thật giúp trong phạm vi
  đã đo; chưa xác định thiếu mẫu, representation hay thông tin sở thích nào
  là giới hạn cuối cùng. Không ngoại suy H .7 hoặc hứa model lớn hơn sẽ chữa.

Chỉ có hai nhánh quality pilot trong roadmap hiện tại: A về ranking, B về
author×topic. Mỗi nhánh có giả thuyết, control và stop rule riêng.

## 2. Benchmark chung và integrity

Dùng inputs có manifest tại `../qwen_remote_v1/inputs`, đúng hai partitions:

| Partition | Fit | Prediction | Incumbent H |
|---|---:|---:|---:|
| outer1_inner0 | 1494 | 502 | .6487764769062809 |
| outer1_inner1 | 1497 | 499 | .6463847722551991 |

Mean H incumbent .6475806245807401. Hai split có group-disjoint fit/pred,
prediction sets không giao nhau, fit sets giao nhau. Folds đã được xem nhiều
lần nên mọi pilot ở đây vẫn exploratory. Không nối hai ranking để tính H;
mean theo partition, counts cộng. Không so trực tiếp với split 298, historical
5-fold .640276 hoặc public .64349.

Trước quality execution: chốt config/protocol/source hashes, baseline metric
parity và ID order. Source/runtime đổi thì namespace mới và smoke mới cho
GPU; giữ failures. Không export/scrape lại input pilot. Không dùng cache
in-sample của adapter đã thấy nhãn query để fit stacking. Prediction labels
chỉ join khi đủ hai partitions; không chọn policy từ các labels này.

## 3. Nhánh A — Qwen chỉ hỗ trợ incumbent ranking

Chi tiết đã khóa trong [NEXT_PLAN.md](NEXT_PLAN.md).

| Arm | Chọn paper | Xếp thứ tự | Vai trò |
|---|---|---|---|
| Incumbent | Incumbent read>.55 | Incumbent gain | Reference |
| A-primary | Đúng selected IDs incumbent | Qwen C gain | Challenger duy nhất |
| A-native | Qwen C read>.55 | Qwen C gain | Diagnostic |
| A-frozen | Đúng selected IDs incumbent | Frozen head B gain | Control phụ |

Mỗi partition fresh base, head warm-start bằng frozen features fit-only,
lambda chọn group CV trong fit; đúng recipe C ba epoch. Tổng hai quality fits
GPU, 564 optimizer updates. Không thử blend weights, model IDs hoặc epochs.

Gate A: mean delta H >=.003, mỗi partition delta H >=0, technical checks
pass. TP/FP/FN/TN, R, missed4/5 phải đúng bằng incumbent vì selection giữ
nguyên; invariant này không phải bằng chứng Qwen học selection.

Gate FAIL → dừng nhánh Qwen hiện tại. Gate PASS → chỉ đề cử A-primary cho
confirmation. Không tự mở thêm một Qwen loss/model-size grid sau khi trượt.

Compute tham chiếu: khoảng 37 phút training từ tốc độ 3.9s/update cũ;
dự trù 45–60 phút gồm extraction/scoring/load nếu tốc độ tương tự. Đo lại
sau 10 updates. Thời gian implement/verify được tính riêng.

## 4. Nhánh B — author có thêm tín hiệu sau nội dung và topic không?

Đây là cách kiểm chứng hẹp hơn các smoothed author stats đã từng thử. Không
gán kết quả author cũ cho một ý tưởng mới: author_venue trước cứu 70 FN nhưng
tạo 70 FP và làm mất 15 TP; nó chưa là corrective run được chứng minh.

### B0. Feasibility và metadata, không dùng prediction labels

Đọc metadata bundle hiện tại cho thấy 192/2494 documents thiếu authors; mọi
record có provenance `original_dataset_unverified`. Conservative full-name
matching (comma split, NFC/casefold/whitespace, không gộp surname/initials) cho
coverage sau; support đếm distinct publication groups và loại query group:

| Partition prediction | Known author | Support1 | Support2–4 | Support5+ | Cold | Missing |
|---|---:|---:|---:|---:|---:|---:|
| outer1_inner0, 502 bài | 55.0% | 76 | 121 | 79 | 174 | 52 |
| outer1_inner1, 499 bài | 62.7% | 76 | 140 | 97 | 144 | 42 |

Đây là name overlap trên train/pred của pilot, không phải train/test overlap
và không xác nhận author identity. Trước B quality:

1. Lưu parser/provenance audit; chọn trước 30 metadata records bằng seed cố
   định 20261102, không dùng labels. Đối chiếu full names với DOI/publisher
   hoặc nguồn chính thức nếu có. Giữ missing/uncertain riêng; không tự sửa
   tên theo lỗi prediction, không gọi cả bundle là verified từ một sample.
2. Fit topic model chỉ trên fit text: TF-IDF(1,2), min_df2, max20k; SVD32 và
   KMeans K=8, seed20261102. Kiểm tra support author×topic trên pred mà không
   đọc labels. Không chọn K bằng held-out outcome.
3. Feasibility gate practical: cả hai partitions có >=20% pred documents
   với ít nhất một author×topic history ở >=2 distinct reference groups.
   Nếu không đạt hoặc parser identity chưa đáng tin, dừng B trước quality;
   không tăng feature complexity để bù thiếu support.

### B1. Ba arms CPU, cùng nội dung và learner

| Arm | Features | Vai trò |
|---|---|---|
| B-text | Capped title/abstract TF-IDF | Content-only reference |
| B-author | B-text + author history/support | Control author rộng đã biết |
| B-topic, primary | B-author + author×topic residual/support | Tín hiệu tương tác mới cần kiểm tra |

Multinomial logistic regression năm lớp, max_iter2000. Chọn chung C trong
[.01,.1,1,10] bằng pooled OOF CE của B-text trong 3-fold group CV của mỗi fit;
tie1e-6 chọn C nhỏ hơn. Giữ C đó cho cả ba arms. Không mở grid cho smoothing,
topic count, threshold hoặc feature combinations. Numerical/convergence
failure phải sửa kỹ thuật và version lại trước scoring.

Author distribution p1…p5 được shrink về reference prior với alpha20. Với
author×topic, shrink về author distribution bằng alpha10. Prior chỉ lấy từ
reference labels. Cold author trở về prior; cold author×topic trở về author
history, có flag riêng; không gán cold-start là negative.

Mỗi paper lấy mean năm xác suất history trên các authors đã parse, mean
log1p support và missing/cold flags. Interaction block là mean
`p(author,topic)-p(author)`, cùng support/topic-cold flags. Numeric blocks
scale bằng training-only statistics; text dùng TF-IDF normalization cố định.
Lưu per-query reference IDs/groups, support, priors và fitted transformations.

### B2. Cross-fitting bắt buộc

- Với mỗi training set F, tạo 3 crossfit group folds. Author target-derived
  features của query fold chỉ dùng labels từ F trừ fold đó; loại toàn query
  publication group. Pred features chỉ dùng F.
- Selection C của B-text không dùng author features. Khi bất kỳ inner CV nào
  fit B-author/B-topic (ví dụ confirmation S→V), phải xây lại topic/scaler và
  các author features bên trong S. Không precompute một OOF table toàn F rồi
  chia S/V, vì reference labels của V có thể đã vào training features của S.
- Topic/vocabulary/SVD fit trong S; author features training S lại cần
  crossfit bên trong S. V chỉ nhận transformations/history từ S.
- Không đưa adapted Qwen in-sample scores vào bất kỳ arm B nào. Phase này
  không cần thêm Qwen OOF fits hoặc CatBoost runtime.
- Validation-label mutation không thay parser, C, topic map hoặc fitted arms;
  group lineage và cold-start fallback phải được kiểm tra trước quality.

### B3. Gate và báo cáo

Cutoff .55, ranking expected gain, đủ p1…p5. Báo CEs/AUC/Brier, H/Q/R,
counts/missed4/5, confusion, transitions theo known/cold, support1/2–4/5+,
và topic support. Không viết rules riêng cho các paper/venue đang bị sai.

Primary B-topic chỉ được đề cử khi đồng thời:

1. Mean CE hơn B-author >=.01 và mean read AUC không giảm: có ích ngoài
   author history rộng, không chỉ thêm lại flat stats.
2. Mean H hơn incumbent >=.003, mỗi partition delta H >=0.
3. Tổng missed4 và missed5 không tăng, xét riêng mỗi nhãn; technical/lineage
   checks pass. Báo tổng FP/FN transitions, không gán CE improvement là H win.

B-author là control, không tự nâng nó thành challenger khi B-topic trượt.
Nếu B0/B3 fail, dừng nhánh B. Ngân sách B không có GPU; CPU fitting/feature
cost được đo ở B0 trước full pilot, không chạy grid rộng.

## 5. Thứ tự execution sau khi toàn bộ plan được chốt

1. Implement/verify A trước; bắt đầu hai GPU fits tuần tự trong tmux.
2. Trong lúc A chạy có thể làm B0 trên CPU và tài liệu lineage; B0 không đọc
   prediction labels. Chỉ implement/execute B quality nếu feasibility đạt.
3. A và B giữ output/config riêng; không đổi recipe của nhánh này theo kết quả
   nhánh kia. Từng evaluator đợi đủ partitions của chính nhánh đó.
4. Kết thúc hai pilot, viết decision matrix dưới đây trước khi đề xuất chi phí
   confirmation. Không tự động nối tới final/test/submission.

| Outcome | Hành động |
|---|---|
| Chỉ A pass | Đề cử A-primary cho confirmation |
| Chỉ B pass | Đề cử B-topic cho confirmation |
| Cả hai pass | Confirmation so incumbent/A/B bằng inner selection đã khóa; không tùy ý blend |
| Không nhánh nào pass | Dừng các quality runs trong roadmap này, giữ incumbent và hoàn thiện reproduction/report |

## 6. Confirmation là phase riêng, có budget review trước chạy

Chỉ mở nếu có challenger pass gate. Cần bundle đúng original 5 outer ×4
inner group memberships và incumbent baselines tương ứng; bundle pilot hai
partition hiện tại chưa đủ. Nếu thiếu artifacts, xác định nguồn/hash và
chuẩn bị bundle riêng; không lấy split ngẫu nhiên khác rồi gọi là replay.

Mọi chọn candidate/calibration nằm trong inner train/OOF, outer không tune.
Phải triển khai validation-label isolation của toàn pipeline. Các folds vẫn
exploratory do lịch sử đã xem; nested CV không xóa human selection bias đó.

Nếu A tham gia: 20 inner +5 outer fresh Qwen fits, khoảng 8–10 GPU giờ theo
runtime hiện tại trước overhead bất định; cần dự trù lại từ actual timings
pilot. B chỉ dùng CPU nhưng crossfit feature lineage vẫn nested. Báo mean H,
từng fold delta, high-label misses, uncertainty và chi phí, không chọn arm
outer tốt nhất để gọi là điểm của inner-selected policy.

## 7. Những hướng chưa đưa vào execution

Model LLM lớn hơn, Qwen ordinal/pairwise loss mới, additional epochs,
prompt/reference tuning, author tên trực tiếp trong prompt, broad ensembles
và stacking Qwen chưa có OOF đều chưa có budget/quality authorization trong
roadmap này. Ordinal, contextual references và semantic extraction đã có
những thí nghiệm liên quan thất bại; muốn mở lại phải chỉ ra đúng giả thuyết
mới và control, không dùng nhãn held-out để thiết kế bản sửa theo từng ca.

Nếu cả A/B thất bại, hướng còn lại là kiểm tra thông tin nào người gán nhãn
đã thấy và độ nhất quán của tiêu chí sở thích, dựa trên dữ liệu/giải thích
thật nếu có. Không tự gán lý do, sửa nhãn gốc hoặc tạo synthetic preference
labels. Incumbent đã là một kết quả hợp lệ của assignment; thêm độ phức tạp
không phải deliverable bắt buộc.

Task hiện tại kết thúc ở roadmap. Final2494/test/Kaggle vẫn là scope riêng.
