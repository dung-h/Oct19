# Kế hoạch: kiểm tra abstract có bổ sung tín hiệu sở thích hay không

Trạng thái: kế hoạch để triển khai; chưa có model hay kết quả mới của thí nghiệm
này. Chỉ tạo kế hoạch trong lượt hiện tại. Khi triển khai, khóa protocol/source
trước khi xem kết quả mới. Đây là classification, đáp ứng yêu cầu phương pháp
môn học. Dùng metric trong phần Evaluation chi tiết của `overview.md`.

## 1. Vấn đề và điều cần đính chính

Incumbent trên 2.494 mẫu outer OOF: nhận 1.616, bỏ 878; FP338, FN313,
bỏ sót Label4/5 là38; tổng lỗi651; mean-fold H=0.6402759969.
Semantic-role blend vừa thử đạt FP352/FN291/high-FN34: cứu paper dương nhưng
đồng thời nhận thêm Label1, chỉ giảm8 lỗi và tốt hơn ở2/5 folds.

Incumbent **đã có thông tin abstract**, không chỉ title:

- `preference_augmentation/features.py`: abstract TF-IDF neighbors, title TF-IDF
  neighbors, và SPECTER2 retrieval neighbors; global/cùng venue, nearest theo
  từng Label, phân bố top5/top15, gain, entropy và support.
- `diagnose_bottleneck/run_diagnostics.py`: SPECTER2 retrieval nhận **clean title
  + SEP + clean usable abstract**, không phải chỉ title.
- CatBoost đã có thể kết hợp nearest-per-class features để so gần nhóm Label1
  và nhóm dương. Vì vậy không gọi positive-versus-negative similarity là ý tưởng
  hoàn toàn mới hay khẳng định model cũ thiếu toàn bộ tín hiệu sở thích.

Phần thực sự cần kiểm tra: thống kê **top-k riêng trong từng nhóm nhãn**, các
chênh lệch được tính rõ, và một không gian MiniLM **abstract riêng** có bổ sung
khả năng phân biệt so với geometry title+abstract đã có hay không. Đây là một
ablation nhỏ có đối chứng, không hứa hẹn một bước nhảy lớn hay điểm0.7.

## 2. Giả thuyết và bốn arm cố định

Giả thuyết: paper có similarity cao với các abstract thuộc Label>1, nhưng thấp
hơn với Label1, có thể được phân biệt tốt hơn bằng thống kê theo nhóm thay vì
chỉ phân bố nhãn của một top-k chung.

| Arm | Feature/model | Vai trò |
|---|---|---|
| `reference` | Đúng277 features và CatBoost incumbent | Tái lập mốc gốc |
| `retrieval_margin` | 277 +43 thống kê từ SPECTER2 title+abstract hiện có | Tách hiệu ứng cách tổng hợp khỏi việc đổi embedding |
| `title_margin` | 277 +43 thống kê từ MiniLM title | Đối chứng cùng encoder và cùng feature recipe |
| `abstract_margin` | 277 +43 thống kê từ MiniLM abstract | **Primary đã chọn trước** |

Không thêm arm combined, blend, model mới, seed mới hoặc tuning top-k sau khi
xem outer. Không chọn arm có outer cao nhất làm final. So sánh primary với
reference là kiểm tra cải thiện thực tế; hai control giải thích cải thiện đến
từ nội dung abstract hay chỉ từ feature engineering/không gian khác.

## 3. Input, coverage và điều kiện sử dụng embedding

- 2.494 train,596 test; giữ nguyên ratings1–5, split/ID/order và publication
  grouping đã dùng. ID chỉ phục vụ join, lineage và tie-break; không là feature.
- Reuse `semantic_roles/outputs/texts.csv`, `embeddings.npz` và manifest: MiniLM
  `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`,384 chiều, mean pooling/L2.
  Chỉ lấy title/abstract vectors; không lấy object/method/result hoặc document kind.
- Abstract dùng được:2.396 train +578 test =2.974/3.090. Train98 và test18
  thiếu abstract. Có12 abstract bị truncation ở512 tokens của encoder; giữ recipe
  này và báo riêng, không âm thầm đổi encoder/chunking trong phép thử.
- Reuse SPECTER2 retrieval từ `diagnose_bottleneck/outputs/embeddings_retrieval.npz`
  cùng manifest. Đây là control có geometry đã có trong incumbent.
- Kiểm tra key `split:id`, input hashes, dimensions, NaN/Inf, norm và zero vectors.
  Abstract có text nhưng vector0/invalid là lỗi input phải xử lý trước scoring.
- Normalize lại vectors một cách xác định trước cosine. Similarity giữ cả giá trị
  âm; không tự cắt tất cả về0.

**Đối chứng cùng coverage:** cả ba arm mới chỉ dùng reference có abstract hợp lệ,
và chỉ kích hoạt neighborhood mới cho query có abstract. `title_margin` và
`retrieval_margin` cũng áp cùng mask này để không lẫn hiệu ứng coverage với nội
dung embedding. Query thiếu abstract dùng nguyên probability vector incumbent
lúc inference; không chọn reference tùy ý cho zero vector.

## 4. Định nghĩa chính xác43 feature mới

Với query q, chỉ xét reference được phép của fit partition; loại toàn publication
group của q trước mọi phép tính. Tạo hai scope: toàn bộ reference và cùng venue
với q. Trong mỗi scope chia thành ba nhóm, dựa **chỉ trên nhãn reference**:

- N: Label1.
- P: Label2–5.
- G: Label4–5. G là tập con của P, có chủ đích để theo dõi paper nhãn cao.

Tính trong từng nhóm C thuộc N/P/G:

1. `max_C`: similarity cao nhất.
2. `mean5_C`: trung bình top min(5,n_C) similarities.
3. `mean15_C`: trung bình top min(15,n_C) similarities.
4. `support15_C = min(n_C,15)/15`: mức đủ reference cho top15.
5. `present_C`: có reference hợp lệ trong nhóm hay không.

Tổng15 features. Thêm6 margins:

`max_P-max_N`, `mean5_P-mean5_N`, `mean15_P-mean15_N`,
`max_G-max_N`, `mean5_G-mean5_N`, `mean15_G-mean15_N`.

Như vậy21/scope ×2 scopes =42; thêm1 query-abstract-available flag =**43**.
Các arm mới có **320 features** khi ghép với277 features nguồn.

Nếu một nhóm rỗng, stats/support/present của nhóm bằng0; margin liên quan chỉ
tính khi cả hai nhóm có reference, ngược lại bằng0 và có present flags để model
nhận biết. Query thiếu abstract có toàn bộ neighborhood mới bằng0. Tie similarity
giải bằng ID string tăng dần. Không đưa raw reference count, ID hoặc label query
vào feature. Không diễn giải margin>0 như xác suất thích paper: class size và
geometry có thể ảnh hưởng cả maximum lẫn mean similarity.

## 5. Cross-fitting và reference-size drift

Reuse đúng277-feature packs và saved three-way grouped history lineage của từng
outer fit; không viết lại hoặc sửa features cũ trong thí nghiệm này.

**Dòng dùng để train classifier:** tính43 features bằng reference thuộc hai phần
train còn lại của chính lineage đó. Toàn fold/group chứa query bị loại. Không
dùng nhãn của query hay các dòng cùng group để xây neighborhood cho chính nó.

**Dòng outer validation/test:** tạo ba view43-feature, mỗi view dùng đúng một
reference subset đã dùng trong ba crossfits của fit partition, sau đó loại
group query nếu có overlap. Ghép từng view với cùng277 features dự đoán nguồn;
predict ba lần và lấy trung bình probability vector. Chỉ fit một classifier/arm.

Nhờ vậy reference sizes của phần mới tương ứng lúc train và predict. Chỉ policy
của43 features mới thay đổi;277 features cũ giữ nguyên. Tất cả arm mới cùng policy,
nên control giúp kiểm tra hiệu ứng này. `reference` predict cùng source frame;
lặp frame ba lần vẫn cho đúng probability cũ.

Hai publication groups overlap train/test đã biết phải bị loại khỏi từng
reference view cho query liên quan. Lưu refs/query/group hashes và counts làm
lineage; mọi nhãn dùng trong feature phải truy được về allowed references.

## 6. Learner, read gate và ranking được khóa

Tất cả dùng CatBoost MultiClass:300 trees, depth4, learning_rate0.05,
l2_leaf_reg8, seed20261018, categorical venue, CPU4 threads; cùng môi trường và
các defaults của source incumbent. Target là Label1–5; không class weighting,
không early stopping dùng outer labels, không tuning capacity.

Probability đọc =1-P(Label1). **Primary read gate giữ cố định strict >0.55.**
Không chọn threshold bằng outer, không thêm threshold search ở lượt này. Đây là
phép thử feature cùng operating rule; đổi calibration/threshold là hypothesis
khác nếu cần nghiên cứu sau.

Ranking các paper được chọn giữ nguyên incumbent expected gain. Có thể dùng
priority `1+9*gain_incumbent` cho paper được chọn và0 cho paper bị bỏ; phép affine
này giữ thứ tự incumbent. Tie-break theo ID string tăng dần. Như vậy delta H ở
đây phản ánh selection, không trộn thêm một ranker mới.

## 7. Validation và ngân sách

Giữ đúng5 grouped outer folds từ `specter2_v2/outputs/nested_splits.csv`.
Không cần chạy20 inner classifier fits vì arm, hyperparameters, threshold và
ranking đều đã cố định; three-way history cross-fitting vẫn bắt buộc trong
mỗi fit partition để tránh target leakage.

- Outer:4 arms ×5 folds =**20 CatBoost fits**.
- Nếu primary đạt tiêu chí bên dưới: final full-train refit4 arms =4 fits nữa.
- Tổng ngân sách tối đa **24 fits**; reuse cached pretrained embeddings,
  không inference LLM, không fine-tune, không thu thập hoặc gán thêm nhãn.

Refit `reference` và so raw probabilities với source cache đúng fit/pred/order
ở từng outer/final partition trước khi diễn giải arm mới. Sai khác quá tolerance
khóa trước (`atol1e-7, rtol1e-6`) phải được điều tra như vấn đề tái lập.

Các outer rows đã được xem qua nhiều lần. Đây là exploratory paired comparison,
không phải holdout độc lập mới. Không gọi việc đổi seed/split trên cùng rows là
có một tập đánh giá hoàn toàn mới.

## 8. Báo cáo kết quả và chẩn đoán calibration

Đối với mỗi arm, mỗi fold và tổng2.494 OOF rows, báo:

- Nhận/bỏ, TP/FP/FN/TN, FP+FN, bỏ sót riêng Label4 và Label5.
- Cứu bao nhiêu FN cũ, loại bao nhiêu FP cũ, tạo bao nhiêu FP mới, mất bao
  nhiêu TP cũ; phân rã theo Label, venue và abstract missing/truncated.
- Mean5fold H/Q/R; không lấy H của pooled OOF ranking thay cho mean-fold H.
  H vẫn là metric assignment; binary errors dùng hiểu hành vi, không thay metric.
- Read-probability AUC, Brier và calibration bins để hiểu thay đổi chất lượng
  phân biệt so với thay đổi scale probability.
- Paired per-fold contrasts: A-reference, A-title, A-retrieval và retrieval-reference.

**Đối chứng cùng số lượng được chọn, chỉ để giải thích:** trong mỗi outer fold,
lấy m là số incumbent chọn bằng gate0.55; mỗi arm mới chọn top m theo probability
đọc, tie ID. m lấy từ prediction incumbent, không từ nhãn validation. Báo lại
FP/FN/high-FN cho policy này. Nếu nó tốt hơn trong khi gate0.55 kém hơn, đó là
bằng chứng cần xem operating point/calibration; không tự chuyển policy này thành
submission hoặc dùng nó để chọn winner sau khi xem outer labels.

Đọc các case thay đổi để tìm failure patterns; các paper đã phân tích trước là
diagnostic probes, không phải benchmark semantic chưa từng thấy. Không viết rule
theo ID hay câu chữ của các case đó sau khi thấy kết quả.

## 9. Tiêu chí quyết định trước khi chạy

Primary `abstract_margin` là candidate có triển vọng chỉ khi đồng thời:

1. FP<=338 và FN<=313, ít nhất một giảm.
2. Bỏ sót Label4<=33 và Label5<=5, nên tổng high-FN<=38; không đổi Label5
   lấy thêm Label2/3 mà che trong tổng FN.
3. FP+FN giảm trong ít nhất4/5 outer folds.
4. Mean-fold H không thấp hơn0.6402759969.
5. Reference replay, leakage/lineage checks và independent verifier đều đạt.

Đây là screening rule thực dụng, không phải kiểm định chắc chắn sẽ tốt hơn trên
Kaggle. Nếu primary không đạt thì giữ incumbent; không tự chọn title/retrieval
chỉ vì có outer tốt nhất. Nếu primary tốt nhưng ngang retrieval/title control,
không quy toàn bộ cải thiện cho nội dung abstract.

Nếu mọi arm không cải thiện, dừng **hypothesis cụ thể này**. Không suy rộng thành
mọi text model đều vô ích, labels chắc chắn nhiễu, hoặc không thể vượt0.7.

## 10. Thứ tự triển khai và điều kiện hoàn thành từng bước

**A. Audit/lock:** lập input/feature inventory; kiểm tra mask, keys, norms,
source probabilities, grouping và splits. Hash raw ZIP, toàn clean input gồm
metadata/labels, embeddings/manifests, nguồn/cache277 features, lineage, source
code và plan. Chỉ sang training khi đúng provenance.

**B. Feature builder:** triển khai cùng một hàm43-feature dùng được cho ba
embedding spaces; export feature catalog và neighbor evidence. Kiểm tra bằng
tiny vectors biết trước kết quả, empty groups, missing abstract, cosine ties,
top-k thiếu support, self/duplicate exclusion. Đổi nhãn ngoài allowed reference
không được đổi feature inference. Đây là kiểm tra leakage, không chỉ test khớp
với chính implementation.

**C. Outer run:** chạy đủ20 fits đã khóa, lưu models/raw probability/fit-pred
indices/lineage từng partition. Không đổi recipe theo điểm xuất hiện sớm. Nếu
có bug kỹ thuật, lưu nguồn/kết quả cũ và ghi amendment trước khi chạy lại phần
bị ảnh hưởng; không coi sửa feature theo outer score là bug fix.

**D. Independent verification/report:** tính lại metrics/counts, source replay,
gate0.55, matched-budget selection, ties và model reload. Rebuild features cho
queries từ saved reference views, kiểm tra nhãn/query groups, missing fallbacks
và input hashes. Báo đủ bốn arms và kết luận theo rule đã khóa.

**E. Final có điều kiện:** nếu primary đạt screening, fit all2.494 labels bằng
train-history crossfit và predict596 test rows theo cùng ba-view policy. Gate
vẫn0.55; test missing-abstract fallback incumbent. Export CSV `id,priority`,596
IDs duy nhất theo template, priority finite/nonnegative; model reload/rebuild
phải tái tạo CSV. Test không có nhãn nên chỉ báo nhận/bỏ/thay đổi IDs, không
ước tính test FP/FN từ OOF.

## 11. Deliverables và giới hạn phạm vi

Thư mục: `experiments/abstract_preference_margin/`.

- `PLAN.md`; khi triển khai có `features.py`, `run.py`, `verify.py`, `report.py`.
- `outputs/protocol.json`, `data_audit.csv`, `feature_catalog.csv`, lineage và
  neighbor-evidence artifacts có provenance.
- `outputs/oof_predictions.csv`, `fold_metrics.csv`, `counts_by_label.csv`,
  `failure_transitions.csv`, `matched_budget_diagnostics.csv`, models và verifier.
- `RESULTS.md` và `DECISION.md` giải thích cả cải thiện lẫn đánh đổi.
- Chỉ khi đạt screening: final models/recipe và CSV candidate riêng.

Giữ nguyên source/artifacts cũ, ratings và incumbent submission; SHA incumbent:
`8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf`.
Không dùng author-history branch còn bug, không gọi LLM gán preference labels,
không dùng hidden test labels, không upload Kaggle trong thí nghiệm này.
