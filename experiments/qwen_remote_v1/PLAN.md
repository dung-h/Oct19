# Kế hoạch học sở thích với Qwen và kiểm tra author

## Mục tiêu và giả thuyết

Học sở thích của **một người dùng**, từ 2.494 paper có nhãn gốc 1–5. Paper là
đối tượng cần xếp hạng; venue không phải user/query giả. Có 596 test paper
chưa biết nhãn. Sản phẩm cuối của assignment vẫn là `id,priority`; classifier
5 nhãn là phương pháp substantive của môn học, không thay metric bằng accuracy.

Giả thuyết H1: cập nhật LoRA trên rating thật giúp Qwen phân biệt sở thích tốt
hơn cùng Qwen đóng băng trên cùng văn bản. H2: sự phân biệt đó có ích cho
selection/ranking theo metric cuộc thi. H3, chỉ kiểm tra sau H1: author có
thêm thông tin ngoài nội dung, đặc biệt khi xét chủ đề và độ hỗ trợ.

Qwen frozen binary v2 trước đây kết thúc NO_GO_FIT; không phải bằng chứng
fine-tune thất bại. Preflight cũ chỉ có 2 optimizer updates. Package này khởi
tạo thí nghiệm supervised mới; không sửa các config/output đã khóa của v2.
Không hứa đạt H .7/.89.

## 0. Dữ liệu đã đóng gói; không cần tái tạo cache trên remote

`inputs/manifest.json` chứa SHA256 từng input và provenance nguồn local.

| File đã commit | Nội dung |
|---|---|
| documents.jsonl | 2.494 document không chứa target: index, id, group, title, abstract, metadata gốc |
| targets.jsonl | Rating 1–5 nguyên gốc từ ZIP, tách khỏi prompt |
| partitions.json | Hai fit/pred splits kế thừa, group-disjoint |
| baseline.json | Full277 CatBoost `both_d4`, đúng fit/pred và row order từng split |
| manifest.json | Hash/provenance; không có test labels hoặc test document |

Abstract đúng bản cleaned được incumbent sử dụng; abstract bị loại do nguồn
không xác minh được để rỗng. Không overlay các source repair mới trong thử
nghiệm này. Không sinh thêm abstract hoặc nhãn bằng LLM. Author trong bundle
là **metadata gốc chưa xác minh**, không được gọi là verified author.

| Partition | Fit | Held-out prediction | Incumbent H | TP / FP / FN / TN |
|---|---:|---:|---:|---|
| outer1_inner0 | 1494 | 502 | .6487764769062809 | 260 / 56 / 66 / 120 |
| outer1_inner1 | 1497 | 499 | .6463847722551991 | 253 / 79 / 52 / 115 |

Hai held-out sets không giao nhau; fit sets giao nhau. Publication group không
được cắt qua fit/pred. Các folds đã được xem nhiều lần trong nghiên cứu cũ:
**pilot exploratory**, không phải holdout mới chưa từng được sử dụng. Mean H
baseline hai partition = .6475806245807401; không trộn với historical 5-fold
H .640276 hoặc public Kaggle .64349 do user cung cấp.

`export_inputs.py` chỉ ghi lại cách đóng gói trên local từ cache gốc; remote
**không chạy exporter**. `verify.py` xác nhận hashes, membership, baseline và
nhãn ZIP nếu ZIP có sẵn. Pilot không cần embeddings, nearest-neighbor history,
CatBoost runtime hoặc metadata API.

## 1. Runtime và technical gate trước quality run

Theo README: tạo venv mới, tải model đúng revision, verify CPU/tokenizer,
chạy smoke. GPU hỗ trợ BF16; NF4 double quant, BF16 compute và autocast,
SDPA; microbatch 1, không padding/packing, gradient checkpointing non-reentrant.
Không cài FlashAttention hoặc đổi attention backend sau khi thấy scores.

Smoke dùng **fit-only**, 2 vòng qua 5 nhãn, accumulation 5, 2 optimizer updates.
Đây là thao tác kỹ thuật khác cấu hình quality run và được ghi rõ `smoke_only`.
Điều kiện pass:

1. Năm đáp án `1`…`5` mỗi cái đúng một token riêng, decode round-trip đúng.
2. Chỉ LoRA q_proj/v_proj trainable; optimizer không có base parameters.
3. Loss/gradient norm hữu hạn; LoRA thực sự đổi; base không có gradient.
4. Xác suất đủ 5 lớp, hữu hạn và tổng bằng 1.
5. Save/reload adapter khớp trên 5 fit probes, max probability error <=1e-4.
6. Input hash/config/code/package versions/device được lock; quality run phải
   cùng runtime đã smoke. Không bỏ qua failed smoke.

Thay torch/driver stack để tương thích GPU là thay runtime, phải ghi lại và
smoke lại trong output mới. Không được hạ model xuống model khác rồi giữ tên run.

## 2. Pilot có đối chứng: Qwen học hay chỉ thay cách hỏi?

Mỗi partition load base riêng và tạo adapter riêng. Seed 20261027.

| Arm | Input | Trọng số | Vai trò |
|---|---|---|---|
| incumbent | Cache full277 của đúng partition | Đã khóa | Mốc thực tế hiện tại |
| frozen | Title + abstract candidate | Base, disable adapter | Đối chứng cùng input/tokenizer/runtime |
| trained | Title + abstract candidate | Base + LoRA rating-trained | Challenger |

Không dùng 4 references trong pilot này. Không dùng author/venue/year/DOI/ID
trong prompt. Nhờ vậy chênh lệch frozen→trained kiểm tra việc học rating, không
bị lẫn với việc đổi context. Hai partitions không chia sẻ adapter/checkpoint.
Không huấn luyện partition thứ hai từ adapter của partition thứ nhất.

Prompt được tạo từ allowlist title/abstract, chat template của model. Title
head <=80 tokens, abstract head <=560 tokens, missing abstract dùng literal.
Tổng prompt <=1024 tokens; vượt cap phải fail, không cắt toàn prompt. Assistant
start chưa có đáp án gold. Lưu raw/kept token counts và token hash từng doc.
Mutation check thay target/author/venue/year/ID vẫn cho cùng token sequence.

Classifier lấy đúng **5 next-token logits** của `1`…`5` tại assistant start:

```python
z = model(prompt).logits[last_position, answer_token_ids].float()
p = softmax(z)
loss = cross_entropy(z, original_rating - 1)
```

Đây là restricted five-class classification, không phải sinh chain of thought
hoặc full-vocabulary completion SFT. Không đưa gold token vào prompt. Nhãn 1–5
vẫn được dự đoán đầy đủ; binary read chỉ là tổng xác suất dùng lúc selection.

Training khóa trong config:

- NF4 base frozen; LoRA q_proj/v_proj, r8, alpha16, dropout .05, bias none.
- Một epoch cố định, shuffle seed cố định; không chọn checkpoint từ pred labels.
- AdamW lr 1e-4, weight decay .01; linear schedule, warmup 5% optimizer updates.
- Microbatch 1, accumulation16; nhóm cuối chia loss đúng số mẫu thực tế.
- Gradient clipping1; BF16 autocast cả train lẫn inference; use_cache=False.
- CE không class weights; không augmentation/oversampling/contrastive/DPO trong
  pilot đầu. Những thay đổi đó cần giả thuyết và phiên bản đối chứng riêng.

Kiểm tra `training.jsonl` sau khoảng 10 updates: finite loss/norm, VRAM, thời
gian/update và LR. Không quyết định chất lượng từ loss train. Frozen và trained
held-out probabilities được ghi đầy đủ trước khi evaluate joins held-out labels.

## 3. Đánh giá đúng assignment và phân tách nguyên nhân

Với p1…p5:

- `read = 1-p1`.
- `gain = 0*p1 + .25*p2 + .5*p3 + .75*p4 + 1*p5`.
- Native pilot: `priority = 1+9*gain` nếu `read > .55`, ngược lại 0.
- Argmax 1–5 chỉ để confusion matrix; không dùng argmax label làm priority.

Metric authoritative theo phần **Evaluation chi tiết** trong overview.md:
discounted gain Q, worthwhile recall R và `H=2QR/(Q+R)`. Đoạn overview mở đầu
còn ghi 60/40/title similarity không khớp phần Evaluation; không áp công thức
cũ đó. Tie-break theo id dạng text tăng dần. Public/private tự đánh giá hai
nhóm 298; không giả định cân bằng class hoặc ép số paper mỗi nhãn bằng nhau.

Bốn policies được report, cùng cutoff cố định, không grid search trên pred:

1. Native: Qwen chọn và xếp thứ tự.
2. Selection-only: Qwen chọn, incumbent xếp thứ tự.
3. Ranking-only: incumbent chọn, Qwen xếp thứ tự.
4. Matched-budget: chọn top read của Qwen với cùng số bài incumbent chọn,
   rồi xếp bằng incumbent gain. Đây là diagnostic trên validation; không giả
   định biết số positive test hoặc dùng budget này như chân lý cho submission.

Report H/Q/R **mean theo partition**, counts cộng theo partition; không nối
hai ranking thành một danh sách rồi tính H. Báo thêm read AUC/Brier, 5-class
logloss, raw/row-normalized confusion, precision/recall từng label, số nhận/bỏ
theo true label và paper-level changes. Phân biệt cứu FN/tạo FP/mất TP/loại FP.

Continuation gate, chốt trước khi train:

- Evidence học: mean 5-class logloss tốt hơn frozen, mean read AUC không giảm.
- Mean native H hơn incumbent >=.003, từng partition delta H >=0.
- Matched-budget giảm tổng FP >=5 so incumbent native.
- Tổng missed label4 và label5 không tăng, xét riêng từng nhãn cho cả native
  lẫn matched-budget.

Đây là **practical gate**, không phải statistical significance. Trượt gate
không tự động chứng minh representation vô dụng: xem AUC/logloss, selection-only,
ranking-only và calibration để xác định thất bại nằm ở đâu. Không đổi cutoff,
epoch hoặc gate sau khi xem pred rồi gọi đó là cùng phép thử.

## 4. Author: chỉ mở sau khi có bằng chứng học, cần implement riêng

Author có overlap train→test 69,3%, nhưng author-affinity riêng chỉ AUC .594.
Thử nghiệm author_venue trước đã cứu70 FN/loại11 FP, đồng thời tạo70 FP/mất15 TP.
Không trình bày “thêm 19 smoothed author stats” như ý tưởng mới chưa từng thử.

Phép thử mới cần trả lời **author có thông tin bổ sung sau nội dung không?**

1. Kiểm tra metadata trước: NFC/casefold/whitespace; full names; không nối người
   chỉ vì cùng surname; giữ missing/uncertain/cold-start và provenance. Metadata
   đã sửa phải có nguồn đối chiếu; không tự gọi dữ liệu gốc là đúng tuyệt đối.
2. Chỉ dùng fit labels tính lịch sử author; với train query loại toàn publication
   group và tạo cross-fitted features. Held-out author không có history trả về
   fit prior +cold-start, không gán negative.
3. **Ưu tiên một fusion head nhỏ có đối chứng**, không bắt Qwen nhớ tên hiếm:
   baseline head nhận Qwen p1…p5; challenger cùng head thêm support, shrunk mean
   rating/read affinity và một tương tác author×topic đã chốt. Topic assignment
   fit trong fit-only; không chọn topic bằng label pred. Giữ mọi yếu tố khác bằng nhau.
4. Fusion head không được fit trên Qwen predictions in-sample rồi báo held-out
   như stacking sạch. Cần Qwen out-of-group predictions trong fit để train head;
   chi phí là thêm LoRA fits. Nếu chưa có ngân sách, dừng ở planned author stage,
   không dùng predictions từ adapter đã nhìn nhãn của chính paper đó để thay OOF.
5. So sánh content-only head / +author history / +author×topic. Chỉ một interaction
   rõ ý nghĩa trước, có shrinkage mạnh khi support ít. Không spam hàng chục k.
6. Report theo known-author/cold-start và support1/2–4/5+, cộng FP/FN transitions
   từng nhóm. Giữ author chỉ khi lợi ích ổn định ngoài topic, không chỉ fit tốt hơn.

Nếu thử đưa tên author vào Qwen trực tiếp, cần arm metadata-control cùng venue/
year, khác đúng author; mỗi adapter cùng seed/epoch. Không so title-only với
title+author+venue+year rồi gán toàn bộ improvement cho author. Đây là nhánh
thứ hai, chưa chạy song song với pilot đầu.

## 5. Confirmation, calibration và submission: task tiếp nếu gate đạt

Các bước này **chưa được run.py implement**. Remote agent cần implement sau
khi có decision.json; không dùng `--partition` tùy ý để giả vờ có nested CV.

1. Viết và lock protocol confirmation trước khi chạy thêm: 5 outer ×4 inner,
   group memberships kế thừa; export inputs/baselines tương ứng và hashes.
   Mỗi adapter khởi tạo từ base; inner calibration không thấy outer labels.
2. Chọn cutoff từ inner OOF với grid đã chốt `[0,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]`,
   tie-break gần .55 rồi cutoff nhỏ hơn. Chọn strategy/calibration chỉ trong
   inner; không fit temperature hoặc chọn budget trên outer/test.
3. Fit adapter trên outer fit, áp policy chọn từ inner, score outer. So sánh
   incumbent trên cùng groups; báo mean H, per-fold deltas và transition counts.
   Folds cũ vẫn exploratory do đã được xem nhiều lần; bootstrap không xóa hạn chế đó.
4. Chỉ refit final trên 2.494 nhãn khi confirmation đủ thuyết phục. Chốt policy
   từ train OOF, nạp 596 test docs label-free đúng cleaned source, export
   `id,priority`, tất cả ID đúng một lần, priority finite nonnegative.
5. Xuất thêm `test_probabilities.csv` với p1…p5 và predicted_label để audit;
   Kaggle nhận file priority, không phải file nhãn. Không áp cân bằng 5 lớp.
6. Xác minh source hashes và lệnh tái tạo; user tự submit hoặc yêu cầu riêng.
   Giữ nguyên submission_full277.csv và incumbent cho đến khi có bằng chứng thay thế.

## 6. Deliverables và tiêu chí hoàn thành task remote hiện tại

Task hiện tại hoàn thành khi hai partition có PREDICTIONS_COMPLETE, evaluator
tạo đủ 6 CSV +decision.json, và agent báo có/không có tín hiệu học, gain/loss
so incumbent cùng hạn chế. Nếu technical failure, ghi lỗi/runtime rõ ràng;
không báo quality failure khi model chưa train/score xong.

Không cần scraping, thêm abstract, thử graph hoặc đổi embedding trong task
này. Không tune hàng chục biến khi pilot đầu chưa trả lời được H1. Sau report,
chọn bước tiếp dựa evidence: learned+gate pass→confirmation; learned nhưng
selection yếu→calibration/author protocol riêng; không learned→diagnose train
signal/input sufficiency trước khi thêm loss hay model lớn hơn.
