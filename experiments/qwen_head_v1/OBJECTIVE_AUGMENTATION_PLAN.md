# Qwen objective / sample weighting / augmentation — protocol đề xuất

Ngày: 2026-10-28. Trạng thái: **planning only; chưa implement hoặc chạy fit mới**.
User yêu cầu lập plan chi tiết. Document này mô tả experiment mới sau pilot
ranking FAIL; không đổi quyết định, thresholds hoặc artifacts của pilot cũ.

## 1. Câu hỏi và evidence hiện có

Giả thuyết: CE năm lớp chưa tận dụng thứ tự nhãn, hoặc phân bố mẫu khiến head
học các lớp không đều. Chưa kết luận thiếu dữ liệu là nguyên nhân chính.
Phân biệt tác động objective, weighting và augmentation bằng đối chứng riêng.

- Pilot `qwen_rank_pilot_v1/outputs/ranking_pilot_v1_r2_torch212_cu130`:
  mean incumbent H .6475806245807401; trained ranking-only .6427022071861959;
  ΔH -.004878417394544199, cả hai partitions âm. Technical checks pass.
- Frozen→trained: mean CE tăng .0519204; read AUC tăng .0551795; evidence mixed.
  Chưa có evidence hỗ trợ tăng epoch, model size hoặc thay cutoff.
- `../ordinal_preference/RESULTS.md`: CatBoost cumulative ordinal giảm .00493 H
  so multiclass trên năm folds khác. Không so trực tiếp H với hai partitions
  Qwen; không gọi ordinal là ý tưởng chưa từng thử trong repo.
- `../preference_augmentation/RESULTS.md`: đã thử content/history features.
  Paraphrase bảo toàn nội dung cho Qwen là can thiệp khác, chưa được chứng minh.
- Author B0 metadata gate FAIL; experiment này không mở lại B1.

Các partitions đã được xem nhiều lần: mọi kết quả ở đây là exploratory.
CPU screening không chứng minh rằng joint LoRA sẽ tốt, và thất bại screening
chỉ là stop rule theo budget, không chứng minh mọi objective đều vô ích.

## 2. Data, lineage và ranh giới labels

Giữ input bundle/manifest, membership, duplicate groups, prompt/token caps và
incumbent probability cache của ranking pilot. Không export/scrape lại.
Mỗi partition t có fit F_t và prediction P_t; chỉ F_t được dùng chọn cấu hình.

- Khóa ba grouped CV folds trong F_t trước fitting; dùng lại cùng folds cho
  ba arms. Mỗi group chỉ có trong một CV fold. Seed 20261103.
- Trong mỗi split S→V, fit scaler, head, class counts/weights bằng S; V chỉ
  dùng validation. Frozen encoder/features được reuse sau hash/parity checks.
- **Chọn challenger riêng từng partition**. Không gộp CV của F_0/F_1 để chọn
  chung family/lambda: F của partition khác có thể chứa labels thuộc P_t.
- Lưu allowed label IDs và lineage cho từng fit. Training/scoring entry points
  chỉ nhận fit-label cache. Evaluator mới được join P labels khi mọi arm bắt
  buộc của cả hai partitions đã có prediction và completion marker.
- Tập test 596 không tham gia experiment; không pseudo-label hay tạo rationale
  từ held-out ratings. Không train final2494 hoặc thay submission/incumbent.

## 3. Ba arms CPU, công thức khóa

Head affine năm logits trên frozen hidden đã chuẩn hóa; p=softmax(z).
Class order [1,2,3,4,5]. Bias không bị L2. Mọi arm có cùng initialization,
folds, optimizer convergence tolerance và lambda grid [.001,.01,.1,1,10].
Giữ nguyên recipe scaler/L-BFGS hiện tại; max_iterations=2000.

### A — control CE

L_A = mean_i[-log(p_i[y_i])] + lambda/2 * ||W||².
Khôi phục frozen-head control cũ bằng đúng folds/seed cũ trong verification
riêng; screening mới dùng folds/seed mới đã khóa ở trên.

### B — CE + ordinal RPS

F_i(k)=sum_{j=1..k}p_i[j], T_i(k)=1[y_i<=k], k=1..4.
RPS_i=(1/4)*sum_k(F_i(k)-T_i(k))².
L_B=mean_i[CE_i + alpha*RPS_i] + lambda/2*||W||², **alpha=1.0 cố định**.

Softmax tạo đủ p1…p5, cumulative probabilities tự monotonic; không cần PAV.
RPS có range [0,1], báo riêng CE/RPS/penalty. Không đổi alpha theo P labels.
Đây là CE cộng ordinal probability score; không phải implementation CORAL.
Không thêm pairwise loss, binary auxiliary head hoặc focal loss trong phase này.

### C — weighted CE nhẹ

Trong mỗi training reference S, n_c là count lớp c, N=|S|:
u_c=min(2, sqrt(N/(5*n_c))); w_c=u_c / sum_c(n_c*u_c/N).
Nếu thiếu lớp trong S: báo infeasible, không sửa splits để cứu arm.
L_C=mean_i[w_{y_i}*CE_i] + lambda/2*||W||².

Cap 2 áp dụng lên u trước chuẩn hóa; lưu cả u và w. Không balanced sampler,
không weighting theo ID, lỗi đã xem hoặc ratings của prediction partition.

**Sửa prior do weighting:** head C xuất raw q; xác suất dùng mọi metric/policy
là p_c=(q_c/w_c)/sum_j(q_j/w_j). Lưu raw q và corrected p; không chọn raw hay
corrected dựa trên quality. Đây là phép sửa giải tích của weighted log loss,
không bảo đảm calibration trong model hữu hạn. Weights luôn đi cùng checkpoint.

Trong GPU loop xử lý từng mẫu: tính CE unreduced rồi nhân scalar w_y trước
accumulation. Không dùng weighted CE reduction='mean' trên batch một mẫu vì
normalization có thể triệt tiêu trọng số. Chỉ weighted phần CE, không weighted L2.

## 4. CPU screening và chọn challenger

Mỗi partition: 3 arms × 5 lambda × 3 CV fits = 45 fits; hai partitions tối đa
90 CV fits, cộng refits. Không grid thêm alpha, weight cap hoặc seeds.

1. Mỗi arm chọn lambda bằng pooled **unweighted** OOF CE trên p dùng evaluation;
   tie trong 1e-6 chọn lambda lớn hơn. C dùng corrected probabilities.
2. Lưu CE/read AUC/Brier/RPS theo CV fold và pooled; không nối ranking giữa
   folds để tính H. OOF vừa dùng selection nên chỉ là screening, có optimism.
3. B hoặc C eligible nếu pooled CE giảm >=.01 so A, pooled read AUC không
   giảm và CE cải thiện ở >=2/3 CV folds. B còn yêu cầu RPS không tăng.
   Đây là ngưỡng thực dụng khóa trước execution, không phải significance test.
4. Nếu cả B/C eligible, chọn CE thấp hơn; tie 1e-6 ưu tiên B. Lưu quyết định
   chỉ từ F_t. Mỗi partition có thể chọn family khác nhau.
5. Nếu không eligible: selector của partition trả A làm fallback. Nếu cả hai
   đều fallback, kết thúc experiment ở CPU, không mở GPU quality jobs.

Đánh giá GPU sau này là **quy trình lựa chọn objective fit-only**, không phải
chọn family tốt nhất sau khi xem hai prediction partitions. Không dùng results
của P_t để đổi lựa chọn CPU. Screening thất bại không được mở ngoại lệ chạy GPU.

## 5. Implement và checks bắt buộc trước GPU

Package mới dự kiến `experiments/qwen_objective_pilot_v1/`; không sửa helper đã
lock của ranking pilot. Reuse helper bằng wrapper hoặc snapshot có provenance.
Các CLI dự kiến (chưa tồn tại): verify, prepare, screen, smoke, train, evaluate.

- Unit checks: RPS perfect prediction=0; lệch ordinal xa tăng RPS ở one-hot;
  finite gradients cho CE/RPS/weighted CE; alpha=0/weights=1 khôi phục control.
- Weighted per-sample accumulation khớp explicit full-batch formula; penalty
  chỉ cộng một lần mỗi optimizer update về mặt toán học. Test prior correction
  và save/reload weights. Probability shape [N,5], sum=1, finite/nonnegative.
- CV scaler/weights/lambda không dùng V labels fitting; duplicate groups loại
  trùng đúng; allowed-label guard chặn P; cross-partition selector isolation.
- Frozen cache/model/tokenizer parity, CPU/GPU head p max error <=1e-4.
- Optimizer/convergence lỗi: dừng technical stage, không coi là quality FAIL.
  Sửa lỗi có source version/output namespace mới và checks mới; giữ failures.
- Smoke từng objective thực sự sẽ train: finite loss/norm, LoRA/head thay đổi,
  base frozen, adapter+head+scaler+weights save/reload parity <=1e-4.

## 6. GPU pilot có control, budget hữu hạn

Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28;
NF4 double quantization, BF16/SDPA, prompt/caps cũ; q/v LoRA r8 alpha16
dropout .05, LR1e-4, LoRA weight decay .01, grad clip1, warmup .05,
accumulation16. Train seed/shuffle 20261030, đúng 3 epochs, endpoint epoch3.

Mỗi partition chạy A mới và challenger đã chọn; nếu fallback A, reuse prediction
A trong cùng run, không train trùng. Tối đa **4 independent GPU fits**, mỗi fit
282 updates, tổng tối đa1128. Mỗi fit khởi tạo từ base và frozen head của arm đó;
lambda/scaler/weights được chọn/refit trên F_t, không warm-start adapter cũ.

- Chạy sequential trong tmux; sau khoảng10 updates báo CE/RPS/weightedCE/L2,
  grad norm, GPU memory, time/update. Log mọi update; không chọn epoch từ labels.
- Đo riêng loading/training/scoring/reload. Mốc vừa đo ~21 phút/fit gồm scoring;
  4 fits dự trù khoảng85 phút, cộng preparation/CPU/smoke, không cam kết ETA.
- Control A có cùng code path; kiểm tra objective A không bị thay đổi bởi arm B/C.
  So replay cũ như audit, không dùng chất lượng control để chỉnh cấu hình mới.
- Runtime/source thay đổi → namespace mới, smoke mới. Không sửa fingerprints.
  Không resume optimizer; fit bị gián đoạn restart ở namespace fit mới từ base.
  Chỉ reuse completed adapters/predictions có matching fingerprint.

## 7. Evaluator và quality gate

Chỉ evaluate khi cả hai partitions có A và selected-procedure predictions.
Incumbent replay bắt buộc khớp H .6487764769062809 / .6463847722551991.
Evaluator tính H đúng detailed Evaluation; tie priority bằng ID dạng text.

Primary: exact incumbent selected IDs, rank expected gain
g=.25*p2+.5*p3+.75*p4+p5; priority=1+9*g, ngoài selection priority0.
Class C luôn dùng corrected p. Không ép tỷ lệ năm lớp.

Report incumbent, new A và selected procedure; frozen heads là control phụ.
Diagnostics cho A và procedure: native cutoff .55, selection-only với incumbent
ranking, ranking-only, matched-budget theo số selected incumbent.

Gate mở confirmation yêu cầu đồng thời:

- Primary mean ΔH vs incumbent >=.003 và từng partition ΔH>=0.
- Primary mean ΔH vs new A >=.003 và từng partition ΔH>=0: hỗ trợ lợi ích
  thay objective vượt control. Fallback A có paired delta0 theo thiết kế.
- Technical, label lineage, ID alignment và selected/count invariants pass.

Counts/R/missed4/5 của primary phải bằng incumbent. H/Q/R mean theo partition;
TP/FP/FN/TN và confusion cộng counts, không nối hai rankings thành một.
Report CE5, RPS, read CE/AUC/Brier/ECE, conditional CE labels2–5, confusion5×5,
PR từng label, thêm/bỏ theo true label của native, classwise Q contributions,
paired publication-group bootstrap1000 cho ΔH vs incumbent và vs A.
Chỉ diễn giải bootstrap như uncertainty mô tả trên partitions đã xem.

FAIL → dừng objective branch, giữ incumbent; không tune alpha/weights/epoch,
không mở model-size/blend grid, không coi PR/argmax tốt lên là promotion.
PASS → đề cử confirmation có budget/protocol riêng, chưa promote hoặc train final.

## 8. Augmentation — phase có điều kiện, chưa nằm trong GPU budget trên

Chỉ đề cử sau objective pilot PASS. Cần khóa generator model/revision, prompt,
seed, chi phí và protocol riêng trước generation; nếu chưa khóa đủ thì chưa chạy.
Không tự động coi CPU screening PASS là quyền mở augmentation.

Thiết kế đề xuất: control objective đã chọn vs cùng objective + paraphrase.
Tối đa một variant/paper fit, không gửi rating cho generator; giữ technical
claims/method/numbers/entities/negation. Không dùng held-out/test text để generate.

- Audit30 variants chọn theo seed20261104, reviewer không thấy ratings; chỉ kiểm
  tra bảo toàn nội dung, không gán rating mới. Ít nhất27/30 phải đạt; failed
  audit dừng batch, không sửa prompt qua nhiều vòng để vượt audit.
- Lưu source ID/group, input/output hashes, generator provenance, checks và
  rejected variants. Lọc near-duplicates xuyên groups; splits trước generation.
- Mỗi epoch chọn original hoặc variant với xác suất .5 cho mỗi paper gốc;
  paper không có variant luôn dùng original. Số updates/epoch và khối lượng
  mỗi paper giữ bằng control, không nhân đôi trọng số bằng cách nối dataset.
- Synthetic views không tăng số nhãn độc lập. Không viết thêm claims/rationale
  sở thích, không đồng nhất paraphrase với dữ liệu preference mới.
- Gate chính vẫn H vs incumbent và paired improvement vs matched objective
  control. Chi tiết confirmation/augmentation execution được khóa riêng sau này.

## 9. Deliverables và decision tree

Source/config/protocol, input/cache/source/runtime hashes, split/allowed-label
lineage, CV predictions/lambda/weights, selector decision mỗi partition,
unit/contract checks, smoke, timing/monitor logs, adapters, raw/corrected p,
baseline replay, metrics, decision.json và REPORT_VI.md.
Logs/checkpoints/outputs gitignored; commit code/docs khi implementation thay đổi.

Decision: technical FAIL → sửa kỹ thuật và version mới; CPU gate FAIL cả hai →
dừng CPU; CPU có candidate → tối đa4 GPU fits; GPU gate FAIL → dừng; GPU gate
PASS → proposal confirmation/augmentation riêng. Hoàn tất report ở mọi terminal
state. Không có bước final2494, predict test596, upload hoặc thay incumbent.

## 10. Tài liệu phương pháp

- Current trainer: `run.py` CE + explicit head L2; AdamW cho LoRA.
- RPS ở đây định nghĩa trực tiếp trên softmax CDF; CORAL là formulation khác:
  https://arxiv.org/abs/1901.07884
- Text augmentation reference, không phải evidence trên assignment này:
  https://arxiv.org/abs/1901.11196
- CE weights/reduction semantics:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
