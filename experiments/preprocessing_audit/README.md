# Kiểm tra preprocessing và đầu vào model hiện tại

Audit này đọc code, ma trận feature và model đã lưu; không train lại hoặc thay
submission. Có hai pipeline khác nhau: CatBoost đang giữ làm bản chính, và
SPECTER2 title-only trong thí nghiệm 8/16 epoch. Không gộp khả năng của hai
pipeline để mô tả một model duy nhất.

## 1. Các field ban đầu đi đâu?

| Field | Xử lý thực tế | Model đang giữ dùng thế nào? |
|---|---|---|
| `id` | Giữ dạng chuỗi; join bằng `split,id`; kiểm tra duy nhất và khớp template | Không là predictor. Dùng truy vết và tie-break khi priority bằng nhau |
| `title` | Decode HTML entities, bỏ tags, gộp whitespace; không stemming hoặc xóa stopwords ở bước clean chung | TF-IDF/SVD32, SPECTER2 title -> Ridge, cờ lexical, tìm láng giềng |
| `abstract` bổ sung | Join theo `split,id`; sửa LaTeX preamble, kiểm tra nguồn và một số lỗi extraction; mục chưa xác minh/không usable thành chuỗi rỗng | Cờ lexical, TF-IDF láng giềng và embedding retrieval đóng băng; không có nguyên vector abstract trong CatBoost |
| `venue` | Giữ chuỗi category; fallback `unknown` | Categorical native của CatBoost; one-hot trong Ridge phụ; lọc láng giềng cùng venue |
| `year` | Parse float; feature builder có fallback 0, không scale | Một feature numeric của CatBoost. Snapshot hiện tại không thiếu năm |
| `authors` | Clean HTML/whitespace | Không đưa vào incumbent. Các phép thử metadata trước có thử encoding author nhưng chưa cho lợi ích nhất quán |
| `doi` | Giữ identifier nguồn; nhận diện một số nhóm có chung identifier | Không là predictor; dùng nguồn/lookup và duplicate grouping |
| `Label` | Giữ nhãn 1..5 từ ZIP; map sang 0..4 khi fit classifier | Target. Các history features chỉ đọc nhãn của reference train hợp lệ |

Nguồn: [load/clean](../specter2/run_experiment.py),
[sửa và chọn abstract usable](../specter2_v2/prepare_data.py).

`usable` là qua kiểm tra đang có, không bảo đảm toàn bộ abstract đã được người đọc
đối chiếu thủ công. Snapshot có 2.974/3.090 abstract usable: 2.396/2.494 train,
578/596 test. 116 mục còn lại không có abstract trong đầu vào model; chúng vẫn giữ
title. `abstracts_still_missing.csv` ở root không phải toàn bộ 116 trường hợp này:
ngoài thiếu văn bản còn có các trường hợp bị withheld hoặc chỉ dẫn tới bài khác.

## 2. Đúng 277 feature CatBoost nhận

Model đã lưu nhận `(2494,277)` khi fit và `(596,277)` khi predict.
[feature_catalog.csv](outputs/feature_catalog.csv) liệt kê đủ 277 tên cột theo đúng
thứ tự, nguồn, phép biến đổi, regex nếu có, phạm vi fit, dtype, missing/min/max và
feature importance do saved model báo.

| Khối | Số cột | Nội dung |
|---|---:|---|
| Base | 35 | venue + year + 32 title SVD coordinates + 1 Ridge rating |
| Content | 92 | 46 từ title và 46 từ abstract: 39 category matches, 5 unknown flags, word count và availability |
| Neighbors | 150 | 3 kênh tương đồng x 2 phạm vi references x 25 thống kê |

### Title -> TF-IDF -> SVD32

TF-IDF dùng word unigrams/bigrams, lowercase, `min_df=2`, `max_df=.98`,
`max_features=20000`, `sublinear_tf=True`, L2 normalization. Giữ stopwords;
không stemming/lemmatization. Token pattern mặc định giữ word tokens dài ít nhất
hai ký tự; dấu câu và một số ký hiệu toán không thành feature token riêng.

Vocabulary/IDF và SVD chỉ fit trên training partition tương ứng. Vocab của final
fit thực tế có 4.112 features; SVD nén thành 32 coordinates, không L2 lại đầu ra
SVD. Final fit giữ khoảng 12,14% TF-IDF variance. Con số này không phải tỷ lệ
thông tin sở thích được giữ lại, cũng không tự chứng minh SVD gây giảm H.

### Ridge rating phụ

SPECTER2 regression đóng băng mã hóa `title + SEP` thành 768 chiều, chuẩn hóa L2.
Ghép venue one-hot, L2 vector ghép, rồi Ridge `alpha=.01` dự đoán rating 1..5.
CatBoost nhận một số dự đoán `ridge_rating`, không nhận trực tiếp 768 chiều này.
Ridge không clip dự đoán vào1..5: saved final train có khoảng0.249..5.584 và test
có khoảng-0.011..5.123. Đây là feature regression phụ, không phải nhãn nguồn bị
thay đổi hoặc xác suất; khoảng giá trị đó tự nó không là lỗi.

Các rating đầu vào cho train được tạo bằng grouped crossfit ba phần: mỗi paper
được dự đoán bằng Ridge không fit trên paper/duplicate group đó. Khi predict dùng
reference fit partition, loại duplicate group của query nếu có overlap.

### 92 content features

Regex case-insensitive ghi có/không các từ thuộc năm nhóm: document type, problem,
method, contribution, domain. Ví dụ `epistemic` hoặc `world views` kích hoạt cờ
epistemic, còn `algorithms?` kích hoạt cờ algorithm. Mỗi cờ có evidence span trong
[content_evidence.csv](../preference_augmentation/outputs/content_evidence.csv).

Cờ bằng 1 chỉ có nghĩa tìm thấy cách viết; không xác minh paper thật sự thuộc
category đó. `unknown=1` là không tìm thấy bằng chứng, không phải phủ định ngữ
nghĩa. Bộ từ gốc còn có `students?` trong education: đây là hạn chế diễn giải đã
được phát hiện; nhánh sửa rule chưa được chọn thay incumbent.

### 150 neighbor features

Ba kênh: cosine của frozen SPECTER2 retrieval trên title+abstract (CLS, max512,
L2); cosine TF-IDF title; cosine TF-IDF abstract. Hai phạm vi: toàn reference
train hợp lệ và reference cùng venue. Vocab TF-IDF của từng kênh fit trên reference
subset tương ứng.

Mỗi trong sáu tổ hợp có 25 thống kê: total reference count, eligible support,
max similarity tới từng nhãn 1..5, và cho mỗi top-k=5/15: năm label frequencies,
expected gain, entropy, weighted similarity và count. Chỉ các reference có cosine
`>1e-8` được lấy. Weights `exp((similarity-max_similarity)/.1)`; label frequencies
được smooth bằng ba pseudo counts theo class prior của references. Các cột `p1`
... `p5` này là thống kê nhãn láng giềng, không phải xác suất đã calibration.

Code tạo toàn bộ feature: [features.py](../preference_augmentation/features.py).

## 3. CatBoost dùng feature để tạo priority

Saved model: MultiClass năm lớp, 300 trees, depth4, learning rate .05, L2=8,
seed20261018; `venue` được đánh dấu categorical. Numeric fields không standardize.
Saved defaults gồm Bayesian bootstrap, bagging temperature1 và một leaf-estimation
iteration. Không chỉ dựa vào các default giả định khi mô tả model.

Classifier trả `p1..p5`. `P(read)=1-p1`; expected usefulness gain là
`g=0*p1+.25*p2+.5*p3+.75*p4+1*p5`. Với final cutoff `.55`, priority bằng `10*g`
nếu `P(read)>.55`, còn lại bằng 0. Model không lấy argmax class rồi dùng class đó
làm priority. Cutoff/depth chọn bằng mean H của bốn inner folds.

Nguồn: [fit/predict và priorities](../preference_augmentation/run_augmentation.py).

## 4. SPECTER2 ở lượt 8/16 epoch vừa rồi

Input thực tế chỉ là `title + SEP`; tokenizer pretrained, padding theo batch,
attention mask, giới hạn128 tokens. Snapshot có max81 tokens và 0 title bị cắt.
Không đưa abstract, venue, year, authors, DOI hoặc ID vào forward pass.

Model lấy CLS768 rồi L2; head là `768 -> 64 -> ReLU -> Dropout(.1) -> 5`, vector
đầu vào head nhân10. Base transformer đóng băng; train pretrained regression
adapter và head. Loss `CE(5 classes) + .5*BCE(read)`. AdamW: head LR1e-3,
adapter LR5e-5, weight decay .01, gradient clip1, batch32. Encoder dropout tắt;
head dropout bật khi fit. 8/16 epoch được so với cùng input/head/loss.

Đầu ra cũng dùng `P(read)` và expected gain; threshold chọn bằng inner H. Cấu
hình này chưa thử học trực tiếp từ title+abstract. Nguồn:
[modeling.py](../specter2_preference/modeling.py).

## 5. Điều gì đã kiểm tra và điều gì chưa giải quyết?

- Audit trực tiếp saved frames/model: 277 cột đúng thứ tự, numeric NaN/Inf đều0;
  venue/year trong frames khớp input hiện tại. Nhãn và ID không là predictor thô.
- TF-IDF/SVD/Ridge fit trong các partitions; history labels được crossfit; grouped
  folds chống trùng title/identifier. Frozen pretrained encoding toàn corpus
  không fit trên nhãn validation/test.
- Verification cũ đã rebuild final features, reload model và đối chiếu cùng thứ
  tự submission. Xem [verification.json](../preference_augmentation/outputs/verification.json).
- Crossfit references trong final train chỉ 1.662/1.663, còn test dùng2.493/2.494
  references. Đây là thay đổi phân phối feature cần đối chứng; chưa chứng minh
  nó gây tụt H. Các query overlap duplicate group đã bị loại khỏi reference.
- `baseline.input_digest()` chỉ hash split/id/title/abstract, chưa bao phủ mọi
  field được model dùng như venue/year. Audit này kiểm tra trực tiếp metadata
  hiện tại khớp saved frames; đây vẫn là khoảng thiếu trong bảo vệ cache về sau.
- Chưa có bằng chứng toàn bộ representation giữ đủ các khác biệt về phương pháp
  hay đóng góp để dự đoán sở thích. Authors không được dùng; title bị nén; abstract
  bị rút thành các cờ/statistics; có các rule khớp từ nhưng diễn giải sai.
- Nhiều vòng nghiên cứu dùng lại outer folds. Pipeline không đọc trực tiếp nhãn
  test, nhưng score outer không còn là benchmark chưa từng được xem.

Kết luận: có kiểm soát đáng kể về nạp dữ liệu, phạm vi fit và tái lập; không thể
coi preprocessing/feature representation đã tối ưu chỉ vì các kiểm tra đó qua.
Các hạn chế trên là giả thuyết để đối chứng, không cộng chúng thành lời giải
thích chắc chắn cho H khoảng0.64.

## Tái tạo audit

```bash
/tmp/assignment-specter2-venv/bin/python experiments/preprocessing_audit/audit_current.py
```

Lệnh chỉ đọc artifacts có sẵn và ghi catalog/summary của audit; không train lại.
Số liệu máy đọc ở [audit_summary.json](outputs/audit_summary.json).

## Đối chứng sau audit

Đã triển khai riêng các sửa counts, SVD128 title, abstract SVD64, rules có evidence,
matched reference inference và nhánh kết hợp, giữ cùng classifier/folds. Kết quả:
reference H0.64028; full selection policy H0.63864. Final inner chọn lại reference.
Tăng variance retained và giảm reference-size mismatch chưa tạo cải thiện H trong
các can thiệp đã thử. Không dùng những hạn chế ở trên để kết luận chúng gây phần
lớn plateau hoặc chắc chắn sửa được lên0.7.

Lỗi wording có sửa trong module mới và cache guard mới bao phủ metadata; các
artifact gốc vẫn giữ nguyên để đối chiếu. Xem
[kết quả và failure accounting](../feature_repairs/RESULTS.md),
[code sửa feature](../feature_repairs/repair_features.py) và
[catalog mới theo từng arm](../feature_repairs/outputs/feature_catalog.csv).
