# Kết quả learned head 5 nhãn cho Qwen

**Kết luận:** joint head regularized có học thêm tín hiệu xếp hạng. So với
digit-head LoRA trên cùng 298 bài, CE giảm .0667, read AUC tăng .0872 và H
tăng .0200. Tuy vậy, native cutoff .55 bỏ sót thêm 3 bài nhãn 4 và 3 bài
nhãn 5; practical gate không đạt. Kết quả chưa đủ thay incumbent hoặc chọn
cutoff mới.

## Thiết kế và phạm vi

Đã chạy ba arm trên cùng publication-group split: A là digit-head LoRA
epoch 3 đã lưu; B là frozen Qwen với affine head 5 lớp; C khởi tạo từ B rồi
học head cùng q/v LoRA ba epoch. A dùng prediction cache đã xác minh; adapter
cũ không được dùng để khởi tạo C. Lambda được chọn chỉ từ 1,196 train bài bằng
3-fold group CV; chọn `lambda=10`, mức cao nhất trong grid đã khóa. Validation
298 bài đã được xem trong check trước nên đây vẫn là phép đo exploratory,
không phải holdout mới.

Model là `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Runtime: RTX 5070 Ti, driver
595.91.07, CUDA 13.0, Python 3.11.16, PyTorch 2.12.0+cu130, Transformers
4.51.3, PEFT 0.15.2, bitsandbytes 0.48.0, BF16/NF4. Training hoàn tất
225 updates; mất 879.8 giây training, 112.4 giây scoring và 18.9 giây load/
reload. Peak PyTorch CUDA allocation là 12.45 GiB. Hidden-feature parity và
save/reload probability parity đều bằng 0; base không có gradient và cả LoRA
lẫn output head đều cập nhật.

## Native policy

`read = 1-p1`; chọn khi `read > .55`; priority dương xếp theo expected gain.
CE là logloss 5 lớp. Argmax chỉ dùng cho confusion/precision/recall.

| Arm | CE ↓ | Read AUC ↑ | Read Brier ↓ | H ↑ | Q | R | TP/FP/FN/TN | Missed 4/5 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Prior | 1.4872 | .5000 | .2402 | .4906 | .3250 | 1.0000 | 180/118/0/0 | 0/0 |
| TF-IDF control | 1.3712 | .7099 | .2115 | .5839 | .4389 | .8722 | 157/79/23/39 | 2/1 |
| A: digit-head LoRA | 1.4551 | .6319 | .2361 | .5567 | .4223 | .8167 | 147/73/33/45 | 3/2 |
| B: frozen learned head | 1.4248 | .6604 | .2223 | .5453 | .4093 | .8167 | 147/70/33/48 | 5/2 |
| C: joint learned head | 1.3884 | .7190 | .2096 | .5768 | .4730 | .7389 | 133/52/47/66 | 6/5 |

So với A, C giảm FP 21 nhưng cũng giảm TP 14 và chọn ít hơn 35 bài. H tăng
nhờ Q tốt hơn trong danh sách ngắn hơn; coverage giảm .0778. CE/AUC/H delta của
C so A lần lượt là −.0667/+.0872/+.0200. Paired group bootstrap 95% intervals
là CE [−.1472,+.0118], AUC [+.0336,+.1407], H [−.0096,+.0515]. Interval H
bao gồm 0; bootstrap chỉ mô tả độ bất định trên split đã xem.

## Tách selection và ranking

Các phép dưới đây dùng A làm anchor trên đúng cùng split; chúng không phải
so sánh với CatBoost incumbent.

| C policy | H | Q | R | TP/FP/FN/TN | Missed 4/5 |
|---|---:|---:|---:|---|---|
| Native | .5768 | .4730 | .7389 | 133/52/47/66 | 6/5 |
| Selection-only: C chọn, A xếp | .5609 | .4521 | .7389 | 133/52/47/66 | 6/5 |
| Ranking-only: A chọn, C xếp | .5761 | .4451 | .8167 | 147/73/33/45 | 3/2 |
| Matched budget: C chọn 220, A xếp | .5607 | .4224 | .8333 | 150/70/30/48 | 4/3 |

Ranking-only tăng H .0194 khi giữ selection của A. Selection-only chỉ tăng H
.0042, giảm coverage so A và làm miss nhãn 4/5 tăng. Matched-budget chọn được
thêm 3 worthwhile và giảm 3 FP so với A, nhưng vẫn miss thêm một nhãn 4 và
một nhãn 5. Ở operating point đã khóa, bottleneck lớn nhất là selection/
coverage; ranking có tín hiệu tốt hơn. Không tune cutoff trên split này.

## Nhãn 1–5

Confusion matrix của C, hàng là nhãn thật và cột là argmax dự đoán:

| True → Pred | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 87 | 17 | 7 | 0 | 7 |
| 2 | 40 | 10 | 4 | 5 | 6 |
| 3 | 21 | 9 | 8 | 6 | 6 |
| 4 | 9 | 9 | 4 | 5 | 8 |
| 5 | 6 | 1 | 1 | 4 | 18 |

Argmax precision/recall; mỗi cặp là A / B / C:

| Nhãn | Precision A / B / C | Recall A / B / C |
|---|---|---|
| 1 | .430 / .439 / .534 | .814 / .856 / .737 |
| 2 | .276 / .250 / .217 | .123 / .123 / .154 |
| 3 | .000 / 1.000 / .333 | .000 / .020 / .160 |
| 4 | .200 / .286 / .250 | .086 / .057 / .143 |
| 5 | .259 / .250 / .400 | .233 / .233 / .600 |

Đây là diagnostic argmax, không phải cuộc thi yêu cầu đoán một nhãn bằng
argmax. Priority vẫn dựa trên read và expected gain.

## Gate và bước tiếp theo

B cải thiện CE/AUC so A nhưng H giảm .0114 và missed4 tăng hai, nên không đạt
gate. C vượt các ngưỡng CE, AUC và H so A; gate vẫn **không đạt** vì missed4
và missed5 mỗi loại tăng ba. So B, C thêm ích lợi rõ trong split này: CE
−.0364, AUC +.0587; native H tăng .0314, nhưng missed4 tăng một và missed5
tăng ba.

TF-IDF control trên cùng validation có CE 1.3712, AUC .7099, H .5839: CE thấp
hơn và H cao hơn C, trong khi C có read AUC cao hơn (.7190). Vì fold đã xem nhiều lần, không coi các
chênh lệch là xác nhận độc lập. Giữ incumbent. Bằng chứng hiện tại ủng hộ
việc mang **recipe C đã khóa** sang một protocol group-confirmation riêng;
không mở thêm epoch/LR/cutoff grid trên validation cũ. Cần đánh giá coverage
và missed nhãn 4/5 theo gate trước khi cân nhắc thay mô hình.

## Kiểm chứng và artifacts

Postrun audit xác nhận source/runtime/cache hashes, group split 1,196/298,
hash cả ba adapter/head checkpoint, 225 updates, probabilities hữu hạn và
save/reload parity. Row-sum error lớn nhất là 1.20e-7 ở A và 1.09e-7 ở C do
float32 output; `sklearn.log_loss` cảnh báo và tự chuẩn hóa các hàng này.

Kết quả đầy đủ (bị gitignore) ở
`outputs/head_v1_torch212_cu130_r3/`: `decision.json`, `metrics.csv`,
`class_metrics.csv`, `confusion.csv`, `label_transitions.csv`, paired
bootstrap, probabilities, `training.jsonl`, adapter/head checkpoints và
`postrun_verification.json`. Namespace smoke đầu tiên được giữ lại với lỗi
shape logits; namespace tiếp theo được dừng trước GPU khi phát hiện cache Arm A
trùng tên thư mục training. Lần chạy r3 dùng source đã sửa, smoke pass và
evaluation hoàn tất. Không train final 2,494, predict test 596, submit Kaggle
hoặc thay incumbent.
