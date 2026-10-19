# SPECTER2 học theo sở thích — kết quả

Đối chứng matched: cùng title, pretrained regression adapter, CLS L2, head và loss.
`adapted` cập nhật adapter SPECTER2 theo nhãn; `frozen` chỉ học head. Base transformer
giữ cố định. Cả hai chọn epoch/ngưỡng bằng bốn inner folds; outer refit đúng epoch đó.

| family              |       H |   H_std |       Q |       R |   FP |   FN |   missed_high |     AUC |   train_AUC |   AUC_gap |   logloss |
|:--------------------|--------:|--------:|--------:|--------:|-----:|-----:|--------------:|--------:|------------:|----------:|----------:|
| frozen              | 0.59794 | 0.02834 | 0.50774 | 0.75333 |  361 |  397 |            81 | 0.75990 |     0.78326 |   0.02336 |   1.37466 |
| adapted             | 0.63318 | 0.02056 | 0.56622 | 0.72805 |  249 |  436 |            67 | 0.80838 |     0.90271 |   0.09433 |   1.32361 |
| incumbent           | 0.64028 | 0.01120 | 0.53250 | 0.80324 |  338 |  313 |            38 | 0.79946 |     0.87416 |   0.07470 |   1.28296 |
| neural_selection    | 0.63318 | 0.02056 | 0.56622 | 0.72805 |  249 |  436 |            67 | 0.80838 |     0.90271 |   0.09433 |   1.32361 |
| selection_procedure | 0.63277 | 0.02022 | 0.56000 | 0.73871 |  269 |  419 |            60 | 0.80709 |     0.89648 |   0.08939 |   1.32366 |

Adapted − frozen: **+0.03524 H**, cải thiện **5/5 folds**.
Điểm policy được báo riêng, không thay bằng family thắng outer.

## Chẩn đoán tại inner recall gần 0.80

Giữ epoch đã chọn theo inner H, chọn ngưỡng có mean inner recall gần0.80 nhất.
Outer recall không bị ép bằng0.80; cần đọc FP cùng recall thực tế.

| family    |       H |   H_std |       Q |       R |   FP |   FN |   missed_high |     AUC |   train_AUC |   AUC_gap |   logloss |
|:----------|--------:|--------:|--------:|--------:|-----:|-----:|--------------:|--------:|------------:|----------:|----------:|
| frozen    | 0.59096 | 0.02343 | 0.52819 | 0.68094 |  288 |  511 |           101 | 0.75990 |     0.78326 |   0.02336 |   1.37466 |
| adapted   | 0.63826 | 0.02292 | 0.55330 | 0.76014 |  276 |  384 |            58 | 0.80838 |     0.90271 |   0.09433 |   1.32361 |
| incumbent | 0.64028 | 0.01120 | 0.53250 | 0.80324 |  338 |  313 |            38 | 0.79946 |     0.87416 |   0.07470 |   1.28296 |

## Chọn cuối

```json
{
  "frozen": {
    "family": "frozen",
    "epoch": 8,
    "threshold": 0.5,
    "inner_score": 0.6125007682022681,
    "inner_recall": 0.8432811422704454
  },
  "adapted": {
    "family": "adapted",
    "epoch": 8,
    "threshold": 0.45,
    "inner_score": 0.6399621136029068,
    "inner_recall": 0.8296722384534565
  },
  "incumbent": {
    "family": "incumbent",
    "epoch": 0,
    "threshold": 0.55,
    "inner_score": 0.6386873823729493,
    "inner_recall": 0.8090378105564809
  },
  "neural_selection": {
    "family": "adapted",
    "epoch": 8,
    "threshold": 0.45,
    "inner_score": 0.6399621136029068,
    "inner_recall": 0.8296722384534565
  },
  "selection_procedure": {
    "family": "adapted",
    "epoch": 8,
    "threshold": 0.45,
    "inner_score": 0.6399621136029068,
    "inner_recall": 0.8296722384534565
  }
}
```

`submission_neural_challenger.csv` là neural arm được inner chọn trên toàn train.
`submission_candidate.csv` là kết quả global policy, có thể giữ incumbent. Các
model cũ không bị thay thế; chưa gửi Kaggle. Reload neural model và kiểm tra CSV:
`PASSED`; chi tiết `outputs/verification.json`.

## Giới hạn và tái lập

Đây là một cấu hình adapter (LR5e-5, checkpoints1/2/4/8), không kiểm tra full
fine-tuning, mọi learning rate hoặc mọi encoder. Không tự động kết luận 0.7 bất khả
thi khi cấu hình này thất bại. Outer đã được xem trong các nghiên cứu trước; nested
selection không biến nó thành benchmark độc lập. Không đổi nhãn, chọn seed theo
outer, sinh sở thích hay dùng nhãn test.

`PLAN.md`, `outputs/protocol.json` cố định thiết kế và hash. Raw probabilities,
training traces, membership và inner choices được lưu để kiểm tra lại.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/run_preference.py all
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/verify.py
```

Tài liệu: [SPECTER2](https://huggingface.co/allenai/specter2),
[AdapterHub](https://docs.adapterhub.ml/training.html).
