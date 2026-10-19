# SPECTER2 preference adaptation

Đọc `PLAN.md` để xem thiết kế được chốt trước outer evaluation; `RESULTS.md` sẽ
được tạo sau khi chạy. Thư mục này giữ riêng mọi artifact, không ghi đè model cũ.

```bash
/tmp/assignment-specter2-venv/bin/python -m unittest discover -s experiments/specter2_preference -p 'test_*.py'
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/run_preference.py all
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/verify.py
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/analyze_failures.py
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference/blend.py
```

Yêu cầu NVIDIA GPU và các snapshot SPECTER2 pinned đã cache local; script cố ý
không tự tải revision khác. Cần dataset, abstract đã kiểm định, saved nested splits
và các probability caches của incumbent từ các thí nghiệm trước trong repo.

`run_preference.py prepare|outer|final|report` hỗ trợ chạy từng giai đoạn; kết quả
fit được cache và protocol/hash được kiểm tra khi tiếp tục. Inner checkpoint
predictions không dùng outer labels. `verify.py` tính lại các lựa chọn/metric,
encode lại toàn bộ title và tải lại neural model cuối.

`submission_neural_challenger.csv` là neural model được chọn trên toàn train;
`submission_candidate.csv` được chọn giữa neural và incumbent bằng inner CV.
Không có bước upload Kaggle. Các outer folds này đã được nghiên cứu trước đó;
điểm local là kết quả exploratory.

Phép trộn bổ sung có kế hoạch và kết quả riêng trong `BLEND_PLAN.md` và
`BLEND_RESULTS.md`, ghi rõ được đề xuất sau khi đã xem một phần primary outer.
Weight/cutoff vẫn chọn trong inner. CSV ở `outputs/blend` không thay CSV primary.

Model tham khảo: [AllenAI SPECTER2](https://huggingface.co/allenai/specter2).
Adapter training: [AdapterHub](https://docs.adapterhub.ml/training.html).
