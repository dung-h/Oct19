# Sửa feature có đối chứng

Tiếp nối [audit preprocessing](../preprocessing_audit/README.md). Xem
[PLAN.md](PLAN.md) để biết các arm và điều kiện đánh giá đã khóa trước điểm mới.
Đã hoàn tất fit/verification. Kết quả và phân rã failure ở [RESULTS.md](RESULTS.md),
quyết định final ở [DECISION.md](DECISION.md),
catalog theo saved column order ở [feature_catalog.csv](outputs/feature_catalog.csv).

| Arm | Số feature | Can thiệp so với baseline |
|---|---:|---|
| reference | 277 | Giữ nguyên; replay raw probabilities cũ |
| counts | 271 | Bỏ raw reference count, support thành tỷ lệ |
| title128 | 373 | Thay title SVD32 bằng SVD128 |
| abstract64 | 341 | Thêm khối lexical abstract64, giữ availability cũ |
| rules | 279 | Education/neural wording chặt hơn và hai tutorial flags |
| matched | 277 | Cùng model/train features; ba reference views lúc inference |
| combined | 433 | Kết hợp các sửa đổi trên và đo trực tiếp |

Các nhánh cùng CatBoost depth4/300 trees/L2=8. Khác biệt score không được diễn
giải chỉ từ số cột, variance retained hoặc feature importance. Nhánh `matched`
đồng thời thay reference content, Ridge phụ và averaging; không tách riêng hiệu
ứng reference size bằng nhánh đó.

Các file triển khai:

- `repair_features.py`: full input digest, lexical rules có evidence, count
  normalization, title128/abstract64 projections, matched history inference.
- `run_repairs.py`: replay baseline, cache/protocol guard, fit, inner selection,
  outer score và final artifacts. Không đổi source/model cũ.
- `verify_repairs.py`: kiểm chứng độc lập metric/selection, lineage/evidence,
  metadata hashing, label isolation và final inference.
- `write_report.py`: tổng hợp tất cả arm, từng failure/fold và decision sau khi
  verifier PASS. Không lấy arm outer cao nhất thay cho selection policy.
- `diagnose_failures.py`: đối chiếu read gate/gain ranker hậu nghiệm để giải thích
  delta H; chỉ dùng chẩn đoán, không chọn hybrid theo outer labels.

Chạy từ root workspace, dùng môi trường có sẵn:

```bash
/tmp/assignment-specter2-venv/bin/python experiments/feature_repairs/run_repairs.py preflight
/tmp/assignment-specter2-venv/bin/python experiments/feature_repairs/run_repairs.py prepare
/tmp/assignment-specter2-venv/bin/python experiments/feature_repairs/run_repairs.py fit
/tmp/assignment-specter2-venv/bin/python experiments/feature_repairs/run_repairs.py score
/tmp/assignment-specter2-venv/bin/python experiments/feature_repairs/verify_repairs.py
/tmp/assignment-specter2-venv/bin/python experiments/feature_repairs/write_report.py
```

Có thể thay lệnh fit bằng hai process độc lập `fit --shard 0 --shards 2` và
`fit --shard 1 --shards 2`; chờ cả hai xong rồi mới score. Không chạy các shard có
partition giao nhau. Mỗi process giới hạn bốn CPU threads.

`outputs/protocol.json` khóa full input, raw ZIP, source, splits và tất cả parent
feature/control caches. Source/input thay đổi thì dừng; dùng thư mục thí nghiệm
mới thay vì tái sử dụng kết quả khác recipe. Kết quả/candidate ở đây được giữ
riêng; incumbent không bị ghi đè và không upload Kaggle.

Các outer folds đã được nghiên cứu nhiều lần; đây là exploratory CV. Thêm
variance retained, sửa wording hoặc kiểm tra hết NaN không tự chứng minh tăng H.
