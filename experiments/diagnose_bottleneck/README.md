# Thí nghiệm phân biệt embedding, mô hình dự đoán và metadata

So sánh có kiểm soát trên đúng các nested splits của v2. Protocol ghi trước khi đánh giá; giữ nhãn gốc, fit mọi vocab/scaler trong training fold. Không thay model hoặc submission hiện có.

- [RESULTS.md](RESULTS.md): bảng kết quả, paired contrasts, phương pháp và giới hạn diễn giải.
- [FAILURE_REVIEW.md](FAILURE_REVIEW.md): phân tích cơ chế lỗi, oracle ngưỡng/xếp hạng/chọn bài, audit author và thứ tự thí nghiệm nên ưu tiên.
- [outputs/summary.csv](outputs/summary.csv): 17 họ mô hình.
- [outputs/paired_contrasts.csv](outputs/paired_contrasts.csv): thay embedding/head/metadata và chênh lệch từng fold.
- [outputs/oof_predictions.csv](outputs/oof_predictions.csv): dự đoán ngoài tập train cho từng mục.
- [outputs/verification.json](outputs/verification.json): kết quả kiểm tra toàn bộ lựa chọn inner và score outer.

```bash
python -m pip install -r experiments/diagnose_bottleneck/requirements.txt
python experiments/diagnose_bottleneck/run_diagnostics.py all
python -m unittest discover -s experiments/diagnose_bottleneck -p 'test_*.py' -v
python experiments/diagnose_bottleneck/verify_results.py
python experiments/diagnose_bottleneck/analyze_results.py
```

Có thể chạy riêng `embed`/`train`; cache từng outer fold cho phép tiếp tục nếu gián đoạn. `protocol.json` ngăn chạy lại trên cùng thư mục với cấu hình khác.
