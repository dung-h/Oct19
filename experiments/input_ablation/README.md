# Thực hiện kế hoạch input → metadata → preference adapter → final model

[RESULTS.md](RESULTS.md) ghi toàn bộ kết quả, lựa chọn cuối và lỗi còn lại. [final_outputs/submission_workflow.csv](final_outputs/submission_workflow.csv) là CSV596 ID được tạo từ cấu hình chọn bằng dữ liệu train.

```bash
python experiments/input_ablation/run_plan.py
```

Môi trường đã dùng: `/tmp/assignment-specter2-venv/bin/python`. Dependencies trong `requirements.txt`. Runner gọi các stage có cache và kiểm tra protocol; có thể tiếp tục khi gián đoạn. Audit metadata mặc định dùng snapshot đã đóng băng để giữ hash và kết quả tái tạo; thay snapshot cần thư mục thí nghiệm mới.

- `outputs`: sáu tổ hợp đầu vào, field embeddings, metadata audit, paired deltas và verification.
- `metadata_outputs`: original/verified metadata với cùng cách mã hóa và inner-selected text input.
- `../preference_adapter/outputs`: frozen/adapter random/adapter hard, calibration, seed thứ hai và verification nếu gate yêu cầu chạy.
- `final_outputs`: full-train inner search, model cuối, test predictions, toàn bộ workflow OOF, submission và kiểm tra cuối.

Tất cả labels giữ từ ZIP. Không ghi đè model/CSV baseline hoặc tự submit lên Kaggle. Các outer folds đã được dùng để phát triển nhiều lượt nên kết quả có giới hạn từ adaptive research, được nêu trong report.
