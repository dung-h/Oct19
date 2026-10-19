# Cumulative ordinal preference

`PLAN.md` chốt trước outer evaluation; `RESULTS.md` được tạo sau khi verify.
Hai family: incumbent CatBoost5lớp và4binary CatBoost +PAV. Cùng277features đã
cross-fit, membership, hyperparameters mỗi model và cách chọn priority.

```bash
/tmp/assignment-specter2-venv/bin/python -m unittest discover -s experiments/ordinal_preference -p 'test_*.py'
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_preference/run_ordinal.py all
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_preference/verify.py
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_preference/write_report.py
```

Phụ thuộc caches/data của các thí nghiệm trước trong workspace, đặc biệt
`preference_augmentation/outputs/*_features.joblib`, nested splits và incumbent
probabilities. Các file đầu vào được hash trước lúc chạy. Hỗ trợ tiếp tục từ cache;
đổi source/protocol cần output directory mới. CPU4, không cần GPU cho fit.

Ordinal CSV lưu riêng; candidate CSV tuân theo lựa chọn inner. Không có bước
upload Kaggle. `verify.py` kiểm tra group separation, independent PAV/metric,
original labels, rebuild features và model reload. Score outer là exploratory.
