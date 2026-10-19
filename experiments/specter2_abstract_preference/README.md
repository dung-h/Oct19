# SPECTER2 học trực tiếp từ abstract

Đối chứng representation: `title_adapted`, `joint_frozen`, `joint_adapted`, cùng
CLS768/head/loss/seed/effective batch. Đây là pilot5outer với một calibration
group (`inner_fold0`) mỗi outer; không phải mean4inner nested evaluation. Đọc
[PLAN.md](PLAN.md) trước khi diễn giải score.

- [model_abstract.py](model_abstract.py): tokenization512, fresh frozen vectors,
  microbatch8/gradient accumulation32, adapted/frozen fits và model reload.
- [run_abstract.py](run_abstract.py): protocol hash toàn input/metadata/labels,
  calibration chọn epoch/cutoff, outer evaluation và final candidate.
- [verify_abstract.py](verify_abstract.py): metric/calibration tính độc lập,
  gradients/drift, group separation, original labels, fresh vector samples,
  cả ba final model reload và CSV596 IDs.
- [write_report.py](write_report.py): paired contrasts, từng failure/group,
  traces/checkpoints, final decision sau verification.

Chạy từ root workspace:

```bash
/tmp/assignment-specter2-venv/bin/python experiments/specter2_abstract_preference/run_abstract.py preflight
/tmp/assignment-specter2-venv/bin/python experiments/specter2_abstract_preference/run_abstract.py prepare
/tmp/assignment-specter2-venv/bin/python experiments/specter2_abstract_preference/run_abstract.py all
/tmp/assignment-specter2-venv/bin/python experiments/specter2_abstract_preference/verify_abstract.py
/tmp/assignment-specter2-venv/bin/python experiments/specter2_abstract_preference/write_report.py
```

Model/source revisions dùng cùng cache local với thí nghiệm SPECTER2 trước. Base
transformer cố định; chỉ regression adapter và head nhận cập nhật ở các adapted
arms. Các vector frozen được encode lại ở recipe mới, không fit trên nhãn của
validation/test. Tất cả fit partitions/caches có protocol hash và membership.
Prerequisites là ZIP gốc, cleaned abstracts của `specter2_v2`, splits/incumbent
artifacts của `preference_augmentation` và reference caches của `claim_stability`.
Các pipeline kế thừa có hướng tái tạo trong README/RESULTS tương ứng. Môi trường
kế thừa `experiments/diagnose_bottleneck/requirements.txt`; versions thực tế và
model revision hashes được ghi trong `outputs/protocol.json`. Script yêu cầu
snapshots SPECTER2 đã có local như các thí nghiệm trước, không tự đổi revision.

Candidate/model giữ tách khỏi incumbent; không upload Kaggle. Folds đã được xem
trong nghiên cứu trước, score exploratory và không bảo đảm đạt0.7.

`preflight_micro8/` và `preflight_micro16/` lưu source, protocol và log của các
technical trials kiểm tra overhead/bộ nhớ, dừng trước mọi calibration/outer score.
Cấu hình cuối quay về microbatch8 để có VRAM headroom. Các trial không tham gia
chọn epoch, model hoặc báo cáo điểm. Effective batch vẫn32.
