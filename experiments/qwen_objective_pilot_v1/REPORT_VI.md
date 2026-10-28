# Báo cáo screening objective Qwen

## Kết quả và quyết định

CPU screening hoàn tất trên hai fit partitions, mỗi partition có 3 shuffled
group folds. **Không arm challenger nào đạt gate; không chạy GPU fit, Qwen
training mới hay augmentation.** Đây là quyết định theo gate screening đã khóa,
không phải kết luận mọi loss/augmentation đều không thể có tác dụng.

| Partition | Arm | Lambda | OOF CE | Read AUC | RPS | CE gain vs CE | Eligible |
|---|---|---:|---:|---:|---:|---:|---|
| outer1_inner0 | CE control | 10 | 1.430231 | .713653 | .170765 | — | — |
| outer1_inner0 | CE + ordinal RPS | 10 | 1.426567 | .717457 | .169728 | +.003664 | Không |
| outer1_inner0 | Weighted CE | 10 | 1.430524 | .711997 | .170741 | −.000293 | Không |
| outer1_inner1 | CE control | 10 | 1.442955 | .692706 | .169923 | — | — |
| outer1_inner1 | CE + ordinal RPS | 10 | 1.439750 | .696576 | .168960 | +.003205 | Không |
| outer1_inner1 | Weighted CE | 10 | 1.442812 | .691025 | .169878 | +.000143 | Không |

Ordinal CE improved in both partitions, and its RPS also decreased slightly.
The CE gains (.0032–.0037) were below the locked .01 requirement. Weighted CE
did not improve sufficiently; its read AUC was below CE in both partitions.
No family qualified to proceed to GPU.

## Execution evidence

- Inputs were frozen cached features and fit-label caches only; three CV groups
  stayed disjoint in each partition. Prediction labels were not loaded.
- v1 screening artifacts are preserved. Inspection found it used deterministic
  GroupKFold despite the planned seed. v2 uses shuffled GroupKFold with seed
  20261103 and is the only run used for this decision.
- `verify.py`: loss/probability contracts and both partition artifacts verified.
- GPU and augmentation stages were not run. Existing incumbent and previous
  Qwen pilot outputs were untouched. No final2494, test596 prediction, or upload.

Detailed OOF results and CV fold sizes are in `outputs/objective_screen_v2/`.
The outer partitions have been inspected in earlier work; this screening is
exploratory and does not estimate fresh held-out competition quality.
