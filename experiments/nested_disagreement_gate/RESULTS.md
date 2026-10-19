# Nested disagreement gate — verified outer OOF

| arm | mean H | mean Q | mean R | accepted | TP | FP | FN | errors | missed L4 | missed L5 | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 0.640276 | 0.532500 | 0.803238 | 1616 | 1278 | 338 | 313 | 651 | 33 | 5 | 0.799457 | 0.172009 |
| meta_gate | 0.639936 | 0.557271 | 0.758268 | 1478 | 1204 | 274 | 387 | 661 | 39 | 9 | 0.816374 | 0.166881 |

Meta gate thắng H ở 3/5 outer folds. Chuyển trạng thái: `{"unchanged": 2218, "lost_TP": 120, "FP_removed": 87, "FN_rescued": 46, "new_FP": 23}`.

Decision: **REJECT_SCORE_ONLY_GATE**. Independent verifier: PASSED (25 source partitions, feature/model/meta replay).
