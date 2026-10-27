# Qwen learned-head ranking pilot protocol

Status: implementation of branch A in `experiments/qwen_head_v1/MASTER_PLAN.md`.
The two prediction partitions have been examined in earlier work, so this is an
exploratory pilot and not a clean confirmation.

## Data and model lock

- Use the committed `qwen_remote_v1/inputs` bundle and its manifest. Do not
  export or scrape data again.
- Fit from scratch on each partition's committed fit IDs. Prediction IDs are
  never used for fitting, lambda selection, epoch selection, or policy tuning.
- Use `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, the existing title/abstract
  caps and prompt, NF4 double quantization, BF16, SDPA, q/v LoRA r8/alpha16,
  dropout .05, and the recipe in `config.json`.
- Select affine-head lambda separately in each fit set by three-fold
  publication-group CV, pooled five-class cross-entropy, grid
  `[.001,.01,.1,1,10]`, with ties within 1e-6 resolved toward the larger
  penalty. Standardization and head/prior are refit within every fold.
- Warm start joint LoRA from that fit-only frozen head. Train exactly three
  epochs (282 optimizer updates per partition) with the locked seed and
  schedule. Each partition starts from a fresh base model.

## Arms and policies

1. Incumbent native: incumbent read probability > .55; order by incumbent
   expected gain.
2. Primary: the exact incumbent selected IDs; order by trained Qwen expected
   gain. This isolates ranking under a fixed selection set.
3. Trained native: Qwen read probability > .55; order by Qwen expected gain.
4. Frozen-head control: exact incumbent selected IDs; order by frozen-head
   expected gain.
5. Diagnostics: selection-only (Qwen selection, incumbent ranking),
   ranking-only (incumbent selection, Qwen ranking), and matched budget
   (top incumbent-sized Qwen read set, Qwen ranking).

For all arms, gain is `.25*p2 + .5*p3 + .75*p4 + p5`; positive priority is
`1 + 9*gain`, and unselected priority is zero. Keep all five probabilities.
Do not force class proportions. Confusion/argmax metrics are diagnostics only.

## Integrity and gates

- Verify frozen-feature cache lineage and exact source/runtime compatibility.
  Any source/runtime change requires a new output namespace and fresh smoke.
- Fit/scoring code receives only fit labels before both prediction files for
  both partitions exist. The evaluator is the only stage that joins prediction
  labels and quality metrics.
- Each partition must pass fit-only smoke: finite loss/gradient norms, LoRA
  changes, base frozen, frozen-feature parity, and save/reload probability
  parity <= 1e-4.
- Primary selected IDs, counts, TP/FP/FN/TN, recall, missed label 4 and 5 must
  exactly match incumbent per partition.
- Promote only to a separately budgeted confirmation if mean primary H gain is
  at least .003, each partition is nonnegative, and all technical/invariant
  checks pass. Otherwise stop this Qwen branch.

Do not train the final 2,494-row model, score the 596-row test set, upload, or
replace the incumbent/submission in this pilot.
