# Evidence-grounded semantic roles: fixed experiment

User request: improve actual selection errors using the observed semantic
differences between same-topic documents. This experiment is classification,
with frozen language models for feature extraction. No preference label is
created or changed. No human test labeling, API fees, Kaggle uploads, or old
artifact replacement.

## Fixed representation

- Use existing clean title/abstract snapshot, all 3,090 rows. LLM sees only
  numbered title and abstract sentences, never IDs, venues, years, authors,
  ratings, predictions, neighbors, or diagnostic pair labels.
- Local Qwen2.5-7B-Instruct, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, NF4 double quantization,
  float16 compute, greedy, batch2, max input2,048 tokens, max new32.
  Select one source sentence for object,
  method, result, plus one of six document kinds. null means unavailable.
  Exact source spans only. Invalid JSON/index becomes missing plus invalid flag;
  no synthetic fallback and no ID-specific prompt amendments.
- Preflight amendment: initial unconstrained generation returned invalid schema
  on13/16 hash-selected label-blind rows. Archive initial source/protocol/results
  under outputs/preflight_v1*. Before full extraction or classification, enforce
  a token-NFA JSON grammar over the existing sentence indices/null and kind
  enum. Prompt, model and semantic selection remain the same; decoding can no
  longer output descriptions in place of source indices. This is schema
  enforcement, not evidence that the chosen sentence has the correct role.
- Second preflight amendment before full extraction/scoring: grammar-valid
  responses selected null object for16/16 and null result for14/16. Add three
  generic, preference-free format demonstrations (research, evaluation,
  collection) to the prompt. These are not augmented training papers or rating
  labels. Archive v2 prompt/plan/protocol/outputs. Review the same hash sample
  descriptively; it is not an independent extraction-quality test.
- Third preflight amendment before full extraction/classification: 1.5B still
  confused background with contributions and swapped roles on descriptive
  review. Switch to the cached-tokenizer7B revision above with NF4 on the local
  8GB GPU. Preserve v3 sources/prompt/protocol/output. Change only output format
  to compact JSON array [object,method,result,kind_index] to reduce forced
  syntax decoding steps; role definitions and generic examples unchanged.
  Semantic quality is still an approximation, not established by valid syntax.
- Decoder performance amendment, before any classifier fit/score: standard
 7B extraction took about1.3 seconds/document after warmup. Archive v4
  preflight and partial full extraction. Bulk-process tokens forced uniquely by
  the same output grammar; only ambiguous tokens require a greedy-logit step.
  Prompt/model/role choices unchanged. Require exact raw-output parity with all
 16 standard-decoder preflight records before full extraction with this path.
- Further prefill-memory optimization before any classifier fit: compute the
  vocabulary projection only on the last hidden state, not every input token.
  This avoids gigabyte-size prefill logits and allows batch4. Archive v5 code
  and preflight. Again require exact output parity with the original16 records.
- The last-token projection + batch4 test matched15/16 raw outputs, so reject
  that optimization and restore the previously verified bulk-forced decoder
  with batch2 and the ordinary full vocabulary projection. Canonical preflight
  is the16/16 match archived as preflight_v5_extraction.jsonl; v6 records the
  rejected resource variant. No supervised fit/score occurred in these trials.
- Prompt and parser frozen before full extraction and any new model result.
  Preflight 16 cases selected by hash of key, without labels. Resource-only
  adjustments (batch size) require a recorded amendment before scoring.
- Encode title, abstract and the three role sentences separately using cached
  frozen MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), mean pooling/L2, max512.
  Keep all 384 dimensions per block; no SVD/projection.
- `text`: title/abstract word TFIDF, title char TFIDF, full title/abstract dense
  vectors. Each block L2, then combined L2. Word (1,2), min_df2, max_df.98,
  max30k each; title char_wb(3,5), min_df3, max30k. Sublinear TF.
- `roles`: normalized text block plus equally weighted normalized role block:
  three role word-TFIDFs (same params, max10k each), their 384d vectors,
  presence flags, document-kind onehot, extraction-invalid flag. Final L2.
  Vocabularies/IDF fit only the exact model fit partition. Input without abstract
  can use title as object, with unknown method/result; original text retained.

## Fixed learner and controls

Two LogisticRegression binary learners, target Label>1, C4, L2,
solver liblinear, max_iter2000, tol1e-5, seed20261018, no class/sample weights.
No tuning C, architecture, prompt, model revision, branch or seed by outer scores.

Four new reporting branches: text_raw, roles_raw, text_blend, roles_blend.
Blend = 0.5 incumbent P(read) + 0.5 new P(read), fixed before results.
Primary = roles_blend, not whichever branch wins outer.
Incumbent probabilities reused exactly from the original four-inner caches.

Use saved grouped 5 outer / 4 inner splits and the original final4 inner.
For each branch, choose a single threshold on the four-inner OOF predictions
within outer fit, to minimize FP subject to FN <= incumbent-inner FN and
missed Label4/5 <= incumbent-inner missed-high, at source cutoff0.55.
Tie: fewer FN, fewer high FN, then larger threshold. Candidate thresholds are
0 plus observed inner probabilities; strict >. All-selected is feasible.
No outer label used in threshold selection. This is an operational recall
constraint, not a guarantee that the constraints hold on outer/test.

Priority when selected = 1+9 incumbent expected gain, otherwise0. Thus ranking
matches original incumbent, and H changes here reflect changes in selection.
Report original incumbent with original gate and equivalent affine ranking.
H/Q/R are mean fold metrics; pooled OOF counts are summed. No pooled H is
substituted for mean-fold H.

## Integrity and budget

Lock sources/inputs/model config hashes before inference; freeze extraction
manifest before classification. Hash clean full input including all metadata
and labels, raw ZIP, clean abstract source, splits, incumbent source caches.
Pretrained label-independent extraction can be cached across folds; training
TFIDF and supervised learner only fit the relevant partition.
Author-history branch is excluded: its earlier implementation bug remains an
unresolved separate experiment and is not silently patched or reused.

Maximum 60 logistic fits (2 x [20 inner + 5 outer +4 final-inner +1 full]),
no neural fine-tuning. Resumable partition predictions include protocol hash,
exact fit/pred arrays, new model probabilities and source input hashes.
Save both outer and final fitted classifiers/vectorizers for reload checks.

## Reporting and stopping

Report every branch: selected/dropped, TP/FP/FN/TN, missed high, per-label/venue
counts, per-ID rescued/new errors, calibration metrics, AUC, mean-fold H/Q/R.
Also report text-vs-roles and raw-vs-blend contrasts, not only primary vs source.
Audit role schema, exact evidence spans, missing/invalid/truncation coverage,
hash-selected extraction sample, and previously inspected pairs as diagnostic
probes only, not an independent extraction benchmark.

Promote primary only if outer FP<=338, FN<=313, one strict, missed-high<=38,
total error lower in >=4/5 folds, and verification passes. Otherwise retain
incumbent and explain tradeoffs. No follow-on search in this experiment.
Existing outer folds have been repeatedly inspected: exploratory evidence,
no guarantee on Kaggle or score0.7.

Final refit both learners on all 2,494 labels; threshold from final four-inner
OOF by the same rule. Export all four branch CSVs and recipes,596 IDs, never
estimate test TP/FP/FN. Preserve incumbent SHA
`8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf`.

Verifier must independently compute counts/metric/threshold search, audit every
evidence span and fold membership, test parser/error/boundary constraints,
reload vectorizers and final/outer classifiers, reproduce predictions and
selected IDs/order, and verify source hashes and unchanged incumbent.
