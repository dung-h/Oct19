# Failure analysis: semantic-role experiment

This note explains the locked outer-fold results in `RESULTS.md`. It is a
descriptive review of existing predictions, not a new benchmark or a reason to
select a different branch after seeing the outer folds.

## Where the errors moved

For the primary `roles_blend` branch, compared with the incumbent:

| Measure | Incumbent | roles_blend | Change |
|---|---:|---:|---:|
| Label 1 selected (FP) | 338 | 352 | +14 |
| Positive papers dropped (FN) | 313 | 291 | -22 |
| Label 4/5 dropped | 38 | 34 | -4 |
| Total binary errors | 651 | 643 | -8 |
| Mean-fold H | 0.64028 | 0.64198 | +0.00170 |
| Outer folds with fewer errors | — | 2/5 | — |

The 62 rescued false negatives were 27 Label 2, 25 Label 3, 9 Label 4 and 1
Label 5. The 40 previously selected positives that became false negatives
were 24 Label 2, 10 Label 3, 5 Label 4 and 1 Label 5. Thus the net gains were
3 Label 2, 15 Label 3, 4 Label 4 and zero Label 5. For Label 1, the model
dropped 34 old false positives but selected 48 new ones. It traded errors;
it did not broadly resolve the semantic misses.

## Are the remaining failures caused by missing extracted roles?

No. All 62 rescued false negatives and all 48 new false positives had object,
method and result sentences. Role coverage was also high among the 40 lost
true positives: 97.5% had an object, 95.0% a method and 92.5% a result. Missing
extractions do not explain the main tradeoff.

There is redundancy in the extracted representation. Among the 2,494 training
papers, object and method were the same source sentence for 399 papers (16.0%);
method and result matched for 166 (6.7%); all three matched for 62 (2.5%). The
classifier also sees the full title and abstract, so role vectors often repeat
content already present in its text features.

## What the changed cases suggest

Some rescued papers contain a clear, technically specific contribution. For
example, Label 4 paper 27, “Search-Guided Generation of Properties for Program
Analyzers,” describes testing static analyzers and reports finding bugs with
reasonable overhead. The role representation can make such contribution
sentences easier to recognize.

The same signal also promotes Label 1 papers. Label 1 paper 1640, “Guessing
Winning Policies in LTL Synthesis by Semantic Learning,” has explicit object,
method and result sentences. Label 1 paper 2547 states a completeness theorem
and its construction. The model reads these as strong research contributions,
but that is not the same as predicting this reader's preference. In papers
2182 and 426, the extractor repeats one sentence in all three roles, so the
feature block can count the same evidence several times.

These examples explain plausible failure modes; they do not establish why a
person assigned an individual rating. The assignment has one reader's labels,
and the existing paper fields do not record the reader's reasons.

## Decision and next useful test

The role arm adds a generic “what problem, how, and what result” signal. It
recovers some positive papers, especially Labels 2–4, while also rewarding
technically substantial Label 1 papers. Its 14 additional Label 1 selections
outweigh the 22 additional positive selections enough to leave only eight fewer
binary errors. Since the primary fails the predeclared promotion rule, retain
the incumbent.

Further role extraction or embedding-size changes are unlikely to address this
failure mechanism. A useful next experiment would test whether *reader
preference* can be recovered from abstract-level history: for each held-out
paper, compare its abstract embedding with the reader's accepted and Label 1
papers in that training fold, then add a positive-versus-negative similarity
margin. This must be computed from training-fold history only and compared with
the existing title-based neighborhood features. It is a new hypothesis, not a
result of this experiment, and should be judged by the same accepted/dropped,
FP/FN and high-label counts.
