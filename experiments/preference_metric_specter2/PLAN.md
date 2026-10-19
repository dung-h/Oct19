# Pilot: preference-aware SPECTER2 metric learning

## Hypothesis

The frozen retrieval embedding groups papers by scientific topic, but may leave a
worthwhile paper close to an uninteresting paper.  Updating only the local
SPECTER2 regression adapter with *training-only*, semantically hard
positive/negative pairs may separate this boundary without discarding the
original scientific geometry.

This is distinct from the earlier MiniLM `preference_adapter` experiment: the
encoder is SPECTER2, text is title plus abstract up to 256 wordpieces, and the
loss contains an explicit cosine-margin term.  It is a pilot only, not a
submission candidate.

## Locked pilot partition

Use outer fold 0's fit rows, excluding its inherited inner fold 0 as the held
out probe.  The optimiser sees only the remaining fit rows and their labels.
No test row, outer-validation label, or previously inspected error list is
used to form a training pair.

For each worthwhile training row (`Label>1`), select one Label-1 paper with
maximum frozen `retrieval_margin` cosine, preferring the same venue when one
exists; publication groups must differ.  This forms one hard pair per positive
row.  Ties are broken by ID.

## Model and test

Start from the local frozen `allenai/specter2_base` plus its regression adapter.
Only that adapter and a scalar read head are trainable.  Each two-paper batch
minimises a fixed sum of: pairwise read ranking, a cosine margin that pushes the
positive and negative apart, and an anchor penalty to the frozen retrieval
vector.  Train for two epochs with no checkpoint or hyperparameter selection.

On the held-out inner fold, report two predeclared diagnostics:

1. mean cosine of its independently mined hard positive-negative pairs before
   and after adaptation; and
2. read AUC of the trained scalar head versus an L2 logistic regression on the
   frozen retrieval vectors.

Only if both diagnostics improve does a separate, fresh nested-CV classifier
experiment become justified.  The pilot neither chooses a cutoff nor produces
test predictions.
