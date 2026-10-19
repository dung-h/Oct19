# Cross-encoder preference V3 pilot

V2 diagnosed a genuine ReLU dead-region failure in `outer3_inner3_query_only`:
13/498 held-out papers had all 96 preactivations negative, identical bias-only
probabilities, and exactly zero gradient from their own loss to the normalized CLS
embedding. The 13 papers are retained only as a diagnostic cohort, not as a tuning
target.

V3 changes exactly one operation: `ReLU(x)` in `QueryHead` and `ContextHead` becomes
`LeakyReLU(x, negative_slope=0.01)`. All parameter shapes, initialization sequence,
SPECTER2 base/adapter, normalized CLS, text and token budget, reference rule, split,
two epochs, losses, optimizer, batch order, and AMP settings remain V2's.

The pilot is the V2 failed inner partition, `outer3_inner3`, for both arms. It must
show more than a nonzero Leaky activation: the former 13-row cohort must have finite,
nonzero gradients from each individual loss to its input CLS and output must depend on
query text. It also checks output variation, input permutation sensitivity, and
checkpoint save/reload parity. The report records any new large nearly-identical output
cluster without using labels to change the method.

Only a passed pilot permits a separate V3 nested-CV runner and fresh cache namespace.
The pilot does not choose a threshold and does not estimate H.
