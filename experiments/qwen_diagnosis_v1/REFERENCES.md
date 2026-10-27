# Additional descriptive references

The missing-abstract subgroup was inspected after pilot completion. Therefore
the missingness reference is explicitly post hoc, not a preregistered challenger.
It uses only each partition's fit labels: one five-class frequency vector for
abstract-present documents and one for abstract-missing documents, each with
add-one smoothing. It does not use in-sample Qwen predictions or fit a stacker.
Its held-out performance only helps quantify how much signal can be explained
by this simple availability indicator. No threshold or smoothing is tuned.

references.py also uses the exact probability factorization:

five-class CE = read binary CE + fraction_read * CE(rating conditional on read).

This describes read separation and ordinal discrimination separately while
preserving all five labels. It does not replace the competition H/Q/R metric.
