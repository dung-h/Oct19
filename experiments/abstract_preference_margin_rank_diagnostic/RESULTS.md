# Fixed-selection rank diagnostic for abstract preference margin

Post-hoc exploratory diagnostic on already examined OOF rows. No new fit, tuning, final-model selection, or Kaggle upload.

All prediction arms use exactly the incumbent selected set and their own expected rating gain for order. Counts are summed; H/Q/R are five-fold means. Oracles use labels and cannot be deployed.

| policy                                   | kind       |        H |        Q |        R |   delta_H |   improved_folds |   accepted |   FP |   FN |
|:-----------------------------------------|:-----------|---------:|---------:|---------:|----------:|-----------------:|-----------:|-----:|-----:|
| reference                                | prediction | 0.640276 | 0.532500 | 0.803238 |  0.000000 |                0 |       1616 |  338 |  313 |
| retrieval_margin                         | prediction | 0.640293 | 0.532542 | 0.803238 |  0.000017 |                2 |       1616 |  338 |  313 |
| title_margin                             | prediction | 0.641280 | 0.533877 | 0.803238 |  0.001004 |                5 |       1616 |  338 |  313 |
| abstract_margin                          | prediction | 0.639915 | 0.532013 | 0.803238 | -0.000361 |                2 |       1616 |  338 |  313 |
| oracle_order_fixed_selection             | oracle     | 0.660783 | 0.561398 | 0.803238 |  0.020507 |                5 |       1616 |  338 |  313 |
| oracle_remove_selected_FP_existing_order | oracle     | 0.716558 | 0.646909 | 0.803238 |  0.076282 |                5 |       1278 |    0 |  313 |
| oracle_positive_selection_existing_order | oracle     | 0.755371 | 0.606957 | 1.000000 |  0.115095 |                5 |       1591 |    0 |    0 |

Oracle order is the maximum possible Q/H from reordering the current selected set. Oracle FP removal keeps the current true positives and order while eliminating known rating-1 items. Oracle positive selection includes all known rating>1 items with incumbent gain for order.

Reference metrics reproduced; selection invariants, independent ranking/metric calculations and source hashes passed.
