# Interpretation and next action

The new fixed-selection diagnostic rules out reordering the current selected
set as a sufficient route to mean outer H=0.7: even the true-label oracle order
only reaches H=0.660783. This bound applies to this fixed OOF selected set, not
to other selection policies, future data, or Kaggle test performance.

The title-margin probabilities improve order in all five folds but only by
0.001004 mean H. Abstract-margin order reduces H by 0.000361. The earlier
fixed_gate_ranking study also showed a small ranking gain, so additional work
on this family of rerankers has a limited observed return.

Selection needs to improve for a substantially larger gain. Label-based
counterfactual removal of the 338 selected Label1 papers, while retaining the
current selected positives and their order, raises H to 0.716558. This uses
ground truth and is not an achievable model, a deployment policy, or evidence
that a threshold can accomplish the removal. Selecting all actual positives
with incumbent order gives an additional explanatory oracle H=0.755371.

Next recommended task: audit the incumbent's false positives across genuinely
different existing OOF models, separating persistent consensus errors from
disagreements. The output should identify how many errors another model can
detect, the accompanying loss of true positives/high labels, and which input
attributes could explain persistent errors. Inspecting labels is diagnostic;
do not encode the resulting case IDs or manually create preference labels.
Only if errors are complementary should a separately specified ensemble be
tested with weights chosen inside training folds. If the different models
share the same errors, prioritize new preference supervision or a verifiable
new input attribute before further model/threshold searches.

Binary gate, author history/calibration and semantic-role extraction have
already been tested in this project; repeating those generic suggestions
without a new discriminating signal would not address the evidence.

All results are exploratory on previously examined folds. Keep the incumbent;
no final refit, submission candidate, or Kaggle upload was performed here.
