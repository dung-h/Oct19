# Qwen learning check v1 — fixed before execution

Question: should we add training, change the output head, or investigate a
different representation/input? This is an exploratory fit-internal diagnostic,
not a promotion or final-data run.

Use only the 1,494 fit documents of outer1_inner0. Split their publication groups
once with GroupShuffleSplit(test_size=.2, random_state=20261030). Existing
outer1_inner0 held-out indices are excluded. Do not reuse either trained pilot
adapter: they have already seen labels of this internal validation set.

Lock membership, source/code/config hashes, package versions, driver and allocator
environment before GPU work. Keep the original pilot files unchanged. Run a fresh
technical smoke; require LoRA updates, frozen base and save/reload parity<=1e-4.

Fixed controls, with five output classes and identical capped title/abstract:

1. Constant inner-train class prior.
2. Frozen Qwen five digit logits.
3. Frozen digit logits +five biases, fitted only on inner train; mean CE +
   .5*1e-4*||bias||², L-BFGS. No trained in-sample Qwen stacker is used.
4. Frozen Qwen final assistant-start hidden state +StandardScaler fitted only
   on inner train +multinomial LogisticRegression(C=1, max_iter=2000).
5. Same hidden-state classifier with one fixed permutation of train labels.
6. Word TF-IDF(1,2), min_df2, sublinear_tf, max_features20000 +the same C=1
   multinomial classifier, using exactly the capped title/abstract text.
7. Fresh q/v LoRA r8, alpha16, dropout.05, AdamW1e-4, wd.01, accumulation16,
   clipping1, BF16 NF4, trained for exactly three epochs. Single linear schedule
   over all three epochs, warmup5%; save predictions at epochs1,2,3. This is a
   fixed three-epoch trajectory, not an exact replay of the original one-epoch
   LR schedule. Shuffle seed+epoch. Do not early-stop or change the schedule.

Extract frozen features once. No feature scaling, vocabulary fitting, classifier
fitting or bias estimation uses internal validation labels. Produce predictions
after each epoch, but join validation labels and evaluate quality only after all
arms and all three epochs finish. Fixed train probes: 128 random inner-train
indices, selected without labels before fitting. Evaluate the same probes for
all arms. Report CE, read AUC/Brier, H/Q/R and rating4/5 misses. Cutoff stays .55.

Primary comparisons:

- epoch3 vs epoch1: extra training is promising if validation CE falls>=.03
  and read AUC does not fall; compare train-probe CE to distinguish fitting
  improvement from validation deterioration. Retain epoch2 as a learning curve.
- frozen hidden head vs epoch3: a CE advantage>=.03 and AUC advantage>=.02
  supports an output/learning-method bottleneck over base model replacement.
- hidden head vs shuffled labels and TF-IDF: determine whether the frozen
  representation contains useful preference signal. Failure of one linear
  head alone cannot establish that the entire Qwen model lacks capacity.
- bias-only vs LoRA: quantify how much is global digit-prior correction.

Report paired 500 publication-group bootstrap intervals for CE/AUC differences
as descriptive uncertainty, not proof of new independent validation. Do not pick
a new model, LR, loss, prompt, cutoff or epoch budget after seeing these results.
Any quality follow-up needs a separate locked protocol. No final2494/test/Kaggle.

Clear unused CUDA allocator cache between scoring calls and optimizer updates
to share VRAM. Use the previously verified expandable-segments allocator and
record it explicitly. Do not overwrite or resume interrupted training; preserve
failed artifacts and use a new namespace with another smoke.
