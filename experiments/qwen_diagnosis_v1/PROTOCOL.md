# Qwen pilot diagnosis v1 — locked before fit probes

Purpose: distinguish implementation failure, learning only class frequencies,
weak read discrimination, weak ordinal discrimination, and probability calibration.
This is a diagnosis of the completed exploratory pilot, not a new promotion run.

Source: experiments/qwen_remote_v1/outputs/pilot_torch212_cu130.
Keep its code, config, adapters, predictions, and fingerprints unchanged.

CPU analysis:

- Validate both completed partitions and probability lineage first.
- Compare frozen/trained/incumbent with a constant classifier estimated from
  each partition's fit labels only. Report its held-out logloss and Brier.
- Describe read reliability using fixed bins [0,.2,.4,.55,.65,.8,1]. Do not
  search thresholds, fit temperatures, or select any checkpoint.
- Measure ordinal discrimination within ratings 2–5, positive-class mixture,
  probability variance, gain/read correlation, and missing-abstract subgroups.
- Decompose the already locked native/matched-budget policy transitions.

GPU fit probes (no optimizer step):

- Independently load each completed adapter on the pinned base and same stack.
- First reproduce the five saved fit reload probes, tolerance 1e-4.
- Use 128 fit indices per partition sampled without labels, seed 20261028.
  Never include that adapter's held-out indices.
- Compare frozen and trained probabilities on these same full prompts.
- On the first 32 sampled indices additionally remove title or abstract.
  Also score a single empty-content prompt. These are sensitivity diagnostics;
  blank prompts are out of distribution and cannot prove causal attribution.
- Compare aligned fit predictions with 200 fixed permutations of their labels.
  This describes content association, not independent OOF evidence.
- Backpropagate one fit example through each trained adapter to audit gradient
  connectivity and frozen base parameters; do not call optimizer.step or save
  a modified adapter.
- Read adapter A/B norms and verify checkpoint hashes. Log loading/scoring
  timings separately and release unused CUDA allocator cache between examples
  to share memory with the existing user job. Audit parity with saved probes.

No new model, epoch, prompt, loss, cutoff, or quality gate is selected here.
Any subsequent corrective quality experiment needs its own locked protocol;
calibration/fusion training needs clean cross-fitted lineage.

Runtime amendment after the first probe attempt: fit_probe_v1 failed while
PEFT temporarily upcast the large frozen embedding/head to FP32; another
user process held 5.07 GiB. No diagnostic predictions were produced. Keep
that failure and use a new namespace with PYTORCH_ALLOC_CONF set to
expandable_segments:True. Run the original fit-only technical smoke first,
then require parity against each completed pilot's saved fit probes. Record
the allocator environment explicitly; the original runtime lock omits it.
