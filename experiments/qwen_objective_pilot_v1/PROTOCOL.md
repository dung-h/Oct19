# Objective screening protocol, terminal v1

Status: CPU screen complete; no GPU quality fit was opened. The source config
and fit-only thresholds follow `../qwen_head_v1/OBJECTIVE_AUGMENTATION_PLAN.md`.
The final screening namespace is `outputs/objective_screen_v2`; v1 artifacts
are preserved because that run used deterministic, non-shuffled GroupKFold.

- Inputs: verified frozen hidden cache and committed fit-label caches from each
  partition of the previous Qwen ranking pilot.
- Per partition, shuffled `GroupKFold(3, random_state=20261103)`; groups stay
  disjoint. No prediction labels loaded.
- Compare CE, CE+RPS(alpha=1), and weighted CE with capped square-root inverse
  class weights. Five lambda values per arm; select lambda by pooled OOF CE,
  tie within 1e-6 toward stronger regularization.
- Weighted CE OOF probabilities are prior-corrected by dividing by class
  weights then renormalizing before CE/AUC/RPS evaluation.
- Challenger eligibility requires CE improvement >=.01 vs CE control, no read
  AUC decrease, and >=2/3 fold CE improvements. Ordinal also requires RPS not
  to increase. If both eligible select lower CE, tie favors ordinal.
- The rule runs separately in each partition. Both selectors returned CE, so
  stop before GPU. No held-out quality is calculated and no threshold is tuned.

Output files under ignored `outputs/objective_screen_v2/` include fold metrics,
lambda choices, eligibility, and `decision.json`. `verify.py` checks the RPS
contracts, group-disjoint status, five-class metrics, and terminal decision.
