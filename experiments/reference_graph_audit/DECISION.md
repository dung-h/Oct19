# Reference graph pilot decision

Collection/identity audit is complete for 314 selected records. Outgoing reference graphs
are technically obtainable, but this pilot has not established extra predictive value.
Defer model training until identity and broader fit-pool coverage are adequate.

The cohort is all 137 persistent incumbent false positives, 137 distinct true-positive
controls sharing outer fold and venue, and two random 20-paper samples (train and test).
133 matched pairs share publication year exactly; four differ by one year. This sample
is selected using known training outcomes and is not an unbiased generalization test.

Strict identity matching accepts 237/314 papers (75.5%); 185/314 (58.9%) have nonempty
outgoing-reference lists. Among 137 FP, 81 have references; among 137 matched TP, 80 do.
In the 62 pairs where both have references, TP has more references in 23 pairs, FP in 34,
and counts tie in 5. These descriptive results give no clear reason to use reference
availability or reference count as a preference classifier.

The verified pilot graph has 637 shared-reference edges involving 168 of the 314 records.
Edge meaning and label usefulness remain untested. The graph is restricted to the pilot;
it does not measure connectivity to the full current fit pool, and the selected pilot's
labels must not become a hand-picked reference bank.

Identity problems remain: 30 author conflicts, 11 ambiguous works/versions, four year
conflicts, and 32 records with no verified match from the bounded searches. Search failure
does not prove that a publication or its references do not exist. Two author conflicts
have been independently confirmed against publisher pages; see
`outputs/PRIMARY_METADATA_REVIEW.md`. The remaining conflicts are not assumed to be
dataset errors.

The useful next acquisition step is to resolve bibliographic identity systematically
and extend reference collection uniformly to the full train/test pool. Then a supervised
comparison can test bibliographic coupling to fit-only positive and negative history,
with missing-graph fallback to the incumbent. Compare against the same learner using
only verified metadata so that any gain can be attributed to the graph. Current-day
incoming citation counts should remain excluded. No preference model was trained and no
submission was generated in this pilot.
