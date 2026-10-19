# V3 pilot result — passed

The V3 pilot ran both `query_only` and `contextual` on `outer3_inner3`, the same
partition where V2 `query_only` produced a 13-paper ReLU-dead cluster. V3 changes
only the head activation from ReLU to `LeakyReLU(0.01)`; its V2/V3 initial head state
hash was asserted equal before each arm began training.

| Check | Query-only | Contextual |
| --- | ---: | ---: |
| Epoch loss | 1.80537 → 1.62796 | 1.92052 → 1.67007 |
| Unique probability vectors / 498 | 497 | 498 |
| Largest exactly identical output group | 2 | 1 |
| Read-score standard deviation | 0.24341 | 0.26467 |
| Former 13 rows: nonzero individual gradient | 13/13 | 13/13 |
| Former 13 rows: unique output vectors | 13 | 13 |
| Save/reload maximum probability difference | 0 | 0 |

For the former 13-row cohort, the individual gradient maximum absolute values ranged
from 0.20143 to 1.59522 in query-only, and from 0.05633 to 0.42580 in contextual.
Their maximum first-layer preactivations were positive after training (ranges 2.96–4.96
and 2.60–6.23 respectively), whereas the V2 diagnostic found every one negative.

This validates that the specific dead-ReLU failure is absent in this controlled pilot.
It does not estimate H or select a threshold: that requires the predeclared full V3
nested CV in a fresh cache namespace.
