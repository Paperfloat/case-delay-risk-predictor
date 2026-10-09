## Cross-state check (v2)

Script: `scripts/experiments/cross_state_v2.py` (results: `experiments/cross_state_v2.csv`). Same features, split (seed 42)
and settings as `cross_state.py`; district is excluded because district names are state-specific. 95% intervals are
from 60 bootstrap draws. Test C-index:

| Setup | Delhi | Odisha | Bihar |
|---|---|---|---|
| In-state | 0.7258 | 0.7043 | 0.7484 |
| Pooled + state input | 0.7258 | 0.7043 | 0.7482 |
| Pooled, no state input | 0.7247 | 0.6973 | 0.7469 |
| Hold-out (train on the other two) | 0.5888 | 0.5124 | 0.6047 |
| In-state, no case type | 0.6498 | 0.6190 | 0.7045 |
| Hold-out, no case type | 0.6044 | 0.5126 | 0.5411 |

**Reading.** A pooled model with a `state` input matches the per-state models to four decimals because trees can split
on state first, so this says nothing about shared structure. Without the state input the loss is small (0.001 Delhi,
0.002 Bihar, 0.007 Odisha), but court tiers and case-type labels differ by state, so the features still reveal the
state and the model can learn state-specific rules; this does not show shared structure either. The informative test is
the hold-out: 0.51-0.60, with Odisha at chance. With case type removed (the least comparable feature), the hold-out is
still 0.045 (Delhi), 0.106 (Odisha) and 0.163 (Bihar) below the in-state score, with non-overlapping intervals.
Conclusion: knowledge does not transfer between states with these features, so per-state models stay.

**Limits.** One split. Court-tier names and rules are still state-specific, so "no case type" is not a fully comparable
feature set. In-state numbers here are lower than the saved models' C-index (0.7508 / 0.7490 / 0.7969) because district
is excluded. The bootstrap resamples test cases only.
