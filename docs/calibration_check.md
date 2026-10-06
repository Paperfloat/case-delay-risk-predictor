## Calibration check of the survival tiers (all three states)

Script: `scripts/survival_calibration.py --state <state>`. It rebuilds the held-out test split used for the C-index
(it stops if the C-index does not match the saved model), assigns each case a tier from its predicted days, and fits a
Kaplan-Meier curve per tier so pending cases are handled correctly.

**Tier order: passes in all three states.** The share of cases decided within 1, 2, 3 and 5 years falls from Low to
Medium to High at every horizon.

| State | Tier | Decided within 1 year | 2 years | 3 years | 5 years |
|---|---|---|---|---|---|
| Delhi | Low | 0.776 | 0.894 | 0.940 | 0.974 |
| Delhi | Medium | 0.389 | 0.661 | 0.796 | 0.910 |
| Delhi | High | 0.134 | 0.250 | 0.350 | 0.559 |
| Odisha | Low | 0.175 | 0.328 | 0.476 | 0.805 |
| Odisha | Medium | 0.027 | 0.091 | 0.187 | 0.429 |
| Odisha | High | 0.007 | 0.024 | 0.054 | 0.178 |
| Bihar | Low | 0.287 | 0.382 | 0.469 | 0.628 |
| Bihar | Medium | 0.023 | 0.065 | 0.119 | 0.255 |
| Bihar | High | 0.003 | 0.011 | 0.024 | 0.058 |

**Predicted day counts: not calibrated.** Median predicted days against the Kaplan-Meier median:

| State | Low | Medium | High |
|---|---|---|---|
| Delhi | 89 vs 119 | 411 vs 489 | 1,174 vs 1,615 |
| Odisha | 1,124 vs 1,139 | 2,361 vs 2,059 | 4,950 vs not reached |
| Bihar | 1,477 vs 1,230 | 3,666 vs 3,069 | 11,580 vs not reached |

In Odisha and Bihar fewer than half of High-tier cases were decided by the data cutoff, so no observed median exists
and the predicted High day counts are extrapolations beyond the data (Bihar's 11,580 days is about 32 years).

**Conclusion.** Use the tier and the ranking. Do not present the predicted day count as an estimate, especially for the
High tier in Odisha and Bihar.

**Limits.** One train/test split per state. Tiers are tertiles of predicted days, so the groups have different case
mixes. Full numbers: `reports/*_survival_calibration_*.csv`.
