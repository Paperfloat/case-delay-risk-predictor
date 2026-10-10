## Calibration check of the survival tiers (all three states)

Script: `scripts/survival_calibration.py --state <state>`, run after the case-type spelling cleanup. It rebuilds the held-out test split used for the C-index (it stops if the C-index does not match the saved model), assigns each case a tier from its predicted days, and fits a Kaplan-Meier curve per tier so pending cases are handled correctly.

**Tier order: passes in all three states.** The share of cases decided within 1, 2, 3 and 5 years should fall from Low to Medium to High.

| State | Tier | Decided within 1 year | 2 years | 3 years | 5 years |
|---|---|---|---|---|---|
| Delhi | Low | 0.776 | 0.894 | 0.939 | 0.974 |
| Delhi | Medium | 0.388 | 0.661 | 0.797 | 0.911 |
| Delhi | High | 0.134 | 0.250 | 0.350 | 0.559 |
| Odisha | Low | 0.174 | 0.327 | 0.475 | 0.805 |
| Odisha | Medium | 0.027 | 0.092 | 0.188 | 0.429 |
| Odisha | High | 0.007 | 0.024 | 0.054 | 0.177 |
| Bihar | Low | 0.285 | 0.380 | 0.466 | 0.626 |
| Bihar | Medium | 0.025 | 0.067 | 0.121 | 0.256 |
| Bihar | High | 0.003 | 0.012 | 0.024 | 0.059 |

**Predicted day counts: not calibrated.** Median predicted days against the Kaplan-Meier median:

| State | Low | Medium | High |
|---|---|---|---|
| Delhi | 89 vs 118 | 411 vs 488 | 1,171 vs 1,616 |
| Odisha | 1,137 vs 1,140 | 2,378 vs 2,066 | 4,924 vs not reached |
| Bihar | 1,533 vs 1,240 | 3,640 vs 3,062 | 11,450 vs not reached |

In Odisha and Bihar fewer than half of High-tier cases were decided by the data cutoff, so no observed median exists and the predicted High day counts are extrapolations beyond the data (Bihar's is about 31 years).

**Conclusion.** Use the tier and the ranking. Do not present the predicted day count as an estimate, especially for the High tier in Odisha and Bihar.

**Limits.** One train/test split per state. Tiers are tertiles of predicted days, so the groups have different case mixes. Odisha and Bihar durations for 2010-2012 filings may be biased long (see bihar_pending_investigation.md). Full numbers: `reports/*_survival_calibration_*.csv`.
