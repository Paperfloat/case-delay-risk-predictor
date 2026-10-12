## Calibration check of the survival tiers (all three states)

Script: `scripts/survival_calibration.py --state <state>`, run on the models retrained after the case-type spelling cleanup and the party-field change. It rebuilds the held-out test split used for the C-index (it stops if the C-index does not match the saved model), assigns each case a tier from its predicted days, and fits a Kaplan-Meier curve per tier so pending cases are handled correctly.

**Tier order: passes in all three states.** The share of cases decided within 1, 2, 3 and 5 years should fall from Low to Medium to High.

| State | Tier | Decided within 1 year | 2 years | 3 years | 5 years |
|---|---|---|---|---|---|
| Delhi | Low | 0.777 | 0.894 | 0.939 | 0.974 |
| Delhi | Medium | 0.388 | 0.660 | 0.794 | 0.908 |
| Delhi | High | 0.136 | 0.253 | 0.353 | 0.561 |
| Odisha | Low | 0.173 | 0.321 | 0.469 | 0.802 |
| Odisha | Medium | 0.027 | 0.093 | 0.186 | 0.424 |
| Odisha | High | 0.008 | 0.027 | 0.059 | 0.182 |
| Bihar | Low | 0.285 | 0.379 | 0.465 | 0.622 |
| Bihar | Medium | 0.025 | 0.068 | 0.121 | 0.257 |
| Bihar | High | 0.003 | 0.012 | 0.025 | 0.062 |

**Predicted day counts: not calibrated.** Median predicted days against the Kaplan-Meier median:

| State | Low | Medium | High |
|---|---|---|---|
| Delhi | 87 vs 118 | 419 vs 489 | 1,117 vs 1,610 |
| Odisha | 1,138 vs 1,152 | 2,367 vs 2,077 | 4,772 vs not reached |
| Bihar | 1,568 vs 1,250 | 3,635 vs 3,085 | 9,702 vs not reached |

In Odisha and Bihar fewer than half of High-tier cases were decided by the data cutoff, so no observed median exists and the predicted High day counts are extrapolations beyond the data (Bihar's is about 27 years).

**Conclusion.** Use the tier and the ranking, and the history of similar cases shown in the app. Do not present the predicted day count as an estimate, especially for the High tier in Odisha and Bihar.

**Limits.** One train/test split per state. Tiers are tertiles of predicted days, so the groups have different case mixes. Odisha and Bihar durations for 2010-2012 filings may be biased long (see bihar_pending_investigation.md). Full numbers: `reports/*_survival_calibration_*.csv`.
