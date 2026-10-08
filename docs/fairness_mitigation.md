## Fairness mitigation experiment: reweighting did not close the gaps

Script: `scripts/fairness_mitigation.py --state <state>` (results: `reports/fairness_mitigation_*.json`). It trains the
baseline survival model and three variants on the same split, each weighting training cases by (group size)^-0.5 for
one dimension (court tier, district, or case type), so cases from small groups count more. Gaps are best-minus-worst
subgroup Macro F1 on tiers (groups with at least 1,000 scored test cases). "Noise" is the standard deviation of a gap
when the test cases are resampled (100 bootstrap draws); a change counts as improved or worse only if it is beyond
2 noise units.

**Baseline gaps (after the case-type spelling cleanup), with noise**

| State | Court tier | District | Case type |
|---|---|---|---|
| Delhi | 0.171 (noise 0.006) | 0.150 (0.010) | 0.248 (0.008) |
| Odisha | 0.093 (0.019) | 0.295 (0.018) | 0.170 (0.012) |
| Bihar | 0.185 (0.007) | 0.250 (0.029) | 0.418 (0.014) |

**Result.** Across 27 comparisons (3 variants x 3 gap types x 3 states), none improved and none got worse beyond noise.
Side effects were small: C-index within 0.0012 and Macro F1 within 0.0021 of the baseline.

**Reading.** Eight of the nine gaps are at least 4 noise units above the 0.10 target, so they are not sampling noise.
Odisha's court-tier gap (0.093) is within 0.4 noise units of 0.10, so it cannot be called met. Reweighting is not a
fix: the gaps are probably caused by groups that are harder to predict from filing-time features (not tested).

**Limits.** One weighting scheme (exponent 0.5). One train/test split per state. The noise measure covers test-case
resampling only, not variation from retraining. Not tried: separate models per group, per-group tier thresholds,
interaction features, or collecting more features. The mitigated models were not saved.
