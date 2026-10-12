## Survival models: per-subgroup precision, recall and Macro F1

The AFT survival models predict days to disposition. To report the PRD's classification metrics, predicted days are turned into Low / Medium / High and scored against the true tier on the same held-out test split used for the C-index (script: `scripts/survival_tier_report.py --state <state> --auto`). Numbers are from the models retrained after the party-field change.

**How tiers are made**
- Predicted tier: tertiles of predicted days (the same cut points the API uses).
- True tier: tertiles of observed days, calculated separately for each state.
- Pending cases: a pending case is scored only if it has already waited longer than the High cutoff, because only then is its tier certain. Other pending cases are left out.

**Overall results (test split)**

| State | Macro precision | Macro recall | Macro F1 | Cases scored |
|---|---|---|---|---|
| Delhi | 0.620 | 0.619 | 0.619 | 100% |
| Odisha | 0.539 | 0.520 | 0.524 | 88% |
| Bihar | 0.560 | 0.529 | 0.503 | 74% |

**Fairness gaps (best minus worst subgroup Macro F1, groups with n >= 1000)**

| State | Court tier | District | Case type |
|---|---|---|---|
| Delhi | 0.168 | 0.168 | 0.260 |
| Odisha | 0.076 | 0.290 | 0.192 |
| Bihar | 0.191 | 0.227 | 0.402 |

**Limits**
- No state reaches the PRD target of Macro F1 >= 0.75 (Delhi is closest).
- 8 of the nine gaps are above the PRD target of < 0.10. Odisha's court-tier gap (0.076) is below it on this split but only about 1.3 times the bootstrap noise (0.019, measured in the mitigation run), so it is not clearly met.
- Odisha and Bihar scores cover only cases whose tier is known. Bihar's Medium tier is small, so its score is the least reliable.
- Full per-subgroup tables: `reports/*_survival_tier_report_*_auto.csv`.
