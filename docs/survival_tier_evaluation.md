## Survival models: per-subgroup precision, recall and Macro F1

The AFT survival models predict days to disposition. To report the PRD's
classification metrics, predicted days are turned into Low / Medium / High
and scored against the true tier on the same held-out test split used for the
C-index (script: `scripts/survival_tier_report.py --state <state> --auto`).

**How tiers are made**
- Predicted tier: tertiles of predicted days (the same cut points the API uses).
- True tier: tertiles of observed days, calculated separately for each state.
- Pending cases: a pending case is scored only if it has already waited longer
  than the High cutoff, because only then is its tier certain. Other pending
  cases are left out.

**Overall results (test split)**

| State | Macro precision | Macro recall | Macro F1 | Cases scored |
|---|---|---|---|---|
| Delhi | 0.620 | 0.620 | 0.620 | 100% |
| Odisha | 0.545 | 0.527 | 0.531 | 88% |
| Bihar | 0.566 | 0.534 | 0.507 | 74% |

**Fairness gaps (best minus worst subgroup Macro F1, groups with n >= 1000)**

| State | Court tier | District | Case type |
|---|---|---|---|
| Delhi | 0.168 | 0.150 | 0.238 |
| Odisha | 0.109 | 0.296 | 0.261 |
| Bihar | 0.189 | 0.236 | 0.362 |

**Limits**
- No state reaches the PRD target of Macro F1 >= 0.75 (Delhi is closest at 0.62).
- All gaps are above the PRD target of < 0.10.
- Odisha and Bihar scores cover only cases whose tier is known. Bihar's Medium
  tier is small, so its score is the least reliable.
- Full per-subgroup tables: `reports/*_survival_tier_report_*_auto.csv`.
