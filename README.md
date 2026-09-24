# case-delay-risk-predictor
A resource-planning tool that flags cases at risk of indefinite delay, built on public eCourts/NJDG data — with fairness and drift monitoring as first-class requirements, not afterthoughts.
## Target Definition & Known Limitations
- Target: days between date_of_filing and date_of_decision
- v1 scope: only cases with a recorded decision date are used for training
  (~11% of Delhi 2012 cases are still pending and excluded)
- Planned v2 improvement: treat pending cases as right-censored
  (survival analysis / classification-style risk bucketing) instead of dropping them
- 29 rows excluded due to corrupted year values in date fields (e.g. "1204" instead of a plausible year)
- 139 rows excluded due to negative days_to_disposition (decision recorded before filing)
- purpose_name decoded via key file join; unmatched/corrupted values imputed as "unknown" post-decode
- District names decoded using the 2010 district key mapping (state_code + dist_code only,
  without year), since cases_district_key.csv has no entry for Delhi in 2012+. District
  boundaries confirmed stable — all 11 dist_codes in the 2012 case data matched the 2010 key.
- Case-type taxonomy: normalized 114 raw categories to 96 by stripping punctuation/
  whitespace and merging one known duplicate (mact/m a c t). Some plural/singular
  variants (e.g. "cr case" vs "cr cases") likely remain as separate categories —
  acceptable for v1, revisit if model performance suggests case-type signal is
  fragmented across near-duplicate labels.
- Court tier derived from court_name's role prefix (5 tiers: District and Sessions
  Judge, Chief Metropolitan Magistrate, Senior Civil Judge cum RC, Principal Judge
  Family Court, POLC and POIT) — used for fairness subgroup monitoring per PRD.
- multiple_hearings proxy: the 5000-01-01 placeholder in date_last_list (330 rows)
  was initially causing all such rows to incorrectly show multiple_hearings=1.
  Fixed by treating the placeholder as missing; these 330 rows now correctly
  show multiple_hearings as unknown (NA) rather than a false positive.
- Model finding: purpose_name="unknown" (~2% of rows) strongly correlates with
  near-immediate case resolution (median 1 day vs. 388 days for cases with a
  recorded purpose). Likely explanation: purpose_name is only populated when a
  case requires a scheduled follow-up hearing — cases resolved immediately
  (allowed, withdrawn, compromised at first hearing) never accumulate one.
  Kept as a legitimate feature rather than excluded, since it reflects a real
  procedural pattern, not a data artifact.

## Fairness Findings (v1 baseline model)
- District: MAE ranges from 307 days (North East) to 535 days (Shahdara) — a ~74%
  relative gap. Correlation between district sample size and MAE is -0.37 (moderate),
  meaning sample-size imbalance partially but not fully explains this — some districts
  are genuinely harder to predict, not just underrepresented.
- Court tier: MAE ranges from ~323 days (Family Court, Sessions Judge) to ~454 days
  (POLC/POIT, Chief Metropolitan Magistrate) — a ~40% relative gap.
- Gender (female_defendant): modest gap (347 days female vs. 393 days male) — smaller
  than district/court-tier disparities. The "-9999 missing name" subgroup has only 5
  test rows and should not be treated as a meaningful estimate.
- Conclusion: real, moderate reliability disparities exist across districts and court
  tiers. Not disqualifying for v1, but flagged as a monitoring priority — a case from
  Shahdara or a POLC/POIT court currently gets a meaningfully less reliable delay
  prediction than one from North East or a Family Court.

## Example: Per-Case Explanation
Case 26-09-02-202100620012016 (Chief Metropolitan Magistrate, West district,
type "cr case", purpose "order"): actual duration 1,341 days, predicted 2,077
days. Model correctly identified this as a high-delay-risk case (1,341 days is
well above the 373-day median) but overestimated the magnitude by ~55% — an
expected outcome given R²=0.43, illustrating that the model has real directional
signal but substantial residual uncertainty in exact day counts. This supports
the plan to reframe as risk-tier classification rather than precise day
prediction for the eventual product.

Note: SHAP values exist for every one-hot encoded column, not just the "active"
category for a given row — only the column matching the case's actual attribute
(e.g. district_name_West=1) reflects that case's own contribution; other
category columns show the model's learned baseline shift from not being in
that category, not signal from this case.

## Week 3 Model Results (v1 Classifier)
- Macro F1: 0.598 — below PRD target of ≥0.75
- Case-type F1 disparity: 73.68% — far exceeds PRD target of <10%.
  Confirmed NOT a sample-size artifact (correlation between subgroup
  size and F1 ≈ -0.02) — this is a genuine, substantial fairness gap.
- Court tier disparity: 13.56%, district disparity: 13.14% — both
  modestly exceed the 10% target.
- Conclusion: v1 classifier, trained on only 6 filing-time categorical
  features from a single state-year, does not yet meet PRD success
  criteria. This is an expected, honest v1 result given the limited
  feature set and single year of data — not a bug. Next steps to close
  the gap: expand training data to Delhi 2010-2013 (previously planned),
  investigate whether additional filing-time features exist, and
  consider per-case-type calibration given the demonstrated disparity.
  - Case type "lac" shows an outlier F1 of 0.947, but this is a class-imbalance
  artifact, not genuine model skill: 94% of "lac" cases (808/858) fall in the
  High risk tier (median 1,876 days), so a trivial always-predict-High rule
  would score similarly. Excluded from the "genuine disparity" interpretation.

  ## Known Gap: CI Workflow Needs Cloud DVC Remote
.github/workflows/data_validation.yml currently references data via the local
DVC remote (/home/srish/dvc-storage/...), which GitHub Actions' runners cannot
access. Before this CI workflow can actually run, it needs a cloud-reachable
DVC remote (e.g. a free-tier S3/GCS bucket or DVC's own hosted storage) and a
dvc pull step added to the workflow. Deferred to Week 4 (CI/CD is explicit
Week 4 PRD scope) rather than fixed now.
