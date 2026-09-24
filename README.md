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

Rows with valid target: 413855
New thresholds — Low <= 186, Medium <= 700, High > 700
Feature matrix shape: (413855, 173)

Macro F1 Score: 0.5931
              precision    recall  f1-score   support

         Low       0.63      0.72      0.67     27326
      Medium       0.54      0.38      0.45     27308
        High       0.62      0.70      0.66     28137

    accuracy                           0.60     82771
   macro avg       0.60      0.60      0.59     82771
weighted avg       0.60      0.60      0.59     82771


=== Subgroup Fairness Report: type_name_normalized ===
[arbtn] (n=4753): Macro F1 = 0.4063
[ca] (n=1541): Macro F1 = 0.3494
[civ dj] (n=1164): Macro F1 = 0.4444
[civ suit] (n=3612): Macro F1 = 0.5069
[cr case] (n=2187): Macro F1 = 0.3358
[cr cases] (n=3812): Macro F1 = 0.4126
[cr criminal revision] (n=718): Macro F1 = 0.3392
[cr rev] (n=2175): Macro F1 = 0.3396
[cs] (n=3059): Macro F1 = 0.4355
[cs dj] (n=3703): Macro F1 = 0.4393
[cs dj adj] (n=1130): Macro F1 = 0.4935
[cs scj] (n=5452): Macro F1 = 0.4608
[ct cases] (n=16350): Macro F1 = 0.5136
[e x] (n=67): Macro F1 = 0.2569
[ex] (n=5531): Macro F1 = 0.3719
[ex civil] (n=359): Macro F1 = 0.3313
[ex crl] (n=333): Macro F1 = 0.3795
[gp] (n=328): Macro F1 = 0.5597
[hindu adp] (n=142): Macro F1 = 0.6510
[hma] (n=5318): Macro F1 = 0.5064
[l i d] (n=539): Macro F1 = 0.4284
[l i r] (n=3383): Macro F1 = 0.4420
[lac] (n=300): Macro F1 = 0.3317
[lc] (n=485): Macro F1 = 0.4583
[lca] (n=415): Macro F1 = 0.4443
[m] (n=198): Macro F1 = 0.3537
[mact] (n=3700): Macro F1 = 0.5075
[mc] (n=193): Macro F1 = 0.4863
[mca dj] (n=267): Macro F1 = 0.4386
[mca scj] (n=181): Macro F1 = 0.4260
[misc crl] (n=501): Macro F1 = 0.4927
[misc dj] (n=1657): Macro F1 = 0.3437
[misc scj] (n=1034): Macro F1 = 0.4517
[mt case] (n=913): Macro F1 = 0.4634
[pc] (n=290): Macro F1 = 0.3786
[poit] (n=303): Macro F1 = 0.6053
[ppa] (n=171): Macro F1 = 0.3964
[rc arc] (n=972): Macro F1 = 0.4182
[rca civil dj adj] (n=50): Macro F1 = 0.3575
[rca dj] (n=931): Macro F1 = 0.4346
[rca scj] (n=92): Macro F1 = 0.3997
[rct arct] (n=276): Macro F1 = 0.4354
[sc] (n=2186): Macro F1 = 0.3091
[succ court] (n=605): Macro F1 = 0.4199
[succession court] (n=54): Macro F1 = 0.4656
[tm] (n=122): Macro F1 = 0.4476
[tp c] (n=201): Macro F1 = 0.4859
[tp crl] (n=253): Macro F1 = 0.4950
Max F1 Disparity Gap for type_name_normalized: 39.41%

=== Subgroup Fairness Report: court_tier ===
[Chief Metropolitan Magistrate] (n=21850): Macro F1 = 0.5738
[District and Sessions Judge] (n=34959): Macro F1 = 0.5283
[POLC and POIT] (n=4610): Macro F1 = 0.5397
[Principal Judge Family Court] (n=5090): Macro F1 = 0.5470
[Senior Civil Judge cum RC] (n=16262): Macro F1 = 0.4996
Max F1 Disparity Gap for court_tier: 7.42%

=== Subgroup Fairness Report: district_name ===
[Central] (n=14669): Macro F1 = 0.5628
[East] (n=5726): Macro F1 = 0.4685
[New Delhi] (n=4595): Macro F1 = 0.5565
[North] (n=5852): Macro F1 = 0.5562
[North East] (n=7639): Macro F1 = 0.5643
[North West] (n=8375): Macro F1 = 0.5606
[Shahdara] (n=3639): Macro F1 = 0.4563
[South] (n=5685): Macro F1 = 0.5611
[South East] (n=5521): Macro F1 = 0.5096
[South West] (n=12695): Macro F1 = 0.5750
[West] (n=8375): Macro F1 = 0.5430
Max F1 Disparity Gap for district_name: 11.87%
2026/09/24 10:24:37 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.