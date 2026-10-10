# Case Delay Risk Predictor

Predicts how long an Indian court case will take to reach a decision, using only information known when the case is filed, for three states: **Delhi, Odisha and Bihar**. It is an end-to-end MLOps project: versioned data and models (DVC), config-driven pipelines, data validation, per-group fairness audits with confidence intervals, a multi-state API, Docker, and CI.

The data is the public Development Data Lab (DDL) e-Courts dataset: about 1.75 million cases filed in 2010-2013. Cases with no recorded decision are treated as still pending at a data cutoff date (right-censored), so the model is a **survival model** (XGBoost AFT), not a classifier on decided cases only.

**Not a verdict:** this tool estimates how long a case is likely to take, for resource-planning purposes only; it does not predict how any case will be decided and must not be used to judge the merits of a case or the people involved.

## Results

| State | Cases | Pending at cutoff | C-index, random split | C-index, time split (train 2010-12 filings, test 2013 filings) |
|---|---|---|---|---|
| Delhi | 459,712 | 10.4% | 0.7508 | 0.7304 |
| Odisha | 470,869 | 39.2% | 0.7490 | 0.7275 |
| Bihar | 816,809 | 55.8% | 0.7969 | 0.8008 |

Time-split values were re-measured after the case-type spelling cleanup, with the same model settings (baseline of `scripts/acts_experiment.py`).

The C-index measures how well the model **ranks** cases by duration (0.5 is random, 1.0 is perfect). It is not comparable across states, because each state has a different censoring rate and case mix. The time-split column is the more realistic estimate, since the real task is predicting new filings from older ones.

What the experiments showed:

- **Model settings barely matter.** Training to convergence at a higher learning rate gained +0.002 to +0.005; tree depth and the AFT error scale added nothing further.
- **States don't share a model.** A pooled model with a `state` input matches the per-state models to four decimals only because trees can split on state first. Without that input it loses 0.001 to 0.007, but court tiers and case-type labels differ by state, so the features still reveal the state. The informative test is the hold-out: a model trained on the other two states scores 0.51-0.60, with Odisha at chance. See [docs/cross_state_check.md](docs/cross_state_check.md).
- **Court workload looked like a win and wasn't.** Adding the number of filings in the same court in the prior 90 days raised the C-index by 0.010-0.019 on a random split, but lowered it by 0.005-0.019 on the time split in all three states. It was rejected (the training script keeps it behind an off-by-default `--workload` flag).
- **The act and section help in Delhi, not reliably elsewhere.** On Delhi cases with both an act and a section, adding them raised the time-split C-index from 0.7461 to 0.7891 (+0.043). In the same clean test Odisha gained 0.010 and Bihar lost 0.015 on the time split. Whether a case has act information depends on its outcome in these records, so the API uses the optional Delhi model only when both fields are supplied ([docs/acts_sections_experiment.md](docs/acts_sections_experiment.md)).
- **Civil suits are the hardest to rank.** The Civil Judge (Senior Division) tier has the lowest C-index in all three states, with confidence intervals that don't overlap the best tier. See [docs/fairness_case_study.md](docs/fairness_case_study.md).

## Architecture

```mermaid
flowchart LR
    RAW["DDL court records<br/>2010-2013 filings + key tables"] --> LOAD["load<br/>filter by state"]
    LOAD --> FEAT["features<br/>key joins, dates, event flag"]
    FEAT --> NORM["normalize<br/>court tiers, top-100 case types"]
    CFG["config/states.yaml<br/>state codes, cutoffs,<br/>tier rules, validation bounds"] -.-> LOAD
    CFG -.-> FEAT
    CFG -.-> NORM
    NORM --> VAL["Great Expectations<br/>14 checks per state, run in CI"]
    NORM --> TRAIN["train<br/>XGBoost AFT, early stopping"]
    TRAIN --> MODEL["model + metadata<br/>per state"]
    TRAIN --> REPORT["fairness report<br/>C-index by group, 95% CIs"]
    TRAIN --> MLF["MLflow run"]
    MODEL --> REG["MLflow Model Registry<br/>alias champion"]
    MODEL --> TIERS["tier metrics<br/>precision, recall, Macro F1"]
    NORM --> MON["survival_monitor.py<br/>PSI drift + fairness dashboard"]
    MON --> TRIG["retrain.yml<br/>monthly or on demand"]
    TRIG -.->|"retrain drifted states, open pull request"| TRAIN
    MODEL --> API["FastAPI<br/>predict, options, states, health"]
    API --> LOG[("SQLite audit log")]
    API --> DOCKER["Docker image"]
    DOCKER --> SMOKE["CI: build and smoke test<br/>all three states"]
    SMOKE -->|"push to main"| GHCR["GitHub Container Registry"]
    GHCR --> RENDER["Render, free plan<br/>live API"]
    RENDER --> LIVE["CD: smoke test<br/>the live service"]
```

Data, models and reports are versioned with DVC (remote on DagsHub). Every stage is defined once in `dvc.yaml` and runs per state from `config/states.yaml`, so adding a state is a config change plus a court-name check. See [docs/architecture.md](docs/architecture.md).

## Quickstart

Pull the processed datasets and models (needs DagsHub credentials in `.dvc/config.local`):

```
pip install -r requirements.txt
dvc pull
```

Rebuilding from raw data also needs the DDL national case files and key tables downloaded into `data/raw/cases/` and `data/raw/keys/` (they are not stored in the DVC remote). Then:

```
dvc repro
```

Run the API locally, or in Docker (after `dvc pull`, because the image copies `models/`):

```
pip install -r requirements-api.txt
uvicorn api.main:app --port 8000

docker build -f Dockerfile.api -t case-delay-api .
docker run -p 8000:8000 case-delay-api
```

Interactive docs are at `http://localhost:8000/docs`. `GET /options/{state}` lists the valid values for each input field. A request:

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "state": "delhi",
  "type_name_normalized": "hma",
  "court_tier": "Family Court",
  "district_name": "North West",
  "female_defendant": "0 male",
  "female_petitioner": "1 female",
  "female_adv_def": "0 male",
  "female_adv_pet": "0 male"
}'
```

The response contains the predicted days, a Low/Medium/High tier, the top five contributing factors (TreeSHAP, on a log-days scale), the model version, and warnings for any input value the model has not seen. Every prediction is written to an audit table with its state and model version.

## Repository layout

```
config/states.yaml          state codes, cutoffs, tier rules, validation bounds
dvc.yaml, dvc.lock          pipeline: load, features, normalize, train (per state)
scripts/pipeline/           pipeline steps and the cutoff estimator
scripts/train_survival.py   training, bootstrap fairness audit, model metadata
scripts/experiments/        tuning, cross-state and court-workload experiments
validation/                 Great Expectations checks (one script, config-driven)
api/main.py                 multi-state survival API; api/legacy_classifier.py is the v1 classifier API
monitoring/                 drift and fairness dashboards for the v1 classifier
reports/                    per-state fairness reports with confidence intervals
.github/workflows/          data validation matrix, Docker build and smoke test
docs/                       fairness case study, architecture, PRD v2 addendum, demo script
```

## Tier-level results and fairness

The survival models predict days. To report precision, recall and Macro F1 (the PRD metrics), predicted days are turned into Low / Medium / High tertiles within each state and scored per subgroup on the held-out test split (`scripts/survival_tier_report.py --state <state> --auto`). Details and limits: [docs/survival_tier_evaluation.md](docs/survival_tier_evaluation.md).

| State | Macro precision | Macro recall | Macro F1 | Cases scored |
|---|---|---|---|---|
| Delhi | 0.621 | 0.621 | 0.621 | 100% |
| Odisha | 0.542 | 0.524 | 0.528 | 88% |
| Bihar | 0.565 | 0.534 | 0.506 | 74% |

Best-minus-worst subgroup Macro F1 (groups with at least 1,000 test cases): court tier 0.093-0.185, district 0.150-0.295, case type 0.170-0.418. **Eight of the nine gaps are above the PRD target of 0.10 (Odisha's court-tier gap, 0.093, is within about half a noise unit of it), and no state reaches the 0.75 Macro F1 target.** Pending cases are scored only when they have already waited past the High cutoff, so Odisha and Bihar cover only the cases whose tier is known.

## Monitoring, retraining and deployment

**Drift and fairness monitoring.** `monitoring/survival_monitor.py` compares cases filed up to 2011 with later filings (PSI, threshold 0.10) and writes `monitoring/survival_dashboard.html`, per-state Evidently drift reports and `monitoring/survival_drift_status.json`. Current result: Delhi is stable (largest PSI 0.091); Odisha (0.169) and Bihar (0.177) drift on case type, so a retrain is triggered.

**Automatic retrain.** `.github/workflows/retrain.yml` runs monthly and on demand. It runs the drift check, retrains only the drifted states, pushes the models to DagsHub and opens a pull request. Merging that pull request starts the deploy pipeline.

**Simulated time-sliced retrain (Odisha).** A model trained on filings up to 2011 scored C-index 0.720 and Macro F1 0.510 on newer cases; after the drift trigger and a retrain it scored 0.750 and 0.544. One state, one seed, no no-drift control. See [docs/simulated_retrain.md](docs/simulated_retrain.md).

**Retrain-to-deploy time (target: under 15 minutes).** Training took 3m 21s (Delhi), 5m 18s (Odisha) and 12m 38s (Bihar), plus 4m 44s for the build and deploy run. Delhi (8m 05s) and Odisha (10m 02s) meet the target; Bihar (17m 22s) does not. Only the training stage was timed. See [docs/retrain_benchmark.md](docs/retrain_benchmark.md).

**Deployment.** CI builds the Docker image and smoke-tests all three states. On every push to `main`, a deploy job pushes the image to GitHub Container Registry, triggers Render, and smoke-tests the live service at https://case-delay-api.onrender.com (`/health` shows the loaded states). On Render's free plan the service sleeps when idle, so the first request can take about a minute.

**Model registry.** `python scripts/register_models.py` registers the three survival models in a local MLflow Model Registry with the alias `champion` (view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Further checks

**Calibration.** The tier order is correct in all three states: the share of cases decided within 1, 2, 3 and 5 years falls from Low to Medium to High. Predicted day counts are not calibrated (they differ from observed medians by roughly 15-28%, and High-tier days in Odisha and Bihar are extrapolations). Use the tier. See [docs/calibration_check.md](docs/calibration_check.md).

**Fairness mitigation.** Reweighting training cases by court tier, district or case type changed no gap beyond noise in 27 comparisons. See [docs/fairness_mitigation.md](docs/fairness_mitigation.md).

**Act and section (optional Delhi model).** Delhi only, used when a request supplies both fields; see [docs/acts_sections_experiment.md](docs/acts_sections_experiment.md) and the architecture notes.

**Pending share.** In Odisha and Bihar, decisions made before about 2013 look largely unrecorded, which would explain why older filings look more pending. This is not proven. See [docs/bihar_pending_investigation.md](docs/bihar_pending_investigation.md).

**Frontend.** `frontend/app.py` is a Streamlit app that reads the API address from `NYAYA_API_URL` (default `http://localhost:8000`), lists valid values from `/options/<state>`, shows the tier first and the day count in a collapsed section, and uses plain-English labels from `frontend/labels.py`. It is not part of the Docker image or CI.

## Limitations

- **Use the tier, not the day count.** The model is validated for ranking only. The predicted days were checked against Kaplan-Meier medians and are not calibrated (the tier order is correct in all three states; see [docs/calibration_check.md](docs/calibration_check.md)), and the Medium/High cut points for Odisha and Bihar lie at or beyond the longest duration the data can show (about 9 years), so they are extrapolations. Tiers are tertiles of predicted days within each state.
- **Features are filing-time only.** Case type, court tier, district and four gender fields (gender is inferred from names in the court records). Hearing history is not used. The act and section of a case are used only by an optional Delhi model, and only when both are supplied.
- **Pending status is an assumption.** Cases with no decision date are treated as pending at the cutoff. Their listing dates support this (over 99% have a next hearing date). In Odisha and Bihar newer filings have a lower pending share, the opposite of what pure censoring predicts; decisions made before about 2013 look largely unrecorded there, so durations for older filings may be biased long (not proven; see docs/bihar_pending_investigation.md).
- **Court tiers come from hand-written, per-state name rules,** so a tier name in one state is not exactly comparable to the same name in another.
- **Per-group results are single-split estimates.** The confidence intervals resample cases independently, so they understate the uncertainty from cases clustering within courts.
- **Monitoring is a snapshot comparison.** Drift compares early and later filings in one fixed dataset, so Odisha and Bihar flag drift on every run; a live system would compare training data with new cases. The fairness dashboard reports gaps but does not close them.
- **Tier metrics miss the PRD targets.** Best overall Macro F1 is 0.62 (Delhi) against a 0.75 target, and eight of the nine subgroup gaps are above 0.10 (Odisha's court tier, 0.093, is within noise of the target).
- **Fairness mitigation did not help.** Reweighting by court tier, district or case type changed no gap beyond noise; other approaches were not tried.
- **Predicted day counts are not calibrated.** The tier order is right, but predicted days differ from observed medians by roughly 15-28%, and High-tier days in Odisha and Bihar are extrapolations.
- **The act and section model is Delhi-only and optional.** It is not in the monthly retrain workflow, the drift monitor or the model registry, and the frontend lists act and section as stored in the records.
- **Early decisions look missing in Odisha and Bihar.** Durations for 2010-2012 filings may be biased long (not tested); see docs/bihar_pending_investigation.md.
- **Retrain-to-deploy misses 15 minutes for Bihar** (17m 22s), and only the training stage was timed.
- **The live demo is limited by the free host.** The service sleeps when idle, and the audit log lives inside the container, so it resets on every deploy or restart.

---

# Development log (chronological)
]633;E;cat README_top.md;82d24239-f3d4-4c80-8d79-03d29aec86a2]633;CA resource-planning tool that flags cases at risk of indefinite delay, built on public eCourts/NJDG data — with fairness and drift monitoring as first-class requirements, not afterthoughts.
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

## Week 3.2: Feature Addition Experiment (filing_month)
Added filing_month (from date_of_filing, genuinely available at filing time,
no leakage) to test whether seasonality carries signal.

Results: Macro F1 0.593 -> 0.593 (no change). Case-type gap 39.41% -> 35.63%
(marginal). Court tier gap 7.42% -> 7.70% (flat, still under target).
District gap 11.87% -> 13.13% (slightly worse).

Conclusion: two independent attempts to close the Macro F1 gap — 4x more
training data (Week 3.1) and a new filing-time feature (this experiment) —
both failed to move overall performance meaningfully. This is strong,
repeated evidence that the ceiling on this task, given only filing-time
categorical signal (case type, purpose, court, district, gender, month),
is around Macro F1 ~0.59, not the PRD's 0.75 target. Reaching 0.75 likely
requires either features with genuine leakage risk (early case-progress
signal, deliberately excluded in v1 for correctness) or a different
modeling approach (survival analysis, per-case-type threshold calibration).
Documented as the v1 conclusion; not pursued further this session.
## Week 5: Drift Monitoring (Evidently AI)
Simulated a time-sliced drift check using the same 3 fields tracked for
fairness (case type, court tier, district) — reference period: 2010-2011
(183,296 rows), current period: 2012-2013 (231,296 rows).

Results (PSI, threshold 0.1):
- type_name_normalized: 0.110 — DRIFT DETECTED
- court_tier: 0.031 — stable
- district_name: 0.039 — stable

Retrain trigger: fired, correctly isolating case-type distribution as the
source of drift.

This result directly validates a decision already made earlier in this
project: expanding training data from Delhi-2012-only to Delhi 2010-2013
(documented in Week 3.1) was the right call, not just a Macro F1
experiment — a model trained only on 2010-2011 case-type distributions
would have been measurably stale against 2012-2013 data, exactly the kind
of drift this monitor is designed to catch. The retrain that already
happened this session is what a real production system would have done
automatically in response to this exact trigger.

Full interactive report: monitoring/drift_report.html
## Week 5 (cont.): Evidently Fairness Dashboard
Added an Evidently ClassificationPreset report (monitoring/fairness_report.html)
using the same test split as the final model, with type_name_normalized,
court_tier, and district_name included as extra columns — Evidently
automatically breaks down classification quality by these segments,
complementing the custom sklearn-based subgroup audit from Week 3 with
a genuine dashboard artifact.

## Odisha (state code 11): survival model

- **Data:** DDL 2010-2013 filings, 470,869 cases, 181,131 (38.5%) with no decision date.
- **Censoring:** cases with no decision date are treated as still pending at the data cutoff. Checked: 99.7% have last and next listing dates, and only 0.5% were last listed more than 3 years before the cutoff, so they look actively pending rather than stale.
- **Cutoff:** 2019-02-28, taken from where monthly decision volume drops (1,620 to 235). Sensitivity (measured with the earlier training settings): 2020-12-31 gave C-index 0.7400 vs 0.7435.
- **Model:** XGBoost AFT (`survival:aft`, normal, scale 1.2), filing-time features only (case type, court tier, district, litigant/advocate gender fields).
- **Result:** test C-index 0.7490 (random split) and 0.7275 on the time split (train 2010-12 filings, test 2013 filings). Best-worst C-index gap among groups with at least 1,000 test cases: court tier 0.099 (Judicial Magistrate First Class 0.788 vs Civil Judge (Senior Division) 0.689), district 0.136 (Nabarangpur 0.803 vs Kendrapada 0.667), case type 0.116 (other 0.731 vs mac case 0.615). On Low/Medium/High tiers (tertiles within the state) Macro F1 is 0.528 overall (see Tier-level results). The C-index is not comparable across states or with the Delhi v1 classifier's Macro F1.
- **Limitations:**
  - Absolute predicted durations are not calibrated; use the model for ranking cases.
  - Pending rate differs strongly by district (about 5% in Gajapati to about 66% in Jharsuguda) and is flat across filing years. It is unclear whether this reflects real backlog or district-level recording differences.
  - Mass-disposal dates (e.g. 2014-12-06, 2015-12-12) cluster many decisions on single days.
  - Delhi now also has a survival model (below); its original classifier, which drops pending rows, stays as the v1 baseline.
  - Case-type spelling variants are now merged before the top-100 selection (for example `uc` with `uc case`). Per-group C-index also depends on each group's censoring rate, so gaps are not a direct fairness measure.

## Bihar (state code 08): survival model

- **Data:** DDL 2010-2013 filings, 816,809 cases, 54.5% with no decision date (61.3% of 2010 filings falling to 50.4% of 2013 filings).
- **Censoring:** cases with no decision date are treated as still pending at the cutoff. Checked: 99.3% of pending cases have a next listing date, and only 0.9% were last listed more than 3 years before the cutoff.
- **Cutoff:** 2019-05-31, estimated as the end of the month of the 90th percentile of pending cases' last listing date. The same rule gives 2019-02-28 for Odisha, matching the value found from its decision-volume drop.
- **Model:** XGBoost AFT, same setup and filing-time features as Odisha (learning rate 0.10, early stopping; best iteration 1553).
- **Result:** test C-index 0.7969 (random split) and 0.8008 on the time split (train 2010-12 filings, test 2013 filings). Best-worst C-index gap among groups with at least 1,000 test cases: court tier 0.194 (District and Sessions Judge 0.853 vs Civil Judge (Senior Division) 0.659), district 0.211 (Patna 0.828 vs Vaishali 0.617), case type 0.273 (other 0.842 vs regular bail 0.569). On Low/Medium/High tiers (tertiles within the state) Macro F1 is 0.506 overall (see Tier-level results). The C-index is not comparable across states or with the Delhi v1 classifier's Macro F1.
- **Limitations:**
  - The pending share falls for newer filing years, the opposite of what pure right-censoring predicts. Early decisions look largely unrecorded in Odisha and Bihar (see docs/bihar_pending_investigation.md; not proven). Vaishali is 94% pending and Gaya 97%, so districts like these are unreliable.
  - Court-tier names are matched by a Bihar-specific rule set. `Civil Judge (division unclear)` merges senior and junior civil courts (4.2% of cases), 2.2% of cases stay unclassified (e.g. `Criminal Proceeding`, `JJPDJ`), and 63% of cases fall in Chief Judicial Magistrate courts.
  - The C-index depends on each state's censoring rate and case mix, so it is not comparable across states or with the Delhi Macro F1. Part of Bihar's headline value comes from separating court tiers.

### Training-length experiment (Odisha, Bihar)

The DVC training stages use learning rate 0.10 with early stopping (best iteration 1531 for Odisha, 1553 for Bihar, 1113 for Delhi). An earlier sweep (`scripts/experiments/tune_survival.py`) found that training to convergence at learning rate 0.10 beat learning rate 0.05 with a 1,500-round cap by +0.005 on Odisha and +0.002 on Bihar; on Odisha, tree depth 8 and AFT scale 0.8 or 2.0 added nothing further (all within 0.001 on validation).

## Delhi (state code 26): survival mode

- **Data:** DDL 2010-2013 filings, 459,712 cases, about 10% with no decision date. The original Delhi classifier drops those cases and is unchanged; this is a separate `delhi_survival` entry.
- **Censoring and cutoff:** same rule as Odisha and Bihar. Cutoff 2019-02-28, estimated from pending cases' last listing dates and consistent with the drop in monthly decisions (1,649 in Jan 2019, 607 in Feb, 161 in Mar). 99.9% of pending cases have a next listing date.
- **Model:** XGBoost AFT, same setup and features as the other states (learning rate 0.10, early stopping; best iteration 1113).
- **Result:** test C-index 0.7508 (random split) and 0.7304 on the time split (train 2010-12 filings, test 2013 filings). Best-worst C-index gap among groups with at least 1,000 test cases: court tier 0.150 (Family Court 0.770 vs Civil Judge (Senior Division) 0.621), district 0.140 (North West 0.780 vs Shahdara 0.640), case type 0.139 (ct cases 0.711 vs cs scj 0.572). On Low/Medium/High tiers (tertiles within the state) Macro F1 is 0.621 overall (see Tier-level results). The C-index is not comparable across states or with the Delhi v1 classifier's Macro F1.
- **Limitations:** court tiers come from a Delhi-specific rule set, with two tiers that exist only in Delhi (Family Court, Labour / Industrial Tribunal). The C-index is not comparable with the old classifier's Macro F1 of 0.59.

## Cross-state experiment (Delhi, Odisha, Bihar)

`scripts/experiments/cross_state.py` and `cross_state_v2.py` compare models on the same features (case type, court tier, four gender fields; district is excluded because district names are state-specific), with learning rate 0.10 and early stopping. Test C-index (intervals and details in [docs/cross_state_check.md](docs/cross_state_check.md)):

| Setup | Delhi | Odisha | Bihar |
|---|---|---|---|
| in-state | 0.7258 | 0.7043 | 0.7484 |
| pooled + state input | 0.7258 | 0.7043 | 0.7482 |
| pooled shared (no state input) | 0.7247 | 0.6973 | 0.7469 |
| hold-out | 0.5888 | 0.5124 | 0.6047 |
| in-state, no case type | 0.6498 | 0.6190 | 0.7045 |
| hold-out, no case type | 0.6044 | 0.5126 | 0.5411 |

- A pooled model with a `state` input matches the per-state models to four decimals because trees can split on state first; this says nothing about shared structure.
- Without the state input the pooled model loses 0.007 at most, but court tiers and case-type labels differ by state, so the features still reveal the state.
- The hold-out scores 0.51-0.60 (Odisha at chance). With case type removed, the hold-out is still 0.045 to 0.163 below the in-state score. Per-state models stay.
- Court-tier names and rules are still state-specific, so "no case type" is not a fully comparable feature set. Dropping district lowers the in-state C-index by about 0.02-0.04, so district carries real signal. All numbers come from a single train/test split.
