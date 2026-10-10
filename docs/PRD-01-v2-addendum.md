# PRD-01 v2 addendum: what changed and where it stands

This addendum records how the project differs from the original PRD-01 and the status of each requirement. Items marked **proposed** are drafts for the project owner to confirm.

## What changed from v1

| Topic | v1 PRD | Built |
|---|---|---|
| Data source | NJDG-style court data | Development Data Lab public e-Courts dataset (bulk files), filings 2010-2013 |
| Problem framing | Three-class risk tier classifier on decided cases | Survival model (XGBoost AFT) that keeps pending cases as censored; tiers derived from predicted duration |
| Geography | Delhi; multi-state was out of scope | Delhi, Odisha, Bihar, config-driven so another state is a config entry |
| Evaluation metric | Macro F1 | C-index (ranking quality) with bootstrap confidence intervals per group, plus precision, recall and Macro F1 on tiers derived from predicted duration |
| Serving | Single-state classifier API | Multi-state survival API with audit log, deployed to Render (free plan) by CI/CD; the v1 API is kept as `api/legacy_classifier.py` |
| Features | Filing-time features, with process history as a possible later addition | Filing-time features only; the act and section are used only by an optional Delhi model; process history is not used |

## Status against the v1 success metrics and requirements

| Requirement | Status |
|---|---|
| Macro F1 of at least 0.75 | **Not met.** The v1 classifier reached 0.59. For the survival models, predicted days are turned into Low / Medium / High (tertiles within each state) and scored on the test split: Macro F1 is 0.62 (Delhi), 0.53 (Odisha) and 0.51 (Bihar). Odisha and Bihar cover only cases whose tier is known (88% and 74%). See `docs/survival_tier_evaluation.md`. The C-index (0.75 Delhi, 0.75 Odisha, 0.80 Bihar on a random split; 0.73, 0.73, 0.80 on a time split) measures ranking and is a different metric. |
| Case-type subgroup gap under 10% | **Not met.** The v1 classifier's gap was about 35%. For the survival models the best-worst Macro F1 gap by case type is 0.24, 0.17, 0.42 (Delhi, Odisha, Bihar) and the C-index gap is 0.14, 0.12, 0.27. Court-tier gaps are 0.17, 0.09, 0.18 and district gaps 0.15, 0.30, 0.25 (Macro F1): all but Odisha's court tier (0.093, within noise of 0.10) are above the target. Reweighting training cases by group did not close any gap (docs/fairness_mitigation.md). |
| Subgroup reporting on every training run, with intervals on thin slices | **Met for the survival models.** Reports for court tier, district and case type, with 95% bootstrap intervals, are written on every training run. Groups under 1,000 test cases are not reported. |
| Per-prediction explanation (top five factors) | **Met.** Every API response includes the top five TreeSHAP factors on a log-days scale. |
| Audit log of predictions | **Met** (SQLite table with state and model version). |
| Drift monitoring on key features | **Met.** `monitoring/survival_monitor.py` computes PSI per feature for all three survival models and writes a dashboard and Evidently reports. Odisha and Bihar flag case-type drift (PSI 0.175 and 0.170); Delhi is stable (0.093). The comparison is early versus later filings in one fixed dataset, not live traffic. |
| Retrain-to-deploy time under 15 minutes | **Partly met.** Retrain plus build-and-deploy takes 8m 05s for Delhi and 10m 02s for Odisha (met) and 17m 22s for Bihar (not met); all three one after another take 26m 01s. Only the training stage was timed, because the data stages were cached. See `docs/retrain_benchmark.md`. |
| Automatic retrain trigger and simulated time-sliced retrain | **Met.** `.github/workflows/retrain.yml` runs monthly and on demand: it runs the drift check, retrains the drifted states, pushes models to DagsHub and opens a pull request (run once on demand). In a simulation on Odisha, a model trained on filings up to 2011 scored C-index 0.720 and Macro F1 0.510 on newer cases; after the drift trigger and a retrain, 0.750 and 0.544. One state, one seed, no no-drift control. See `docs/simulated_retrain.md`. |
| Model registry | **Met.** `scripts/register_models.py` registers the three survival models in a local MLflow Model Registry with the alias `champion`. |
| Continuous deployment | **Met.** On every push to `main`, CI pushes the Docker image to GitHub Container Registry, triggers Render, and smoke-tests the live service (https://case-delay-api.onrender.com). On the free plan the service sleeps when idle and the audit log resets on each deploy. |
| Data validation in CI | **Met.** Great Expectations checks run per state in a CI matrix. |

## Proposed v2 targets (drafts)

1. **Report, don't hide:** every release reports the best-worst C-index gap by court tier, district and case type with intervals, and flags any group whose interval lies entirely below 0.65.
2. **Court-tier gap:** at most 0.10 C-index in each state (currently 0.099 Odisha, 0.150 Delhi, 0.194 Bihar).
3. **Calibration:** before any predicted day count is shown to users, a per-tier Kaplan-Meier check must show that observed decision rates rise from Low to High tier within each state.
4. **Time-split reporting:** quote the time-split C-index alongside the random-split value in every results table.

## Open items

- Calibration: done (`docs/calibration_check.md`). Tiers are correctly ordered in all three states, but predicted day counts are not calibrated and the High-tier days in Odisha and Bihar are extrapolations. Show the tier, not the day count.

- Act and section: used only in an optional Delhi model (+0.043 time-split C-index on cases with both known). It hurts on the time split in Bihar, is not in the retrain workflow, drift monitor or registry, and whether a case has act information depends on its outcome in the records.
- Odisha and Bihar pending share: early decisions look unrecorded (docs/bihar_pending_investigation.md); not proven, and its effect on the models is untested. Districts such as Vaishali (94% pending) remain unexplained.
- Fairness gaps are above target for tier Macro F1 and C-index in nearly every state; the one mitigation tried (reweighting by group) did not help.
- Retraining on genuinely new data (not a fixed 2010-2013 snapshot) is untested, and Bihar retraining exceeds 15 minutes.
- A persistent audit log on the live host (the free plan resets it on every deploy).
