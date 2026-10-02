# PRD-01 v2 addendum: what changed and where it stands

This addendum records how the project differs from the original PRD-01 and the status of each requirement. Items marked **proposed** are drafts for the project owner to confirm.

## What changed from v1

| Topic | v1 PRD | Built |
|---|---|---|
| Data source | NJDG-style court data | Development Data Lab public e-Courts dataset (bulk files), filings 2010-2013 |
| Problem framing | Three-class risk tier classifier on decided cases | Survival model (XGBoost AFT) that keeps pending cases as censored; tiers derived from predicted duration |
| Geography | Delhi; multi-state was out of scope | Delhi, Odisha, Bihar, config-driven so another state is a config entry |
| Evaluation metric | Macro F1 | C-index (ranking quality), with bootstrap confidence intervals per group |
| Serving | Single-state classifier API | Multi-state survival API with audit log; the v1 API is kept as `api/legacy_classifier.py` |
| Features | Filing-time features, with process history as a possible later addition | Filing-time features only; process history and the act or section of a case are not used |

## Status against the v1 success metrics and requirements

| Requirement | Status |
|---|---|
| Macro F1 of at least 0.75 | **Not met.** The v1 classifier reached 0.59. The survival models are evaluated with the C-index (0.75 Delhi, 0.75 Odisha, 0.80 Bihar on a random split; 0.73, 0.73, 0.80 on a time split), which is not comparable. A Macro F1 on tiers derived from the survival predictions has not been computed. |
| Case-type subgroup gap under 10% | **Not met.** The v1 classifier's gap was about 35%. For the survival models, the best-worst C-index gap by case type is 0.14 (Delhi), 0.14 (Odisha) and 0.29 (Bihar), and the intervals do not overlap. |
| Subgroup reporting on every training run, with intervals on thin slices | **Met for the survival models.** Reports for court tier, district and case type, with 95% bootstrap intervals, are written on every training run. Groups under 1,000 test cases are not reported. |
| Per-prediction explanation (top five factors) | **Met.** Every API response includes the top five TreeSHAP factors on a log-days scale. |
| Audit log of predictions | **Met** (SQLite table with state and model version). |
| Drift monitoring on key features | **Partly met.** Built for the v1 Delhi classifier only; not extended to the survival models. |
| Retrain-to-deploy time under 15 minutes | **Not measured.** Retraining the three models and rebuilding the image are separate steps, and neither has been timed end to end. |
| Data validation in CI | **Met.** Great Expectations checks run per state in a CI matrix. |

## Proposed v2 targets (drafts)

1. **Report, don't hide:** every release reports the best-worst C-index gap by court tier, district and case type with intervals, and flags any group whose interval lies entirely below 0.65.
2. **Court-tier gap:** at most 0.10 C-index in each state (currently 0.104 Odisha, 0.149 Delhi, 0.194 Bihar).
3. **Calibration:** before any predicted day count is shown to users, a per-tier Kaplan-Meier check must show that observed decision rates rise from Low to High tier within each state.
4. **Time-split reporting:** quote the time-split C-index alongside the random-split value in every results table.

## Open items

- Calibration check for the tiers and the predicted day counts (not done).
- Per-state drift and fairness monitoring for the survival models.
- The act or section of each case as a feature (needs a separate DDL download).
- Explain Bihar's pending share falling for newer filings, and check districts with extreme pending shares (for example Vaishali, 96%).
