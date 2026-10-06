# Architecture

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

## Pipeline stages (`dvc.yaml`)

| Stage | Input | Output | Notes |
|---|---|---|---|
| `load@state` | national case files per year | the state's rows per year | filters by state code |
| `features@state` | state rows + key tables | features file | joins case type, court and district names; keeps pending cases with an `event` flag |
| `normalize@state` | features file | normalized file | maps court names to tiers with a per-state rule set; keeps the top 100 case types |
| `train_*` | normalized file | model, feature list, metadata, fairness report | early stopping, bootstrap confidence intervals, tier cut points |

Survival-mode datasets are stored as `.csv.gz`, because uploads of the 150-260 MB uncompressed files to the DVC remote kept failing.

## Configuration

`config/states.yaml` holds, per state: the state code, the file slug, whether pending cases are kept, the data cutoff date used for censoring, the court-tier rule set, and the validation bounds (row count and share of decided cases). Adding a state means adding an entry, adding it to the `foreach` lists in `dvc.yaml`, and checking its court names.

The cutoff for each state is estimated from the data (`scripts/pipeline/estimate_cutoff.py`): the end of the month containing the 90th percentile of pending cases' last hearing dates. It reproduces the cutoff found by hand from the drop in monthly decisions.

## Serving

`api/main.py` loads every state's model at start-up from `models/*_aft_meta.json`. Endpoints: `POST /predict`, `GET /options/{state}`, `GET /states`, `GET /health`. Each prediction returns predicted days, a Low/Medium/High tier (tertiles of predicted days on the state's test set), the top five TreeSHAP factors, the model version, and warnings. Every request is stored in the `survival_predictions` table of a SQLite audit log.

## CI (GitHub Actions)

- **Data validation:** one job per dataset (a matrix), each pulling only the file it validates from the DVC remote and running the Great Expectations checks.
- **Docker build and smoke test:** pulls the model files, builds the image, starts the container, and sends a valid request for each state, checking the response and the audit log write.
- **Deploy (push to `main` only):** after the smoke test passes, the image is pushed to GitHub Container Registry, Render is triggered through a deploy hook, and the live service is smoke-tested.
- **Drift check and retrain (monthly and on demand):** see Monitoring and retraining below.

## Monitoring and retraining

`monitoring/survival_monitor.py` compares cases filed up to 2011 (reference) with later filings (current) for each state. It computes PSI per model feature with a threshold of 0.10, and writes `monitoring/survival_dashboard.html` (overall scores, fairness gaps, weakest subgroups), per-state Evidently drift reports and `monitoring/survival_drift_status.json`. The fairness numbers come from `scripts/survival_tier_report.py`, which turns predicted days into Low / Medium / High (tertiles within each state) and scores precision, recall and Macro F1 per subgroup.

`.github/workflows/retrain.yml` runs the drift check monthly and on demand. It retrains only the states that drifted (`dvc repro -f -s <stage>`), pushes the models to DagsHub and opens a pull request. Merging that pull request starts the deploy pipeline. The drift comparison uses a fixed 2010-2013 snapshot, so Odisha and Bihar flag drift on every run; a live system would compare training data with new cases. `scripts/simulate_retrain.py` runs a time-sliced simulation for one state (see `docs/simulated_retrain.md`), and `docs/retrain_benchmark.md` records the retrain-to-deploy timings.

## Model registry

`scripts/register_models.py` registers each state's saved model in a local MLflow Model Registry (`sqlite:///mlflow.db`, not committed) as `case-delay-aft-<state>` with the alias `champion`. It does not retrain.

## Deployment

The API runs on Render's free plan from the image `ghcr.io/<owner>/case-delay-api` (a public package). The free service sleeps when idle, so the first request can take about a minute, and the SQLite audit log lives inside the container, so it resets on every deploy or restart.
