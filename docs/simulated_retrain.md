## Simulated time-sliced retrain (Odisha)

Script: `scripts/simulate_retrain.py --state odisha`. Result file: `reports/simulated_retrain_orissa.json`.

**Setup.** Cases filed in 2011 or earlier are the "old" data. A stale model is trained on them only.
The drift check then compares old and newer filings: case-type PSI 0.1755 (threshold 0.10), so a retrain is triggered.
A retrained model uses the old data plus 70% of the newer cases. Both models are scored on the other 30% of newer
cases, which neither model saw.

| Model | C-index | Macro precision | Macro recall | Macro F1 | Training time |
|---|---|---|---|---|---|
| Stale | 0.720 | 0.519 | 0.506 | 0.510 | 137 s |
| Retrained | 0.750 | 0.553 | 0.540 | 0.544 | 237 s |

**Limits**
- One state and one random seed.
- No control run that retrains without drift, so part of the gain may come from more data alone.
- "Stale" here means trained on 2010-2011 filings only, which is a harder case than a monthly retrain.
