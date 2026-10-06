## Retrain-to-deploy benchmark (PRD target: under 15 minutes)

Retrain time was measured with `dvc repro -f -s <stage>` on a local laptop (WSL, 4 training threads).
Deploy time is the GitHub Actions run from push to a live, smoke-tested Render service (4m 44s).

| Scope | Retrain | Deploy | Total | Under 15 min? |
|---|---|---|---|---|
| Delhi | 3m 21s | 4m 44s | 8m 05s | Yes |
| Odisha | 5m 18s | 4m 44s | 10m 02s | Yes |
| Bihar | 12m 38s | 4m 44s | 17m 22s | No |
| All three, one after another | 21m 17s | 4m 44s | 26m 01s | No |

**Limits**
- Only the training stage was timed. Data stages (load, features, normalize) were
  cached by DVC, so retraining on brand-new data would take longer.
- Times come from one laptop run. GitHub runners will differ.
- Bihar has the most rows and misses the target.
