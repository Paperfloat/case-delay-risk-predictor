## Party fields: what the gender flags did, and why the models now use "known / not classified"

The earlier models used four fields from the Development Data Lab records: `female_defendant`, `female_petitioner`, `female_adv_def` and `female_adv_pet`. Their values (0 male, 1 female, -9998 unclear, -9999 missing) are inferred from names by the dataset builders. We had never tested whether they help, and the fairness audit did not look at them. Scripts: `scripts/gender_ablation.py` and `scripts/gender_collapse.py` (results: `reports/gender_ablation_*.json`, `reports/gender_collapse_*.json`).

### Test 1: do the fields help? (model with all four fields against a model without them)

| State | Split | With gender | Without | Difference | 95% interval |
|---|---|---|---|---|---|
| Delhi | time | 0.7304 | 0.7119 | +0.0185 | +0.0178 to +0.0191 |
| Delhi | random | 0.7508 | 0.7323 | +0.0185 | +0.0174 to +0.0197 |
| Odisha | time | 0.7275 | 0.7183 | +0.0092 | +0.0086 to +0.0101 |
| Odisha | random | 0.7490 | 0.7343 | +0.0147 | +0.0137 to +0.0156 |
| Bihar | time | 0.8008 | 0.7917 | +0.0092 | +0.0086 to +0.0096 |
| Bihar | random | 0.7969 | 0.7788 | +0.0180 | +0.0172 to +0.0188 |

The fields help in every state on both splits.

### Test 2: where do the models put cases, by party group? (random split, High-tier share and C-index)

| State | Field | Value | Cases | High-tier share, with | without | C-index, with | without |
|---|---|---|---|---|---|---|---|
| Delhi | female_defendant | 0 male | 44,930 | 0.366 | 0.349 | 0.7516 | 0.7348 |
| Delhi | female_defendant | -9998 unclear | 14,506 | 0.262 | 0.301 | 0.7367 | 0.7131 |
| Delhi | female_defendant | 1 female | 9,509 | 0.282 | 0.235 | 0.7530 | 0.7373 |
| Delhi | female_petitioner | 0 male | 30,442 | 0.266 | 0.251 | 0.7355 | 0.7122 |
| Delhi | female_petitioner | -9998 unclear | 27,071 | 0.396 | 0.415 | 0.7674 | 0.7581 |
| Delhi | female_petitioner | 1 female | 11,442 | 0.363 | 0.299 | 0.7437 | 0.7218 |
| Odisha | female_defendant | 0 male | 48,857 | 0.349 | 0.341 | 0.7466 | 0.7314 |
| Odisha | female_defendant | 1 female | 12,160 | 0.302 | 0.319 | 0.7335 | 0.7195 |
| Odisha | female_defendant | -9998 unclear | 9,556 | 0.292 | 0.294 | 0.7747 | 0.7626 |
| Odisha | female_petitioner | -9998 unclear | 38,433 | 0.385 | 0.395 | 0.7553 | 0.7386 |
| Odisha | female_petitioner | 0 male | 22,928 | 0.276 | 0.257 | 0.7446 | 0.7325 |
| Odisha | female_petitioner | 1 female | 9,253 | 0.259 | 0.247 | 0.7259 | 0.7124 |
| Bihar | female_defendant | 0 male | 96,275 | 0.365 | 0.351 | 0.7849 | 0.7691 |
| Bihar | female_defendant | -9998 unclear | 14,502 | 0.176 | 0.221 | 0.8156 | 0.7924 |
| Bihar | female_defendant | 1 female | 11,328 | 0.242 | 0.252 | 0.7897 | 0.7781 |
| Bihar | female_petitioner | 0 male | 56,282 | 0.258 | 0.257 | 0.7623 | 0.7417 |
| Bihar | female_petitioner | -9998 unclear | 42,111 | 0.492 | 0.467 | 0.8540 | 0.8389 |
| Bihar | female_petitioner | 1 female | 23,526 | 0.214 | 0.232 | 0.7289 | 0.7157 |

The "-9998 unclear" petitioner group is placed in the High tier far more often (for example 49% against 26% for male petitioners in Bihar), which fits the idea that this group contains institutions such as the State. That is a hypothesis; the group was not inspected. Group differences remain without the fields (for example Delhi female defendants 24%, male defendants 35%), because case type, court and district correlate with who the parties are.

### Test 3: keep only "known" (name classified as a person) or "unclear"

| State | Split | With gender | Known / unclear only | No party fields | Share of the gain kept |
|---|---|---|---|---|---|
| Delhi | time | 0.7304 | 0.7302 | 0.7119 | 99% |
| Delhi | random | 0.7508 | 0.7493 | 0.7323 | 92% |
| Odisha | time | 0.7275 | 0.7259 | 0.7183 | 83% |
| Odisha | random | 0.7490 | 0.7459 | 0.7343 | 79% |
| Bihar | time | 0.8008 | 0.7995 | 0.7917 | 86% |
| Bihar | random | 0.7969 | 0.7937 | 0.7788 | 82% |

Share of cases whose party name was classified as a person: Delhi: defendant 79%, petitioner 61%, adv_def 1%, adv_pet 2%; Odisha: defendant 87%, petitioner 45%, adv_def 22%, adv_pet 49%; Bihar: defendant 88%, petitioner 65%, adv_def 19%, adv_pet 36%. Delhi's lawyer fields are almost always "unclear", so they carry little there.

### Decision

The pipeline (`scripts/pipeline/normalize.py`) now keeps only whether each party name was classified as a person ("known") or not ("unclear"); the original values stay in `*_raw` columns. This keeps 79%-99% of the gain and costs 0.0002-0.003 C-index against the full fields. The API and the app show the values as "Person (name classified)" and "Not classified (for example an organisation or an unclear name)". The models were retrained and all downstream reports rerun.

### Limits

- One split per state, and the ablation and collapse runs share it. Intervals resample test cases only.
- What falls in "not classified" has not been inspected; the organisation explanation is a hypothesis.
- The act and section experiments, the fairness mitigation run and the cross-state check ran with the earlier male / female / unclear values and were not rerun; their conclusions are not expected to depend on this change. Delhi's act-aware model was retrained.
- Collapsing the fields removes the inferred gender from the model but does not remove group differences.
