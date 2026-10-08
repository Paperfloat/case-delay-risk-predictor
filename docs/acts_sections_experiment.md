## Act and section feature: experiments

Source: the Development Data Lab acts-and-sections table (`data/raw/acts_sections.tar.gz`), joined by `ddl_case_id`
(`scripts/extract_acts.py`). Features: first-listed act and section (top 30 each), `criminal`, `bailable_ipc`, IPC
section count. Each experiment trains the same model with and without these features on the same split
(`scripts/acts_experiment*.py`); the 95% interval comes from a bootstrap of the C-index difference.

**Coverage.** Cases with act info: Delhi 48.7%, Odisha 37.7%, Bihar 71.3%.

**Recording artifact.** Having act info depends on the outcome: pending 77% vs decided 45% (Delhi), 48% vs 31% (Odisha),
79% vs 62% (Bihar). The gap stays large inside several case types (for example Delhi `arbtn`, `hma`; Odisha `uc`; Bihar
`gr case`). Having a first section also differs: pending 71% vs decided 93% (Delhi), 64% vs 77% (Odisha), 80% vs 73%
(Bihar). A model could learn "missing means pending", which a real user could not supply, so the fair test uses only
cases that have both an act and a section.

**Clean test (cases with act and section only, so missing values carry no signal)**

| State | Cases | Random split | Time split |
|---|---|---|---|
| Delhi | 200,084 | 0.7549 -> 0.7986 (+0.044, 95% +0.041 to +0.046) | 0.7461 -> 0.7920 (+0.046, 95% +0.044 to +0.048) |
| Odisha | 124,876 | 0.7698 -> 0.7799 (+0.010, 95% +0.008 to +0.012) | 0.7263 -> 0.7363 (+0.010, 95% +0.008 to +0.011) |
| Bihar | 449,178 | 0.8261 -> 0.8340 (+0.008, 95% +0.007 to +0.009) | 0.8417 -> 0.8264 (-0.015, 95% -0.016 to -0.014) |

**Reading.** The act and section content helps in Delhi and Odisha on both splits. In Bihar it helps on a random split
but hurts on the time split, so it is not used for Bihar. Act information is not recorded for most cases, and I cannot
tell from the data when it is entered. The gain applies only to cases where the user supplies both fields.

Earlier runs (all cases, and cases with an act): `reports/acts_experiment_*.json`, `reports/acts_experiment_subset_*.json`.
