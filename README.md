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
