# Fairness case study: where does the delay model rank cases well, and where doesn't it?

Numbers in this document come from the models retrained after the case-type spelling cleanup (`reports/*_survival_fairness_*.csv`).

## Question

The model ranks cases by expected time to decision using only what is known at filing. If it ranks some kinds of cases much worse than others, the tier it assigns carries less information for those cases, and anyone using it to prioritise work would be relying on it unevenly. This case study checks that for three states (Delhi, Odisha, Bihar) across court tier, district and case type.

## Method

- **Metric:** the concordance index (C-index) within each group, computed on the held-out 15% test set. It measures how well the model orders cases by duration within the group, handling pending cases as censored. 0.5 is random.
- **Groups:** every court tier, district and case type with at least 1,000 test cases and at least 50 observed decisions.
- **Uncertainty:** 95% intervals from 100 bootstrap resamples of each group's test cases.
- **Gap:** best group minus worst group for each dimension.
- **Models:** the final XGBoost AFT model for each state, trained on about 72% of cases (a random split; 13% is held out for early stopping and 15% for this test set). See the README for the cutoffs and censoring rates.

## Finding 1: the gaps are large and are not sampling noise

For every state and dimension, the best and worst groups' intervals do not overlap.

| State | Court tier gap | District gap | Case type gap |
|---|---|---|---|
| Delhi | 0.150 | 0.140 | 0.139 |
| Odisha | 0.099 | 0.136 | 0.116 |
| Bihar | 0.194 | 0.211 | 0.273 |

## Finding 2: civil suits are consistently the hardest to rank

Court tier C-index (n = test cases; the worst tier in each state is in bold):

| Tier | Delhi | Odisha | Bihar |
|---|---|---|---|
| Civil Judge (Senior Division) | **0.621 (n=12,973)** | **0.689 (n=12,904)** | **0.659 (n=7,065)** |
| District and Sessions Judge | 0.730 (n=27,984) | 0.765 (n=9,969) | 0.853 (n=29,998) |
| Chief Judicial Magistrate | 0.736 (n=20,259) | 0.741 (n=22,603) | 0.740 (n=77,483) |
| Labour / Industrial Tribunal | 0.721 (n=3,691) |  |  |
| Family Court | 0.770 (n=4,050) |  |  |
| Other / unclear |  | 0.724 (n=1,558) | 0.774 (n=2,613) |
| Sub-Divisional Judicial Magistrate |  | 0.734 (n=13,117) |  |
| Civil Judge cum JMFC |  | 0.766 (n=1,746) |  |
| Judicial Magistrate First Class |  | 0.788 (n=7,551) |  |
| Civil Judge (division unclear) |  |  | 0.736 (n=5,213) |

Worst and best tier with 95% intervals: Delhi: Civil Judge (Senior Division) 0.621 [0.615-0.627] vs Family Court 0.770 [0.762-0.775]; Odisha: Civil Judge (Senior Division) 0.689 [0.683-0.695] vs Judicial Magistrate First Class 0.788 [0.781-0.798]; Bihar: Civil Judge (Senior Division) 0.659 [0.649-0.668] vs District and Sessions Judge 0.853 [0.851-0.855].

- The Civil Judge (Senior Division) tier is the worst-ranked tier in all three states, and its interval does not overlap the best tier's in 3 of the 3 states.
- Delhi's worst case type is "cs scj" (C-index 0.572) is also a civil suit category.
- Chief Judicial Magistrate courts score 0.736-0.741 across the three states.
- District and Sessions Judge courts score 0.730-0.853 across the three states.
- The best-ranked tier is different in each state (Delhi: Family Court, Odisha: Judicial Magistrate First Class, Bihar: District and Sessions Judge), so there is no single "easy" kind of court.

**Why this might happen (not tested):** civil suits can take very different amounts of time depending on procedural events after filing (service of notice, adjournments, evidence), and those are not available when the model makes its prediction. The model only sees case type, court tier, district and gender fields, so it has little to separate fast civil suits from slow ones.

## Finding 3: some of the worst-looking groups are thin or oddly recorded

- **Vaishali (Bihar)** has the lowest district C-index (0.617) with a wide interval (0.520-0.686), and 96% of its test cases are pending, against 47% in Patna. A follow-up (docs/bihar_pending_investigation.md) found that its pending cases have recent hearing dates (0.4% are stale) and that its recorded decisions fall only in 2018-2020, so it is not clearly a recording artifact; the cause is unexplained. Districts with a pending share of about 95% or higher are treated as unreliable.
- **Case types** with the weakest ranking in Bihar: "regular bail" 0.569 [0.557-0.578]; "gr police cases" 0.575 [0.560-0.590], barely above random, which means the model has little ranking ability for them.
- **Censoring rates differ by group** (for example 6% for Delhi's Civil Judge (Senior Division) cases against 53% for Odisha's best tier), so the C-index is not computed on equally informative samples across groups. Odisha's best tier is among the most censored, so censoring alone does not explain the gaps.

## Fairness mitigation

Reweighting training cases by court tier, district or case type did not close any gap beyond noise in 27 comparisons (docs/fairness_mitigation.md). The gaps are probably caused by groups that are harder to predict from filing-time features, which is not tested.

## What this audit does not show

- **Calibration by group.** The overall tier check (docs/calibration_check.md) shows the tiers are correctly ordered, but it was not repeated by court tier or district. The C-index checks ordering within a group, not whether predicted durations or tiers are systematically too long or too short for it.
- **Tier assignment rates.** I have not checked whether some courts or districts receive the High tier far more often than their observed outcomes justify.
- **Real-world impact.** Nothing here measures what happens when someone acts on the tier.
- **Causes.** The patterns above are associations, and the explanations are hypotheses.

## Caveats on the numbers

- One train/test split and one random seed.
- The intervals resample cases as if independent, but cases in the same court share outcomes, so the true uncertainty is larger than shown.
- Court tier names come from hand-written rules per state (for example, Delhi's "Chief Judicial Magistrate" tier is its Chief Metropolitan Magistrate courts, and its "Civil Judge (Senior Division)" tier is "Senior Civil Judge cum RC"), so a tier is not exactly the same thing across states.
- Groups below 1,000 test cases are not reported, so small courts and rare case types are not audited.
- In Odisha and Bihar, decisions made before about 2013 look largely unrecorded, so durations for older filings may be biased long (not tested; see docs/bihar_pending_investigation.md).

## Next steps

1. Repeat the Kaplan-Meier tier check by court tier and district (the overall check is done).
2. Tier assignment rates by court tier and district.
3. Find out why Vaishali and Gaya (about 95% pending) look the way they do; this needs the dataset's documentation.
4. The act and section of a case were tested as extra features: they help a Delhi model (+0.043 time-split C-index on cases with both fields known) and are used there as an optional model, but they are mixed in Odisha and hurt in Bihar (docs/acts_sections_experiment.md).
