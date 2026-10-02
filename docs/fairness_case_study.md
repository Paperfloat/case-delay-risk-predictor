# Fairness case study: where does the delay model rank cases well, and where doesn't it?

## Question

The model ranks cases by expected time to decision using only what is known at filing. If it ranks some kinds of cases much worse than others, the tier it assigns carries less information for those cases, and anyone using it to prioritise work would be relying on it unevenly. This case study checks that for three states (Delhi, Odisha, Bihar) across court tier, district and case type.

## Method

- **Metric:** the concordance index (C-index) within each group, computed on the held-out 15% test set. It measures how well the model orders cases by duration within the group, handling pending cases as censored. 0.5 is random.
- **Groups:** every court tier, district and case type with at least 1,000 test cases and at least 50 observed decisions.
- **Uncertainty:** 95% intervals from 100 bootstrap resamples of each group's test cases.
- **Gap:** best group minus worst group for each dimension.
- **Models:** the final XGBoost AFT model for each state, trained on a random 70% of cases (see the README for the cutoffs and censoring rates).

## Finding 1: the gaps are large and are not sampling noise

For every state and dimension, the best and worst groups' intervals do not overlap.

| State | Court tier gap | District gap | Case type gap |
|---|---|---|---|
| Delhi | 0.149 | 0.140 | 0.138 |
| Odisha | 0.104 | 0.133 | 0.140 |
| Bihar | 0.194 | 0.211 | 0.291 |

## Finding 2: civil suits are consistently the hardest to rank

Court tier C-index (95% interval for the two extremes), by state:

| Tier | Delhi | Odisha | Bihar |
|---|---|---|---|
| Civil Judge (Senior Division) | **0.621** [0.615-0.627], n=12,973 | **0.687** [0.680-0.693], n=12,904 | **0.658** [0.648-0.667], n=7,065 |
| Chief Judicial Magistrate | 0.736 | 0.740 | 0.743 |
| District and Sessions Judge | 0.730 | 0.761 | 0.852 [0.850-0.855], n=29,998 |
| Family Court | 0.770 [0.762-0.775], n=4,050 | | |
| Judicial Magistrate First Class | | 0.791 [0.784-0.800], n=7,551 | |
| Other tiers | Labour / Industrial Tribunal 0.721 | Sub-Divisional Judicial Magistrate 0.737; Civil Judge cum JMFC 0.767; Other / unclear 0.725 | Civil Judge (division unclear) 0.736; Other / unclear 0.774 |

- The Civil Judge (Senior Division) tier is the worst-ranked tier in all three states, and its interval does not overlap the best tier's in any of them.
- Delhi's worst case type is also a civil suit category ("cs scj", C-index 0.573).
- Chief Judicial Magistrate courts score almost the same in all three states (0.736-0.743), while District and Sessions Judge courts range from 0.730 to 0.852.
- The best-ranked tier is different in each state (Family Court, Judicial Magistrate First Class, District and Sessions Judge), so there is no single "easy" kind of court.

**Why this might happen (not tested):** civil suits can take very different amounts of time depending on procedural events after filing (service of notice, adjournments, evidence), and those are not available when the model makes its prediction. The model only sees case type, court tier, district and gender fields, so it has little to separate fast civil suits from slow ones.

## Finding 3: some of the worst-looking groups are thin or oddly recorded

- **Vaishali (Bihar)** has the lowest district C-index (0.616) but a very wide interval (0.530-0.688), and 96% of its test cases are pending, against 47% in Patna. With so few recorded decisions, its score is not reliable. A 96% pending share also suggests decision dates may be largely unrecorded there. I have not checked that.
- **Case types** such as "gr/police cases" in Bihar (C-index 0.530, interval 0.505-0.556) are close to random, which means the model has almost no ranking ability for them.
- **Censoring rates differ by group** (for example 6% for Delhi's Civil Judge (Senior Division) cases against 53% for Odisha's best tier), so the C-index is not computed on equally informative samples across groups. Odisha's best tier is among the most censored, so censoring alone does not explain the gaps.

## What this audit does not show

- **Calibration by group.** The C-index checks ordering within a group, not whether predicted durations or tiers are systematically too long or too short for it. A model can rank a group well and still place it in the wrong tier.
- **Tier assignment rates.** I have not checked whether some courts or districts receive the High tier far more often than their observed outcomes justify.
- **Real-world impact.** Nothing here measures what happens when someone acts on the tier.
- **Causes.** The patterns above are associations, and the explanations are hypotheses.

## Caveats on the numbers

- One train/test split and one random seed.
- The intervals resample cases as if independent, but cases in the same court share outcomes, so the true uncertainty is larger than shown.
- Court tier names come from hand-written rules per state (for example, Delhi's "Chief Judicial Magistrate" tier is its Chief Metropolitan Magistrate courts, and its "Civil Judge (Senior Division)" tier is "Senior Civil Judge cum RC"), so a tier is not exactly the same thing across states.
- Groups below 1,000 test cases are not reported, so small courts and rare case types are not audited.

## Next steps

1. A Kaplan-Meier check per predicted tier: what share of cases in each tier were actually decided within one, three and five years, by court tier. This tests calibration of the tiers without being biased by pending cases.
2. Tier assignment rates by court tier and district.
3. Check Vaishali's decision-date recording, and the same for any district with an extreme pending share.
4. Add filing-time features that could separate civil suits (for example the act or section of the case, which needs a separate data download).
