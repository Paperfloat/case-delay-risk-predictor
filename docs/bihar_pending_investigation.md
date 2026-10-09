## Why Bihar's pending share falls for newer filings (and what the same check shows for Odisha and Delhi)

Script: `scripts/bihar_pending_analysis.py`. Question: Bihar's share of cases with no decision falls from 61.3% (2010
filings) to 50.4% (2013 filings), the opposite of what pure right-censoring predicts.

**Mix explains part of it.** Holding the 2010 mix fixed: case type 0.618 -> 0.580 (drop 0.038 of the raw 0.109, about
65% explained, using the 44 case types with at least 200 cases per year); district 0.613 -> 0.531; court tier
0.613 -> 0.533.

**Unrecorded decisions are not the cause.** Only 0.5-1.3% of pending cases have a last hearing over 3 years before the
cutoff (0.3-0.8% of all cases).

**Early decisions look missing in Odisha and Bihar, not in Delhi.**

| Share of all cases decided within 1 year, by filing year | 2010 | 2011 | 2012 | 2013 |
|---|---|---|---|---|
| Delhi | 44.5% | 44.3% | 44.1% | 42.0% |
| Odisha | 0.1% | 1.7% | 6.9% | 13.9% |
| Bihar | 2.2% | 7.3% | 8.1% | 18.3% |

Share of all recorded decisions that fall in each decision year: Bihar 0.5% (2010), 3.2%, 4.2%, 13.3% (2013); Odisha
0.0%, 0.3%, 2.9%, 13.6%; Delhi 6.6%, 12.7%, 16.7%, 19.8%.

**Reading.** The pattern fits decisions made before about 2013 not being recorded in Odisha and Bihar, so that cases
filed in 2010-2012 and still open are over-represented. This is not proven (it needs the dataset's documentation).
If true, durations for 2010-2012 filings in those states are biased long, which could affect the true tiers, the
time-split C-index and the calibration of predicted days. That effect has not been tested.

**Districts.** Vaishali: 14,805 cases, 94.4% pending, flat by filing year (0.955, 0.916, 0.944, 0.958); its recorded
decisions fall only in 2018 (304), 2019 (452) and 2020 (74); only 0.4% of its pending cases are stale, so it is not a
clear recording artifact. Gaya is similar (0.97, flat). Districts with a pending share around 95% or higher are
treated as unreliable.
