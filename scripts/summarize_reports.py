import glob
import json
import os

import pandas as pd

print("=== MODELS")
for path in sorted(glob.glob("models/*_aft_meta.json")):
    m = json.load(open(path))
    t = m["tier_thresholds"]
    print(f"{m['model_version']}: test C-index {m['test_c_index']}, rows {m['n_rows']}, "
          f"censored {m['censored_rate']:.1%}, best iteration {m['best_iteration']}, "
          f"tier cut points {t['low_max_days']} / {t['medium_max_days']} days")

print("\n=== FAIRNESS REPORTS (groups with at least 1,000 test cases; 95% bootstrap intervals)")
for path in sorted(glob.glob("reports/*_survival_fairness_*.csv")):
    name = os.path.basename(path).replace("_survival_fairness_", " @ ").replace(".csv", "")
    t = pd.read_csv(path)
    if "ci_low" not in t.columns:
        print(f"\n{name}: no confidence intervals in this report (old format), skipped")
        continue
    print(f"\n--- {name}")
    for dim, g in t.groupby("dimension"):
        g = g.sort_values("c_index")
        w, b = g.iloc[0], g.iloc[-1]
        separated = w.ci_high < b.ci_low
        print(f"{dim}: {len(g)} groups, gap {b.c_index - w.c_index:.3f}, "
              f"intervals {'do not overlap' if separated else 'overlap'}")
        print(f"   worst {w.group}: C {w.c_index:.3f} [{w.ci_low:.3f}-{w.ci_high:.3f}] n={int(w.n)} censored {w.censored:.0%}")
        print(f"   best  {b.group}: C {b.c_index:.3f} [{b.ci_low:.3f}-{b.ci_high:.3f}] n={int(b.n)} censored {b.censored:.0%}")
    print("court tiers in full:")
    tier = t[t.dimension == "court_tier"].sort_values("c_index")
    for _, r in tier.iterrows():
        print(f"   {r.group}: C {r.c_index:.3f} [{r.ci_low:.3f}-{r.ci_high:.3f}] n={int(r.n)} censored {r.censored:.0%}")
