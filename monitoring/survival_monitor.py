import argparse
import html as H
import json
import sys

import numpy as np
import pandas as pd

SLUGS = ["delhi", "orissa", "bihar"]
PSI_THRESHOLD = 0.1
EPS = 1e-4
SAMPLE = 50000
DIMS = ["court_tier", "district_name", "type_name_normalized"]

ap = argparse.ArgumentParser()
ap.add_argument("--no-evidently", action="store_true", help="skip the Evidently HTML reports")
ap.add_argument("--exit-code", action="store_true", help="exit with code 10 if any state needs a retrain")
ap.add_argument("--ref-max-year", type=int, default=2011, help="cases filed up to this year = reference")
args = ap.parse_args()


def psi(ref, cur):
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    return float(((c - r) * np.log(c / r)).sum())


def table(df):
    return df.to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:.3f}")


status, sections = {}, []
for slug in SLUGS:
    meta = json.load(open(f"models/{slug}_aft_meta.json"))
    cats, cutoff = meta["categorical_fields"], meta["cutoff"]
    path = f"data/processed/cases_2010_2013_{slug}_survival_normalized.csv.gz"
    df = pd.read_csv(path, dtype=str, usecols=cats + ["date_of_filing"])
    for c in cats:
        df[c] = df[c].fillna("missing")
    year = pd.to_datetime(df["date_of_filing"], errors="coerce").dt.year
    ref, cur = df[year <= args.ref_max_year], df[year > args.ref_max_year]
    print(f"\n===== {slug} | filing years: {year.value_counts().sort_index().to_dict()}")
    print(f"reference rows {len(ref)} | current rows {len(cur)}")
    if len(ref) < 1000 or len(cur) < 1000:
        sys.exit(f"{slug}: reference or current slice is too small - paste this output and stop")

    psi_df = pd.DataFrame({"feature": cats, "psi": [psi(ref[c], cur[c]) for c in cats]})
    psi_df["status"] = np.where(psi_df.psi > PSI_THRESHOLD, "DRIFT", "stable")
    drifted = psi_df[psi_df.psi > PSI_THRESHOLD].feature.tolist()
    print(psi_df.to_string(index=False))
    print(">>> RETRAIN TRIGGERED:" if drifted else ">>> No retrain needed.", drifted)
    status[slug] = {"retrain": bool(drifted), "drifted_features": drifted,
                    "max_psi": round(float(psi_df.psi.max()), 4), "threshold": PSI_THRESHOLD,
                    "reference_rows": int(len(ref)), "current_rows": int(len(cur))}

    ev_file = f"monitoring/survival_drift_{slug}.html"
    if not args.no_evidently:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
            n = min(SAMPLE, len(ref), len(cur))
            rep = Report([DataDriftPreset(method="psi")], include_tests=True)
            res = rep.run(current_data=cur[cats].sample(n, random_state=42),
                          reference_data=ref[cats].sample(n, random_state=42))
            res.save_html(ev_file)
            print("Saved", ev_file)
        except Exception as e:
            print("Evidently report skipped:", e)

    cidx = pd.read_csv(f"reports/{slug}_survival_fairness_{cutoff}.csv")
    tier = pd.read_csv(f"reports/{slug}_survival_tier_report_{cutoff}_auto.csv")
    overall = tier[tier.dimension == "overall"].iloc[0]
    gap_rows, worst = [], []
    for d in DIMS:
        c_s, t_s = cidx[cidx.dimension == d], tier[tier.dimension == d]
        gap_rows.append((d, len(t_s), c_s.c_index.max() - c_s.c_index.min(),
                         t_s.macro_f1.max() - t_s.macro_f1.min()))
        worst.append(t_s.sort_values("macro_f1").head(3)[["dimension", "group", "n", "macro_f1"]])
    gaps_df = pd.DataFrame(gap_rows, columns=["dimension", "groups", "c_index_gap", "macro_f1_gap"])
    gaps_df["meets_target_<0.10"] = np.where(gaps_df.macro_f1_gap < 0.10, "yes", "NO")

    badge = "RETRAIN TRIGGERED" if drifted else "no retrain needed"
    sections.append(f"""
<h2>{H.escape(slug)} <span class="b {'bad' if drifted else 'ok'}">{badge}</span></h2>
<p>Model cutoff {cutoff} | test C-index <b>{meta['test_c_index']:.3f}</b> |
Macro precision {overall.macro_precision:.3f} | recall {overall.macro_recall:.3f} | F1 <b>{overall.macro_f1:.3f}</b></p>
<h3>Drift (PSI, threshold {PSI_THRESHOLD}; reference = filed {args.ref_max_year} or earlier, current = later)</h3>
{table(psi_df)}
<p><a href="survival_drift_{slug}.html">Open the Evidently drift report</a> (50,000-row sample)</p>
<h3>Fairness gaps (best minus worst subgroup, groups with n &gt;= 1000)</h3>
{table(gaps_df)}
<h3>Three weakest subgroups per dimension (Macro F1)</h3>
{table(pd.concat(worst))}
""")

json.dump(status, open("monitoring/survival_drift_status.json", "w"), indent=2)
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Survival model monitoring</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222}}
.t{{border-collapse:collapse;margin:.5rem 0 1rem}}.t th,.t td{{padding:4px 12px;text-align:left;border-bottom:1px solid #ddd}}
.b{{font-size:.6em;padding:3px 8px;border-radius:6px;vertical-align:middle}}.bad{{background:#fde2e2;color:#a01010}}
.ok{{background:#dff5e1;color:#146c2e}}</style></head><body>
<h1>Survival models: drift and fairness</h1>
<p>Drift compares older and newer filing years. Fairness numbers come from the held-out test split.
Tiers are tertiles within each state. Pending cases count as High only when already past the High cutoff.</p>
{''.join(sections)}</body></html>"""
open("monitoring/survival_dashboard.html", "w").write(page)
print("\nSaved monitoring/survival_dashboard.html and monitoring/survival_drift_status.json")
if args.exit_code and any(v["retrain"] for v in status.values()):
    sys.exit(10)
