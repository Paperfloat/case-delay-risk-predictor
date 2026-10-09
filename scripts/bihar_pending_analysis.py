import numpy as np
import pandas as pd

PATH = "data/processed/cases_2010_2013_bihar_survival_normalized.csv.gz"
cols = pd.read_csv(PATH, nrows=0).columns.tolist()
last_col = "date_last_list_clean" if "date_last_list_clean" in cols else "date_last_list"
want = ["ddl_case_id", "date_of_filing", "date_of_decision", "event", last_col,
        "type_name_normalized", "district_name", "court_tier"]
missing = [c for c in want if c not in cols]
assert not missing, f"columns not found: {missing}; available: {cols}"
df = pd.read_csv(PATH, dtype=str, usecols=want)
df["filed"] = pd.to_datetime(df["date_of_filing"], errors="coerce")
df["decided_on"] = pd.to_datetime(df["date_of_decision"], errors="coerce")
df["last_list"] = pd.to_datetime(df[last_col], errors="coerce")
df = df[df["filed"].notna()].copy()
df["year"] = df["filed"].dt.year
df["pending"] = (df["event"] != "1").astype(int)
CUTOFF = pd.Timestamp("2019-05-31")
STALE = CUTOFF - pd.DateOffset(years=3)
df["stale"] = (df["pending"] == 1) & (df["last_list"] < STALE)
YEARS = [2010, 2011, 2012, 2013]
print(f"rows {len(df)} | last-listing column used: {last_col}")

raw = df.groupby("year")["pending"].mean()
print("\n1. Pending share by filing year (the puzzle):", {int(y): round(float(v), 3) for y, v in raw.items()})


def standardized(dim, min_n=200):
    t = df.pivot_table(index=dim, columns="year", values="pending", aggfunc=["mean", "size"])
    ok = t["size"].min(axis=1) >= min_n
    w = df[df.year == 2010][dim].value_counts(normalize=True).reindex(t.index[ok]).fillna(0)
    w = w / w.sum()
    return {int(y): round(float((t["mean"].loc[ok, y] * w).sum()), 3) for y in YEARS}, int(ok.sum()), len(t)


print("\n2. Pending share if the 2010 mix were held fixed (if the drop disappears, mix explains it):")
for dim in ["type_name_normalized", "district_name", "court_tier"]:
    res, used, total = standardized(dim)
    print(f"   {dim:22s} {res}  (used {used} of {total} groups)")

print("\n3. Among PENDING cases, share whose last hearing was over 3 years before the cutoff (likely unrecorded decisions):")
p = df[df.pending == 1]
print("  ", {int(y): round(float(v), 3) for y, v in p.groupby("year")["stale"].mean().items()})
print("   stale cases as a share of ALL cases filed that year:",
      {int(y): round(float(v), 3) for y, v in df.groupby("year")["stale"].mean().items()})

d = df[(df.pending == 0) & df.decided_on.notna()].copy()
d["days"] = (d["decided_on"] - d["filed"]).dt.days
print("\n4. Decided cases: median days to decision, and share decided within 1 year, by filing year:")
print("   median days:", {int(y): int(v) for y, v in d.groupby("year")["days"].median().items()})
print("   within 1 year:", {int(y): round(float(v), 3) for y, v in (d["days"] <= 365).groupby(d["year"]).mean().items()})
print("   share of ALL cases decided within 1 year:",
      {int(y): round(float(v), 3) for y, v in df.assign(q=(df.pending == 0) & ((df.decided_on - df.filed).dt.days <= 365))
       .groupby("year")["q"].mean().items()})

print("\n5. Districts with the largest drop in pending share 2010 -> 2013 (n >= 500 each year):")
t = df.pivot_table(index="district_name", columns="year", values="pending", aggfunc=["mean", "size"])
ok = t["size"].min(axis=1) >= 500
chg = (t["mean"].loc[ok, 2013] - t["mean"].loc[ok, 2010]).sort_values()
for name in list(chg.index[:6]) + ["..."] + list(chg.index[-4:]):
    if name == "...":
        print("   ...")
        continue
    print(f"   {name:22s} 2010 {t['mean'].loc[name, 2010]:.2f} -> 2013 {t['mean'].loc[name, 2013]:.2f} "
          f"(change {chg[name]:+.2f})")

print("\n6. Vaishali:")
v = df[df.district_name.str.lower().str.contains("vaishali", na=False)]
print(f"   cases {len(v)} | pending share overall {v.pending.mean():.3f} (Bihar {df.pending.mean():.3f})")
print("   pending share by filing year:", {int(y): round(float(x), 3) for y, x in v.groupby("year")["pending"].mean().items()})
vp = v[v.pending == 1]
print(f"   of its pending cases, stale (last hearing over 3 years before cutoff): {vp.stale.mean():.1%} "
      f"(Bihar: {p.stale.mean():.1%})")
print(f"   of its pending cases, with a last-hearing date at all: {vp.last_list.notna().mean():.1%} "
      f"(Bihar: {p.last_list.notna().mean():.1%})")
print("   its most common case types:", v.type_name_normalized.value_counts().head(5).to_dict())
dec = v[v.pending == 0]["decided_on"].dt.year.value_counts().sort_index().to_dict()
print("   decisions by calendar year:", {int(k): int(x) for k, x in dec.items()})
oth = df[(df.pending == 0) & ~df.index.isin(v.index)]["decided_on"].dt.year.value_counts(normalize=True).sort_index()
print("   rest of Bihar, share of decisions by year:", {int(k): round(float(x), 3) for k, x in oth.items()})
vs = pd.Series(dec)
print("   Vaishali share of decisions by year:", {int(k): round(float(x), 3) for k, x in (vs / vs.sum()).items()})
