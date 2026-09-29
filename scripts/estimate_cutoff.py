import pandas as pd

PATH = "data/processed/cases_2010_2013_orissa_features.csv"

cols = pd.read_csv(PATH, nrows=0).columns.tolist()
date_cols = [c for c in cols if "date" in c.lower()]
print("Date-like columns:", date_cols)

FILE_COL = next((c for c in cols if "fil" in c.lower() and "date" in c.lower()), None)
DEC_COL = next((c for c in cols if "decision" in c.lower() and "date" in c.lower()), None)
print("Using filing:", FILE_COL, "| decision:", DEC_COL)
assert FILE_COL and DEC_COL, "Pick the columns manually from the list above"

df = pd.read_csv(PATH, usecols=[FILE_COL, DEC_COL])
df[FILE_COL] = pd.to_datetime(df[FILE_COL], errors="coerce")
df[DEC_COL] = pd.to_datetime(df[DEC_COL], errors="coerce")

# drop corrupted years (same plausible-year rule as Delhi)
ok = df[DEC_COL].dt.year.between(2005, 2025) | df[DEC_COL].isna()
df = df[ok]

dec = df[DEC_COL].dropna()
print("\nRows:", len(df), "| pending:", df[DEC_COL].isna().sum(),
      f"({df[DEC_COL].isna().mean():.1%})")
print("Max decision date:", dec.max())
print("99.9th pct decision date:", dec.quantile(0.999))

monthly = dec.dt.to_period("M").value_counts().sort_index()
print("\nDecisions per month (tail):")
print(monthly.tail(24))

median_vol = monthly.median()
active = monthly[monthly >= 0.10 * median_vol]
print("\nMedian monthly volume:", median_vol)
print("Estimated cutoff (last active month):", active.index.max())

print("\nPending rate by filing year:")
print(df.groupby(df[FILE_COL].dt.year)[DEC_COL].apply(lambda s: s.isna().mean()))
