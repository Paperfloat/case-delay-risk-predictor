import pandas as pd

YEARS = [2010, 2011, 2012, 2013]

for year in YEARS:
    df = pd.read_csv(f'data/raw/cases/cases_{year}_delhi.csv', dtype=str)

    # Drop censored (still-pending) rows for v1 — same rule as 2012
    df = df[df['date_of_decision'].notna()].copy()

    df.to_csv(f'data/processed/cases_{year}_delhi_clean.csv', index=False)
    print(f"{year}: {df.shape[0]} rows after dropping censored cases")

