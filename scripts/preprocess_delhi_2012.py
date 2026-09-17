import pandas as pd

df = pd.read_csv('data/raw/cases/cases_2012_delhi.csv', dtype=str)

# Drop censored (still-pending) rows for v1
df = df[df['date_of_decision'].notna()].copy()

# Impute purpose_name nulls with explicit 'unknown' category


df.to_csv('data/processed/cases_2012_delhi_clean.csv', index=False)
print(df.shape)
