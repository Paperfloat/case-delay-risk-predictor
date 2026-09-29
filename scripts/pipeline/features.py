import os
import pandas as pd
from common import get_args, load_config, features_name

args = get_args()
cfg = load_config(args)
state_code, slug, YEARS = cfg["code"], cfg["slug"], cfg["years"]
out_dir = args.out_dir or "data/processed"
os.makedirs(out_dir, exist_ok=True)
print(f"State {cfg['state']} (code {state_code}, slug {slug}) | keep_pending={cfg['keep_pending']}")

type_key = pd.read_csv('data/raw/keys/type_name_key.csv', dtype=str)
type_key = type_key[['year', 'type_name', 'type_name_s']]
type_key = type_key.drop_duplicates(['year', 'type_name'])

purpose_key = pd.read_csv('data/raw/keys/purpose_name_key.csv', dtype=str)
purpose_key = purpose_key[['year', 'purpose_name', 'purpose_name_s']]
purpose_key = purpose_key.drop_duplicates(['year', 'purpose_name'])

disp_key = pd.read_csv('data/raw/keys/disp_name_key.csv', dtype=str)
disp_key = disp_key[['year', 'disp_name', 'disp_name_s']]
disp_key = disp_key.drop_duplicates(['year', 'disp_name'])

district_key = pd.read_csv('data/raw/keys/cases_district_key.csv', dtype=str)
district_key = district_key[district_key['state_code'] == state_code]
print("District key years available:", sorted(district_key['year'].unique()))
district_key = district_key.sort_values('year')
district_key = district_key.drop_duplicates(['state_code', 'dist_code'], keep='first')
district_key = district_key[['state_code', 'dist_code', 'district_name']]

court_key = pd.read_csv('data/raw/keys/cases_court_key.csv', dtype=str)
court_key = court_key[['year', 'state_code', 'dist_code', 'court_no', 'court_name']]
court_key = court_key.drop_duplicates(['year', 'state_code', 'dist_code', 'court_no'])

all_years = []

for year in YEARS:
    print(f"\n--- Processing {year} ---")
    df = pd.read_csv(f'data/raw/cases/cases_{year}_{slug}.csv', dtype=str)
    total = len(df)
    if cfg['keep_pending']:
        df['event'] = df['date_of_decision'].notna().astype(int)  # 1 = decided, 0 = pending (censored)
        n = len(df)
        print(f"Kept {(df['event'] == 0).sum()} pending rows ({(df['event'] == 0).mean():.1%}) as censored")
    else:
        df = df[df['date_of_decision'].notna()].copy()
        n = len(df)
        print(f"Dropped {total - n} censored rows ({(total - n) / total:.1%})")

    df = df.merge(type_key, on=['year', 'type_name'], how='left')
    df = df.merge(purpose_key, on=['year', 'purpose_name'], how='left')
    df = df.merge(disp_key, on=['year', 'disp_name'], how='left')
    df = df.merge(district_key, on=['state_code', 'dist_code'], how='left')
    court_join = ['year', 'state_code', 'dist_code', 'court_no']
    df = df.merge(court_key, on=court_join, how='left')
    assert len(df) == n, f"Row count changed during merges: {n} -> {len(df)}"

    print("Unmatched type_name:", df['type_name_s'].isna().sum())
    print("Unmatched district_name:", df['district_name'].isna().sum())
    print("Unmatched court_name:", df['court_name'].isna().sum())

    df['purpose_name_s'] = df['purpose_name_s'].fillna('unknown')

    fmt = '%Y-%m-%d'
    df['date_of_filing'] = pd.to_datetime(df['date_of_filing'], format=fmt, errors='coerce')
    df['date_of_decision'] = pd.to_datetime(df['date_of_decision'], format=fmt, errors='coerce')
    for c in ['date_of_filing', 'date_of_decision']:
        bad = (df[c].dt.year < 2005) | (df[c].dt.year > 2020)
        df.loc[bad, c] = pd.NaT
    df['days_to_disposition'] = (df['date_of_decision'] - df['date_of_filing']).dt.days
    df.loc[df['days_to_disposition'] < 0, 'days_to_disposition'] = pd.NA

    if cfg['keep_pending']:
        bad_decided = (df['event'] == 1) & df['days_to_disposition'].isna()
        print(f"Dropped {bad_decided.sum()} decided rows with invalid duration")
        df = df[~bad_decided].copy()

    last = df['date_last_list'].where(df['date_last_list'] != '5000-01-01', pd.NA)
    df['date_last_list_clean'] = last
    df['multiple_hearings'] = (df['date_first_list'] != last).astype('Int64')
    df.loc[last.isna(), 'multiple_hearings'] = pd.NA

    df['court_tier'] = df['court_name'].str.split(',').str[0].str.strip()

    norm = df['type_name_s'].str.replace('.', '', regex=False)
    norm = norm.str.replace(r'\s+', ' ', regex=True).str.strip()
    df['type_name_normalized'] = norm.replace({'m a c t': 'mact'})

    all_years.append(df)

combined = pd.concat(all_years, ignore_index=True)
out = f"{out_dir}/{features_name(cfg)}"
combined.to_csv(out, index=False)
print(f"\nSaved {out}")
print(f"Combined shape: {combined.shape}")
print(combined['year'].value_counts().sort_index())
print("\nCourt tier values:")
print(combined['court_tier'].value_counts().head(15))
print("\nUnique case types:", combined['type_name_normalized'].nunique())
print("Valid targets:", combined['days_to_disposition'].notna().sum())
