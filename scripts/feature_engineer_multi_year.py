import pandas as pd

YEARS = [2010, 2011, 2012, 2013]

type_key = pd.read_csv('data/raw/keys/type_name_key.csv', dtype=str)[['year', 'type_name', 'type_name_s']]
purpose_key = pd.read_csv('data/raw/keys/purpose_name_key.csv', dtype=str)[['year', 'purpose_name', 'purpose_name_s']]
disp_key = pd.read_csv('data/raw/keys/disp_name_key.csv', dtype=str)[['year', 'disp_name', 'disp_name_s']]

district_key = pd.read_csv('data/raw/keys/cases_district_key.csv', dtype=str)
district_key = district_key[district_key['year'] == '2010'][['state_code', 'dist_code', 'district_name']]

court_key = pd.read_csv('data/raw/keys/cases_court_key.csv', dtype=str)[['year', 'state_code', 'dist_code', 'court_no', 'court_name']]

manual_merges = {'m a c t': 'mact'}

all_years = []

for year in YEARS:
    print(f"\n--- Processing {year} ---")
    df = pd.read_csv(f'data/processed/cases_{year}_delhi_clean.csv', dtype=str)

    df = df.merge(type_key, on=['year', 'type_name'], how='left')
    df = df.merge(purpose_key, on=['year', 'purpose_name'], how='left')
    df = df.merge(disp_key, on=['year', 'disp_name'], how='left')
    df = df.merge(district_key, on=['state_code', 'dist_code'], how='left')
    df = df.merge(court_key, on=['year', 'state_code', 'dist_code', 'court_no'], how='left')

    print("Unmatched type_name:", df['type_name_s'].isna().sum())
    print("Unmatched district_name:", df['district_name'].isna().sum())
    print("Unmatched court_name:", df['court_name'].isna().sum())

    df['purpose_name_s'] = df['purpose_name_s'].fillna('unknown')

    df['date_of_filing'] = pd.to_datetime(df['date_of_filing'], format='%Y-%m-%d', errors='coerce')
    df['date_of_decision'] = pd.to_datetime(df['date_of_decision'], format='%Y-%m-%d', errors='coerce')
    df.loc[(df['date_of_filing'].dt.year < 2005) | (df['date_of_filing'].dt.year > 2020), 'date_of_filing'] = pd.NaT
    df.loc[(df['date_of_decision'].dt.year < 2005) | (df['date_of_decision'].dt.year > 2020), 'date_of_decision'] = pd.NaT
    df['days_to_disposition'] = (df['date_of_decision'] - df['date_of_filing']).dt.days
    df.loc[df['days_to_disposition'] < 0, 'days_to_disposition'] = pd.NA

    df['date_last_list_clean'] = df['date_last_list'].where(df['date_last_list'] != '5000-01-01', pd.NA)
    df['multiple_hearings'] = (df['date_first_list'] != df['date_last_list_clean']).astype('Int64')
    df.loc[df['date_last_list_clean'].isna(), 'multiple_hearings'] = pd.NA

    df['court_tier'] = df['court_name'].str.split(',').str[0].str.strip()

    df['type_name_normalized'] = (
        df['type_name_s']
        .str.replace('.', '', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df['type_name_normalized'] = df['type_name_normalized'].replace(manual_merges)

    all_years.append(df)

print(f"\nAll years processed. Number of dataframes collected: {len(all_years)}")
for i, d in enumerate(all_years):
    print(f"  DataFrame {i}: {d.shape}")

print("Concatenating...")
combined = pd.concat(all_years, ignore_index=True)
combined.to_csv('data/processed/cases_2010_2013_delhi_features.csv', index=False)
print("Saved successfully.")
print(f"\nCombined shape: {combined.shape}")
print(combined['year'].value_counts().sort_index())
