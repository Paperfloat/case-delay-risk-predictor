import pandas as pd

df = pd.read_csv('data/processed/cases_2012_delhi_clean.csv', dtype=str)

# --- Decode categorical ID fields using key files ---
type_key = pd.read_csv('data/raw/keys/type_name_key.csv', dtype=str)[['year', 'type_name', 'type_name_s']]
purpose_key = pd.read_csv('data/raw/keys/purpose_name_key.csv', dtype=str)[['year', 'purpose_name', 'purpose_name_s']]
disp_key = pd.read_csv('data/raw/keys/disp_name_key.csv', dtype=str)[['year', 'disp_name', 'disp_name_s']]

df = df.merge(type_key, on=['year', 'type_name'], how='left')
df = df.merge(purpose_key, on=['year', 'purpose_name'], how='left')
df = df.merge(disp_key, on=['year', 'disp_name'], how='left')

# NOTE: cases_district_key.csv only has an entry for Delhi (state_code 26) in year 2010,
# not 2012+. District boundaries are stable, so we reuse the 2010 mapping for all years
# by joining on state_code + dist_code only (dropping year from this specific join).
district_key = pd.read_csv('data/raw/keys/cases_district_key.csv', dtype=str)
district_key = district_key[district_key['year'] == '2010'][['state_code', 'dist_code', 'district_name']]
court_key = pd.read_csv('data/raw/keys/cases_court_key.csv', dtype=str)[['year', 'state_code', 'dist_code', 'court_no', 'court_name']]

df = df.merge(district_key, on=['state_code', 'dist_code'], how='left')
df = df.merge(court_key, on=['year', 'state_code', 'dist_code', 'court_no'], how='left')

print("Unmatched district_name:", df['district_name'].isna().sum())
print("Unmatched court_name:", df['court_name'].isna().sum())
df['purpose_name_s'] = df['purpose_name_s'].fillna('unknown')

print("Unmatched type_name:", df['type_name_s'].isna().sum())
print("Unmatched purpose_name:", df['purpose_name_s'].isna().sum())
print("Unmatched disp_name:", df['disp_name_s'].isna().sum())

# --- Build target variable ---

df['date_of_filing'] = pd.to_datetime(df['date_of_filing'], format='%Y-%m-%d', errors='coerce')
df['date_of_decision'] = pd.to_datetime(df['date_of_decision'], format='%Y-%m-%d', errors='coerce')

# Reject implausible years (data entry corruption, e.g. "1204" instead of "2014")
df.loc[(df['date_of_filing'].dt.year < 2005) | (df['date_of_filing'].dt.year > 2020), 'date_of_filing'] = pd.NaT
df.loc[(df['date_of_decision'].dt.year < 2005) | (df['date_of_decision'].dt.year > 2020), 'date_of_decision'] = pd.NaT

df['days_to_disposition'] = (df['date_of_decision'] - df['date_of_filing']).dt.days

print("Negative days_to_disposition:", (df['days_to_disposition'] < 0).sum())
df.loc[df['days_to_disposition'] < 0, 'days_to_disposition'] = pd.NA

print("NaT in date_of_filing:", df['date_of_filing'].isna().sum())
print("NaT in date_of_decision:", df['date_of_decision'].isna().sum())

# --- Derive hearing-activity proxy ---
df['multiple_hearings'] = (df['date_first_list'] != df['date_last_list']).astype(int)

df.to_csv('data/processed/cases_2012_delhi_features.csv', index=False)
print(df.shape)
print(df['days_to_disposition'].describe())
