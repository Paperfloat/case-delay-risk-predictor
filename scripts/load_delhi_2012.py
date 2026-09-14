import pandas as pd

chunks = []
for chunk in pd.read_csv('data/raw/cases/cases_2012.csv', dtype=str, chunksize=200_000):
    filtered = chunk[chunk['state_code'] == '26']
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
df.to_csv('data/raw/cases/cases_2012_delhi.csv', index=False)

print(df.shape)
print(df.isna().sum())
print(df['female_defendant'].value_counts())
print((df['date_of_decision'] < df['date_of_filing']).sum())
print((df['date_last_list'] == '5000-01-01').sum())
