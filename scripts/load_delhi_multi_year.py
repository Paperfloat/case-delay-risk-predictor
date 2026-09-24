import pandas as pd

YEARS = [2010, 2011, 2012, 2013]

for year in YEARS:
    print(f"\n--- Processing {year} ---")
    chunks = []
    for chunk in pd.read_csv(f'data/raw/cases/cases_{year}.csv', dtype=str, chunksize=200_000):
        filtered = chunk[chunk['state_code'] == '26']
        chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(f'data/raw/cases/cases_{year}_delhi.csv', index=False)
    print(f"{year}: {df.shape[0]} rows")
