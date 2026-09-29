import os
import pandas as pd
from common import get_args, load_config

args = get_args()
cfg = load_config(args)
out_dir = args.out_dir or "data/raw/cases"
os.makedirs(out_dir, exist_ok=True)

for year in cfg["years"]:
    print(f"\n--- Processing {year} ---")
    chunks = []
    for chunk in pd.read_csv(f"data/raw/cases/cases_{year}.csv", dtype=str, chunksize=200_000):
        chunks.append(chunk[chunk["state_code"] == cfg["code"]])
    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(f"{out_dir}/cases_{year}_{cfg['slug']}.csv", index=False)
    print(f"{year}: {df.shape[0]} rows")
