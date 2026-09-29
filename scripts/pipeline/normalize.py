import os
import re
import sys
import pandas as pd
from common import get_args, load_config, features_name, normalized_name

args = get_args()
cfg = load_config(args)
if not cfg.get("normalize"):
    print(f"normalize=false for {cfg['state']}, nothing to do")
    sys.exit(0)

TOP_TYPES = cfg.get("top_case_types", 100)
out_dir = args.out_dir or "data/processed"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(f"{args.in_dir}/{features_name(cfg)}", dtype=str)


def clean(text):
    s = str(text).lower().replace('.', '')
    s = re.sub(r'[()&,\-]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def tier_of(name):
    s = clean(name)
    if 'civil' in s and 'jmfc' in s:
        return 'Civil Judge cum JMFC'
    if 'civil' in s and 'sdjm' in s:
        return 'Civil Judge cum SDJM'
    if 'sdjm' in s or 'sub divisional' in s:
        return 'Sub-Divisional Judicial Magistrate'
    if 'jmfc' in s or 'magistrate first class' in s:
        return 'Judicial Magistrate First Class'
    if 'acjm' in s or 'chief judicial magistrate' in s:
        return 'Chief Judicial Magistrate'
    if 'civil judge' in s or 'cjsd' in s or 'senior civil' in s:
        if 'junior' in s or re.search(r'\bjd\b', s):
            return 'Civil Judge (Junior Division)'
        if re.search(r'senior|sr|\bsd\b|cjsd', s):
            return 'Civil Judge (Senior Division)'
        return 'Civil Judge (division unclear)'
    if 'additional district' in s or 'addl district' in s:
        return 'Additional District / Sessions Judge'
    if 'session' in s or 'district judge' in s:
        return 'District and Sessions Judge'
    return 'Other / unclear'


df['court_tier_raw'] = df['court_tier']
df['court_tier'] = df['court_tier_raw'].apply(tier_of)

top = df['type_name_normalized'].value_counts().head(TOP_TYPES).index
df['type_name_raw_norm'] = df['type_name_normalized']
df['type_name_normalized'] = df['type_name_normalized'].where(
    df['type_name_normalized'].isin(top), 'other'
)

print(df['court_tier'].value_counts().to_string())
unclear = df[df['court_tier'] == 'Other / unclear']['court_tier_raw'].value_counts()
print(f"\nUnclear tier: {unclear.sum()} rows ({unclear.sum() / len(df):.1%})")
print(unclear.head(15).to_string())
print("\nCase types kept:", len(top))
print("Rows grouped as 'other':", (df['type_name_normalized'] == 'other').sum())

out = f"{out_dir}/{normalized_name(cfg)}"
df.to_csv(out, index=False)
print(f"\nSaved {out}")
