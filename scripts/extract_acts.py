import tarfile

import pandas as pd

SLUGS = ["delhi", "orissa", "bihar"]
YEARS = [2010, 2011, 2012, 2013]

state_of = {}
for slug in SLUGS:
    for y in YEARS:
        ids = pd.read_csv(f"data/raw/cases/cases_{y}_{slug}.csv", dtype=str, usecols=["ddl_case_id"])["ddl_case_id"]
        state_of.update(dict.fromkeys(ids, slug))
print(f"cases in the three states: {len(state_of)}")
idset = set(state_of)

tf = tarfile.open("data/raw/acts_sections.tar.gz", "r:gz")
f = tf.extractfile("acts_sections.csv")
parts, seen = [], 0
for chunk in pd.read_csv(f, dtype=str, chunksize=2_000_000):
    seen += len(chunk)
    parts.append(chunk[chunk["ddl_case_id"].isin(idset)])
    print(f"read {seen:,} rows | kept so far {sum(len(p) for p in parts):,}", flush=True)
a = pd.concat(parts, ignore_index=True)
print(f"act/section rows for our cases: {len(a):,}")

for c in ["bailable_ipc", "number_sections_ipc", "criminal"]:
    a[c] = pd.to_numeric(a[c], errors="coerce")
act_name = pd.read_csv("data/raw/keys/act_key.csv", dtype=str).drop_duplicates("act").set_index("act")["act_s"]
sec_name = pd.read_csv("data/raw/keys/section_key.csv", dtype=str).drop_duplicates("section").set_index("section")["section_s"]

g = a.groupby("ddl_case_id", sort=False)
case = pd.DataFrame({
    "n_act_rows": g.size(),
    "n_distinct_acts": g["act"].nunique(),
    "criminal_act": g["criminal"].max(),
    "bailable_ipc": g["bailable_ipc"].max(),
    "ipc_sections": g["number_sections_ipc"].max(),
    "primary_act_id": g["act"].first(),
    "primary_section_id": g["section"].first(),
}).reset_index()
case["primary_act"] = case["primary_act_id"].map(act_name).fillna("missing")
case["primary_section"] = case["primary_section_id"].map(sec_name).fillna("missing")
case["state"] = case["ddl_case_id"].map(state_of)
case.to_csv("data/processed/acts_case_level.csv.gz", index=False)
print("Saved data/processed/acts_case_level.csv.gz", case.shape)

for slug in SLUGS:
    n_state = sum(1 for v in state_of.values() if v == slug)
    c = case[case.state == slug]
    print(f"\n===== {slug}: {len(c)} of {n_state} cases have act rows ({len(c) / n_state:.1%})")
    print(f"criminal share among those: {c.criminal_act.mean():.1%} | with a first section: {(c.primary_section != 'missing').mean():.1%}")
    print("top 8 first-listed acts:")
    print(c.primary_act.value_counts().head(8).to_string())

for slug in ["delhi", "orissa", "bihar"]:
    f = f"data/processed/cases_2010_2013_{slug}_survival_normalized.csv.gz"
    cols = pd.read_csv(f, nrows=0).columns.tolist()
    print(f"\n{slug} normalized file keeps ddl_case_id: {'ddl_case_id' in cols}")
