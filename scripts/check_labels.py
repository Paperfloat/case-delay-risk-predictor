import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))
import labels

F = ["female_defendant", "female_petitioner", "female_adv_def", "female_adv_pet"]
for state, slug in [("odisha", "orissa"), ("bihar", "bihar")]:
    cols = json.load(open(f"models/{slug}_aft_feature_columns.json"))
    types = [c[len("type_name_normalized_"):] for c in cols if c.startswith("type_name_normalized_")]
    d = pd.read_csv(f"data/processed/cases_2010_2013_{slug}_survival_normalized.csv.gz", dtype=str,
                    usecols=["type_name_normalized"] + F)
    vc = d["type_name_normalized"].fillna("missing").value_counts()
    covered = sum(vc.get(t, 0) for t in types if labels.has_label(state, t))
    total = sum(vc.get(t, 0) for t in types)
    print(f"\n===== {state}: {len(types)} case types | with a real label: {sum(labels.has_label(state, t) for t in types)} "
          f"| cases covered: {covered / total:.1%}")
    print("top 20 most common types and their labels:")
    for t in sorted(types, key=lambda t: -vc.get(t, 0))[:20]:
        print(f"   {t!r:34} -> {labels.case_type_label(state, t)}")
    miss = [t for t in sorted(types, key=lambda t: -vc.get(t, 0)) if not labels.has_label(state, t)][:25]
    print("most common codes WITHOUT a label:", [f"{t} ({vc.get(t, 0)})" for t in miss])
    for f in F:
        vals = [c[len(f) + 1:] for c in cols if c.startswith(f + "_")]
        print(f"   {f}: " + " | ".join(f"{v!r} -> {labels.gender_label(v)}" for v in vals))
