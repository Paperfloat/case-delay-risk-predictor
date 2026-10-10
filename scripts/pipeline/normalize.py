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


def tier_of_bihar(name):
    s = clean(name)
    t = s.replace(' ', '')
    if 'cjm' in t or 'chiefjudicial' in t or 'cheifjudicial' in t:
        return 'Chief Judicial Magistrate'
    if 'additionaldistrict' in t or 'addl' in t:
        return 'Additional District / Sessions Judge'
    if 'sess' in t or 'districtjudge' in t or t.startswith('dj') or 'djdiv' in t or 'jdivision' in t and t.startswith('nau'):
        return 'District and Sessions Judge'
    if 'civil' in t or 'cjsd' in t:
        if 'junior' in t or 'jr' in t:
            return 'Civil Judge (Junior Division)'
        if 'senior' in t or 'sr' in t or 'cjsd' in t:
            return 'Civil Judge (Senior Division)'
        return 'Civil Judge (division unclear)'
    return 'Other / unclear'


def tier_of_delhi(name):
    t = clean(name).replace(' ', '')
    if 'polc' in t or 'poit' in t:
        return 'Labour / Industrial Tribunal'
    if 'family' in t:
        return 'Family Court'
    if 'chiefmetropolitan' in t or 'metropolitanmagistrate' in t:
        return 'Chief Judicial Magistrate'
    if 'sess' in t or 'districtjudge' in t:
        return 'District and Sessions Judge'
    if 'seniorcivil' in t or 'civiljudge' in t:
        return 'Civil Judge (Senior Division)'
    return 'Other / unclear'


RULES = {'bihar': tier_of_bihar, 'delhi': tier_of_delhi}
rule = RULES.get(cfg.get('tier_ruleset'), tier_of)
print("Court tier rule set:", cfg.get('tier_ruleset', 'base'))
df['court_tier_raw'] = df['court_tier']
df['court_tier'] = df['court_tier_raw'].apply(rule)

SPELLING = [('execuition', 'execution'), ('mislaneous', 'miscellaneous'), ('misclaneous', 'miscellaneous'),
            ('maintainance', 'maintenance'), ('partion', 'partition')]


def type_key(text):
    """Merge key: ignore case, spaces and punctuation, fix known spellings, drop a trailing 'case(s)'."""
    t = re.sub(r'[^a-z0-9]', '', str(text).lower())
    for wrong, right in SPELLING:
        t = t.replace(wrong, right)
    t = re.sub(r'complain(?!t)', 'complaint', t)
    return re.sub(r'(cases|case)$', '', t) or t


same = [['gr case', 'g r', 'grcase', 'gr-case', 'gr cases', 'gr'], ['u i', 'ui case', 'ui'],
        ['2(a)cc', '2 (a) cc', '2a(cc)'], ['complain', 'complaint', 'complaint case'],
        ['execuition cases', 'execution cases', 'execution'], ['partion suit', 'partition suit'],
        ['maintainance', 'maintenance case'], ['mislaneous', 'miscellaneous', 'misclaneous-case']]
for group in same:
    assert len({type_key(g) for g in group}) == 1, f"merge test failed: {group}"
for a, b in [('cs', 'cs(i)'), ('mac case', 'mact'), ('gr', 'cr'), ('uc', 'ui'), ('misc', 'miscellaneous'),
             ('anticipatory bail', 'abp')]:
    assert type_key(a) != type_key(b), f"wrongly merged: {a} / {b}"

# clean case-type spellings BEFORE picking the top types
counts = df['type_name_normalized'].value_counts()
canon = {}
for name in counts.index:          # sorted by count, so the most common spelling names the group
    canon.setdefault(type_key(name), name)
df['type_name_normalized'] = df['type_name_normalized'].map({n: canon[type_key(n)] for n in counts.index})
print(f"Case types after spelling cleanup: {len(counts)} -> {df['type_name_normalized'].nunique()}")

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
