import sys
import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe

slug = sys.argv[1]
PATH = f"data/processed/cases_2010_2013_{slug}_survival_normalized.csv"

EXPECTED_COLS = {
    "ddl_case_id", "year", "state_code", "dist_code", "court_no", "cino", "judge_position",
    "female_defendant", "female_petitioner", "female_adv_def", "female_adv_pet",
    "type_name", "purpose_name", "disp_name", "date_of_filing", "date_of_decision",
    "date_first_list", "date_last_list", "date_next_list", "event", "type_name_s",
    "purpose_name_s", "disp_name_s", "district_name", "court_name", "days_to_disposition",
    "date_last_list_clean", "multiple_hearings", "court_tier", "type_name_normalized",
    "court_tier_raw", "type_name_raw_norm"}
TIERS = ["Chief Judicial Magistrate", "Sub-Divisional Judicial Magistrate",
         "Civil Judge (Senior Division)", "District and Sessions Judge",
         "Judicial Magistrate First Class", "Civil Judge cum JMFC", "Other / unclear",
         "Additional District / Sessions Judge", "Civil Judge cum SDJM",
         "Civil Judge (Junior Division)"]

results = []
header = pd.read_csv(PATH, nrows=0).columns.tolist()
results.append(("schema: column set matches (32 columns)", set(header) == EXPECTED_COLS and len(header) == 32))

df = pd.read_csv(PATH, dtype=str, usecols=["event", "days_to_disposition", "date_of_filing",
                                            "date_last_list_clean", "court_tier", "type_name_normalized"])
df["event"] = pd.to_numeric(df["event"], errors="coerce")
df["days_to_disposition"] = pd.to_numeric(df["days_to_disposition"], errors="coerce")
df["filing_year"] = pd.to_datetime(df["date_of_filing"], errors="coerce").dt.year
df["last_list_year"] = pd.to_datetime(df["date_last_list_clean"], errors="coerce").dt.year
df["inconsistent"] = (((df.event == 1) & df.days_to_disposition.isna()) |
                      ((df.event == 0) & df.days_to_disposition.notna())).astype(int)
df["tier_unclear"] = (df.court_tier == "Other / unclear").astype(int)
df = df.drop(columns=["date_of_filing", "date_last_list_clean"])

context = gx.get_context(mode="ephemeral")
src = context.data_sources.add_pandas(f"{slug}_survival")
asset = src.add_dataframe_asset("survival_features")
bd = asset.add_batch_definition_whole_dataframe("whole")
batch = bd.get_batch(batch_parameters={"dataframe": df})

checks = [
    ("row count in expected range", gxe.ExpectTableRowCountToBeBetween(min_value=400000, max_value=550000)),
    ("event not null", gxe.ExpectColumnValuesToNotBeNull(column="event")),
    ("event in {0,1}", gxe.ExpectColumnValuesToBeInSet(column="event", value_set=[0, 1])),
    ("decided share within 0.55-0.68 (pending-rate drift guard)",
     gxe.ExpectColumnMeanToBeBetween(column="event", min_value=0.55, max_value=0.68)),
    ("event/duration consistent (decided has duration, pending has none)",
     gxe.ExpectColumnValuesToBeInSet(column="inconsistent", value_set=[0])),
    ("filing date parses", gxe.ExpectColumnValuesToNotBeNull(column="filing_year", mostly=0.999)),
    ("filing year 2010-2013", gxe.ExpectColumnValuesToBeBetween(column="filing_year", min_value=2010, max_value=2013)),
    ("duration 0-5000 days", gxe.ExpectColumnValuesToBeBetween(column="days_to_disposition", min_value=0, max_value=5000, mostly=0.999)),
    ("court_tier not null", gxe.ExpectColumnValuesToNotBeNull(column="court_tier")),
    ("court_tier in known set", gxe.ExpectColumnValuesToBeInSet(column="court_tier", value_set=TIERS)),
    ("unclear tier share <= 5%", gxe.ExpectColumnMeanToBeBetween(column="tier_unclear", min_value=0, max_value=0.05)),
    ("case types <= 100 + other", gxe.ExpectColumnUniqueValueCountToBeBetween(column="type_name_normalized", min_value=50, max_value=101)),
    ("last-listing year plausible (known ~0.1% corrupted)",
     gxe.ExpectColumnValuesToBeBetween(column="last_list_year", min_value=2005, max_value=2025, mostly=0.995)),
]
for name, exp in checks:
    r = batch.validate(exp)
    results.append((name, bool(r.success)))
    if not r.success:
        print("  detail:", r.result)

for name, ok in results:
    print(("PASS  " if ok else "FAIL  ") + name)
n_fail = sum(not ok for _, ok in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
