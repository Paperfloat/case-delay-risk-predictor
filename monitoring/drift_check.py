import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

df = pd.read_csv('data/processed/cases_2010_2013_delhi_features.csv')

# Time-sliced simulation: older years as "reference" (what the model was
# effectively trained to expect), more recent years as "current" (simulating
# a later point where drift might have occurred)
reference = df[df['year'].isin([2010, 2011])][['type_name_normalized', 'court_tier', 'district_name']]
current = df[df['year'].isin([2012, 2013])][['type_name_normalized', 'court_tier', 'district_name']]

print("Reference rows:", reference.shape[0])
print("Current rows:", current.shape[0])

report = Report([DataDriftPreset(method='psi')], include_tests=True)
result = report.run(current_data=current, reference_data=reference)

result.save_html('monitoring/drift_report.html')
print("\nSaved HTML report to monitoring/drift_report.html")

# Print the raw dict so we can see the actual structure before parsing it further
import json
print(json.dumps(result.dict(), indent=2)[:3000])
# --- PSI drift trigger: parse the report to make a real retrain decision ---
result_dict = result.dict()
psi_by_feature = {}
for metric in result_dict['metrics']:
    if 'ValueDrift' in metric['metric_name']:
        psi_by_feature[metric['config']['column']] = metric['value']

PSI_THRESHOLD = 0.1
drifted_features = {k: v for k, v in psi_by_feature.items() if v > PSI_THRESHOLD}

print("\n--- Drift Trigger Decision ---")
for feature, psi in psi_by_feature.items():
    status = "DRIFT DETECTED" if psi > PSI_THRESHOLD else "stable"
    print(f"{feature}: PSI={psi:.4f} ({status})")

if drifted_features:
    print(f"\n>>> RETRAIN TRIGGERED: {len(drifted_features)} feature(s) exceeded PSI threshold {PSI_THRESHOLD}")
    print(f">>> Drifted features: {list(drifted_features.keys())}")
else:
    print(f"\n>>> No retrain needed: all features within PSI threshold {PSI_THRESHOLD}")
