import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

df = pd.read_csv('data/processed/cases_2012_delhi_features.csv', dtype=str)

df['date_of_filing'] = pd.to_datetime(df['date_of_filing'], errors='coerce')
df['date_of_decision'] = pd.to_datetime(df['date_of_decision'], errors='coerce')
df['days_to_disposition'] = pd.to_numeric(df['days_to_disposition'], errors='coerce')

context = gx.get_context(mode='file')
source = context.data_sources.add_or_update_pandas('case_delay_source')
asset = source.add_dataframe_asset(name='cases_2012_delhi')
batch_def = asset.add_batch_definition_whole_dataframe('full_batch')
batch = batch_def.get_batch(batch_parameters={'dataframe': df})

suite = gx.ExpectationSuite(name='cases_2012_delhi_suite')
# The 5000-01-01 sentinel should never appear as a literal parsed date value —
# it must be excluded/handled before this stage, not silently present
suite.add_expectation(gxe.ExpectColumnValuesToNotBeInSet(
    column='date_last_list',
    value_set=['5000-01-01'],
    mostly=0.996
))
for col in ['state_code', 'dist_code', 'type_name', 'date_of_filing']:
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

suite.add_expectation(gxe.ExpectColumnPairValuesAToBeGreaterThanB(
    column_A='date_of_decision', column_B='date_of_filing', or_equal=True, mostly=0.998
))
# Reject implausible filing/decision years (corrupted data, e.g. "1204" instead of "2014")
suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
    column='date_of_filing', min_value=pd.Timestamp('2005-01-01'), max_value=pd.Timestamp('2020-12-31'), mostly=0.999
))
suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
    column='date_of_decision', min_value=pd.Timestamp('2005-01-01'), max_value=pd.Timestamp('2020-12-31'), mostly=0.999
))
# days_to_disposition must be non-negative (a few negative-duration rows are a known, tolerated rate)
suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
    column='days_to_disposition', min_value=0, mostly=0.998
))
suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
    column='female_defendant',
    value_set=['0 male', '1 female', '-9998 unclear', '-9999 missing name']
))

context.suites.add_or_update(suite)
result = batch.validate(suite)
print(result)
