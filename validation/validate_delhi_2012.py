import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

df = pd.read_csv('data/processed/cases_2012_delhi_clean.csv', dtype=str)

context = gx.get_context(mode='file')
source = context.data_sources.add_or_update_pandas('case_delay_source')
asset = source.add_dataframe_asset(name='cases_2012_delhi')
batch_def = asset.add_batch_definition_whole_dataframe('full_batch')
batch = batch_def.get_batch(batch_parameters={'dataframe': df})

suite = gx.ExpectationSuite(name='cases_2012_delhi_suite')

for col in ['state_code', 'dist_code', 'type_name', 'date_of_filing']:
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

suite.add_expectation(gxe.ExpectColumnPairValuesAToBeGreaterThanB(
    column_A='date_of_decision', column_B='date_of_filing', or_equal=True, mostly=0.998
))

suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
    column='female_defendant',
    value_set=['0 male', '1 female', '-9998 unclear', '-9999 missing name']
))

context.suites.add(suite)
result = batch.validate(suite)
print(result)
