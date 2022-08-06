# %% TO GENERATE RISK TIER CONVERSION MATRIX FOR LIVE CHECK AND DIRECT LOAN RENEWALS
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns

pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'

renewal_data = pd.read_csv(origination_file,usecols=['ProductType','Tier_MultipleModels','Unique_ContractID','IP_Unique_ContractID','State'])

renewal_data.dropna(subset=['Tier_MultipleModels'],inplace=True)

renewal_data.loc[(renewal_data.ProductType == 'LC') & (renewal_data.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
renewal_data.loc[(renewal_data.ProductType == 'LC') & (renewal_data.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
renewal_data.loc[(renewal_data.ProductType == 'LC') & (renewal_data.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
renewal_data.loc[(renewal_data.ProductType == 'LC') & (renewal_data.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
renewal_data.loc[(renewal_data.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1

# %%

renewal_data = renewal_data[['Unique_ContractID','State','ProductType','Tier_MultipleModels']].merge(renewal_data[['IP_Unique_ContractID','Unique_ContractID','ProductType','Tier_MultipleModels']],how='inner',left_on='Unique_ContractID',right_on='IP_Unique_ContractID')
renewal_data = renewal_data.drop(columns=['Unique_ContractID_x']).rename(columns={'ProductType_x':'IP_ProductType','Tier_MultipleModels_x':'IP_Tier_MultipleModels','Unique_ContractID_y':'Unique_ContractID','ProductType_y':'ProductType','Tier_MultipleModels_y':'Tier_MultipleModels'})
renewal_data = renewal_data.loc[renewal_data.ProductType != 'LC']

del(renewal_data)

pivoted_counts = renewal_data.pivot_table(index=['State','IP_ProductType','IP_Tier_MultipleModels'],columns=['ProductType','Tier_MultipleModels'],values=['Unique_ContractID'],aggfunc='count',fill_value=0,margins=True)

pivoted_counts = pivoted_counts.div(pivoted_counts.Unique_ContractID['All'], axis='index')

pivoted_counts.columns = pivoted_counts.columns.droplevel(0)
pivoted_counts = pivoted_counts.drop('All', axis=1, level=0)
pivoted_counts = pivoted_counts.drop('All', axis=0, level=0)

pivoted_counts.to_csv('renewal_matrix.csv')
