# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'

orig = pd.read_csv(origination_file, sep=',', low_memory=False)
orig.drop(columns=['TotalNote', 'CashToCustomer', 'Segment', 'IP_Unique_ContractID', 'RegularPayment', 'Unique_ApplicationID'], inplace=True)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))

orig['BookDate'] = pd.to_datetime(orig['BookDate'])


# %% create data frames

orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[orig.Tier_MultipleModels == 0.5, 'Tier_MultipleModels'] = 1
orig = orig[['Unique_ContractID', 'State', 'Tier_MultipleModels', 'ProductType', 'BookDate']]
perf = perf[['Unique_ContractID', '30+_Indicator', 'MonthsOnBook']]

#%% combine origin, perf and calculate currentmonth
combined = orig.merge(perf, on='Unique_ContractID', how='left')
combined['MonthsOnBook'].dropna(inplace=True)
combined['months_added'] = pd.to_timedelta(combined['MonthsOnBook'], 'M')
combined['step_one'] = combined['BookDate'] + combined['months_added']
combined['CurrentMonth'] = combined['step_one'].dt.strftime('%m/%Y')

# %% average 30+_indicator over time
combined['CurrentMonth'] = pd.to_datetime(combined['CurrentMonth'])
combined = combined.loc[(combined['CurrentMonth'] >= dt.datetime(2018,1,1)) & (combined['CurrentMonth'] < dt.datetime(2019,10,1))]

delinquency = combined.groupby(combined.CurrentMonth.dt.to_period('M'))['30+_Indicator'].mean().to_frame().reset_index()

sns.lineplot(x='CurrentMonth', y='30+_Indicator', data=delinquency, err_style='band')
### But check this out
delinquency

# %% lets try this again but over MonthsOnBook
combined['MonthsOnBook'].dropna(inplace=True)
combined = combined.loc[(combined['BookDate'] >= dt.datetime(2017,1,1))]
delinquency_mob = combined.groupby('MonthsOnBook')['30+_Indicator'].mean().to_frame().reset_index()
delinquency_mob
sns.lineplot(x='MonthsOnBook', y='30+_Indicator', data=delinquency_mob)

del_mob_state = combined.groupby(['MonthsOnBook', 'State'])['30+_Indicator'].mean().to_frame().reset_index()
del_mob_tier = combined.groupby(['MonthsOnBook', 'Tier_MultipleModels'])['30+_Indicator'].mean().to_frame().reset_index()
del_mob_prod = combined.loc[combined['ProductType'] != 'Sales'].groupby(['MonthsOnBook', 'ProductType'])['30+_Indicator'].mean().to_frame().reset_index()


del_mob_tier.Tier_MultipleModels.unique()

sns.lineplot(x='MonthsOnBook', y='30+_Indicator', hue='State', data=del_mob_state)

sns.lineplot(x='MonthsOnBook', y='30+_Indicator', hue='Tier_MultipleModels', data=del_mob_tier)


sns.lineplot(x='MonthsOnBook', y='30+_Indicator', hue='ProductType', data=del_mob_prod)
