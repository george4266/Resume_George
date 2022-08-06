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
perffile1 = datafolder / 'perf1_with_CurrentMonth_2.txt'
perffile2 = datafolder / 'perf2_with_CurrentMonth_2.txt'
perffile3 = datafolder / 'perf3_with_CurrentMonth_2.txt'
perffile4 = datafolder / 'perf4_with_CurrentMonth_2.txt'

orig = pd.read_csv(origination_file, sep=',', low_memory=False)
orig.drop(columns=['TotalNote', 'CashToCustomer', 'Segment', 'IP_Unique_ContractID', 'RegularPayment', 'Unique_ApplicationID'], inplace=True)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))

orig['BookDate'] = pd.to_datetime(orig['BookDate'])

perf['CurrentMonth'] = pd.to_datetime(perf['CurrentMonth'])


# %% create data frames

orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[(orig.Rescored_Tier_2018Model == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.Rescored_Tier_2018Model == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.Rescored_Tier_2018Model == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.Rescored_Tier_2018Model == 1), 'Tier_MultipleModels'] = 2
orig.loc[ (orig.Rescored_Tier_2018Model == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[orig.Tier_MultipleModels == 0.5, 'Tier_MultipleModels'] = 1
orig = orig[['Unique_ContractID', 'State', 'Tier_MultipleModels', 'ProductType', 'BookDate', 'Rescored_Tier_2018Model', 'Rescored_Tier_2017Model']]
perf = perf[['Unique_ContractID', 'CurrentMonth', 'MonthsOnBook', 'Solicitation_Memos', 'Contacted_Memos']]

#%% combine origin and perf
combined = orig.merge(perf, on='Unique_ContractID', how='left')
combined['CurrentMonth'].dropna(inplace=True)

# %% SC ratio over MOB

sc_ratio = combined
sc_ratio['MonthsOnBook'].dropna(inplace=True)
sc_ratio.loc[(sc_ratio['Solicitation_Memos'] != 0) & (sc_ratio['Contacted_Memos'] !=0), 'SC_ratio'] = sc_ratio['Contacted_Memos']/sc_ratio['Solicitation_Memos']
sc_ratio['SC_ratio'].fillna(value=0, inplace=True)

sc_ratio['SC_ratio'].unique()

sc_ratio = sc_ratio.loc[(sc_ratio['CurrentMonth'] >= dt.datetime(2018,1,1))]
sc_ratio = sc_ratio.loc[(sc_ratio['BookDate'] >= dt.datetime(2017,1,1))]
sc_ratio.isna().sum()

sc_ratio['Rescored_Tier_2017Model'].unique()
sc_ratio['Rescored_Tier_2018Model'].unique()

sc_ratio.loc[(sc_ratio['Tier_MultipleModels'].isnull()) & (sc_ratio['Rescored_Tier_2017Model'] >= 1), 'Tier_MultipleModels'] = sc_ratio['Rescored_Tier_2017Model']


sc_ratio['Tier_MultipleModels'].dropna(inplace=True)

sc_ratio_mob = sc_ratio.groupby('MonthsOnBook')['SC_ratio'].mean().to_frame().reset_index()

sns.lineplot(x='MonthsOnBook', y='SC_ratio', data=sc_ratio_mob)



sc_ratio_mob_tier = sc_ratio.groupby(['MonthsOnBook', 'Tier_MultipleModels'])['SC_ratio'].mean().to_frame().reset_index()
sns.lineplot(x='MonthsOnBook', y='SC_ratio', hue='Tier_MultipleModels', data=sc_ratio_mob_tier)

sc_ratio_mob_pt = sc_ratio.groupby(['MonthsOnBook', 'ProductType'])['SC_ratio'].mean().to_frame().reset_index()
sns.lineplot(x='MonthsOnBook', y='SC_ratio', hue='ProductType', data=sc_ratio_mob_pt)

sc_ratio_mob_state = sc_ratio.groupby(['MonthsOnBook', 'State'])['SC_ratio'].mean().to_frame().reset_index()
sns.lineplot(x='MonthsOnBook', y='SC_ratio', hue='State', data=sc_ratio_mob_state)
