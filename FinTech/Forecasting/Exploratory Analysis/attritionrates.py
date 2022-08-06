# %% Imports and file load
import pathlib, datetime
import numpy as np, pandas as pd, seaborn as sns, matplotlib, matplotlib.pyplot as plt

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'
origfile = datafolder / 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
orig = pd.read_csv(origfile, sep=',', low_memory=False)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))

# %% Processing into one table
orig = orig[['Unique_ContractID','Unique_BranchID','BookDate','Term','State','ProductType','Tier_MultipleModels']]
perf = perf[['Unique_ContractID','MonthsOnBook','ProcessStatus','NetReceivable']]

orig = orig.loc[orig.ProductType != 'Sales']
orig.dropna(subset=['Tier_MultipleModels'],inplace=True)
orig.BookDate = orig.BookDate.apply(pd.to_datetime)
orig['BookYear'] = orig.BookDate.dt.year
orig['BookMonth'] = orig.BookDate.dt.month

orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1

loans = orig.merge(perf,how='inner',on='Unique_ContractID')
loans.loc[(loans.MonthsOnBook < loans.Term) & (loans.ProcessStatus == 'Closed'),'ProcessStatus'] = 'PD/CO'

loans['BookDate'] = loans['BookDate'].values.astype('datetime64[M]')
loans['months_added'] = pd.to_timedelta(loans['MonthsOnBook'], 'M')
loans['CurrentMonth'] = loans['BookDate'] + loans['months_added']
loans['PrevMonth'] = loans['BookDate'] + loans['months_added'] - pd.to_timedelta(1, 'M')
#loans['CurrentMonth'] = loans['step_one'].dt.strftime('%m/%Y')
loans.drop(columns=['months_added'],inplace=True)

# %%
loans.loc[loans.BookYear == 2017].pivot_table(index=['MonthsOnBook'],columns=['ProcessStatus','State'],aggfunc='size').to_csv(pathlib.Path(outputfolder/'ST_attritionrates_2.csv'))

# %% for re-doing conversion/renewal rate graphs
loans.loc[(loans.ProductType == 'LC')&(loans.BookYear == 2017)].pivot_table(index=['MonthsOnBook'],columns=['ProcessStatus','Tier_MultipleModels'],aggfunc='size').to_csv(pathlib.Path(outputfolder/'riskconv_rates.csv'))

# %%
pd.to_timedelta(loans['MonthsOnBook'], 'M')
loans.loc[loans.ProcessStatus == 'Open'].pivot_table(index=['CurrentMonth'],values=['NetReceivable'])
#.to_csv(pathlib.Path(outputfolder/'riskconv_rates.csv'))

# %%
