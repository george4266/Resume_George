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
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'



orig = pd.read_csv(origination_file, sep=',', low_memory=False)

perf1 = pd.read_csv(perffile1, sep=',', low_memory=False)
perf2 = pd.read_csv(perffile2, sep=',', low_memory=False)
perf3 = pd.read_csv(perffile3, sep=',', low_memory=False)
perf4 = pd.read_csv(perffile4, sep=',', low_memory=False)

orig = orig[['BookDate', 'Unique_ContractID']]
orig['BookDate'] = pd.to_datetime(orig['BookDate'])

date1 = orig.merge(perf1, on='Unique_ContractID')
date2 = orig.merge(perf2, on='Unique_ContractID')
date3 = orig.merge(perf3, on='Unique_ContractID')
date4 = orig.merge(perf4, on='Unique_ContractID')

perf_list = [date1, date2, date3, date4]

for df in perf_list:
    df['months_added'] = pd.to_timedelta(df['MonthsOnBook'], 'M')
    df['step_one'] = df['BookDate'] + df['months_added']
    df['CurrentMonth'] = df['step_one'].dt.strftime('%m/%Y')

date1['CurrentMonth'] = date1[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)
#date2['CurrentMonth'] = date2[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)
#date3['CurrentMonth'] = date3[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)
#date4['CurrentMonth'] = date4[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)

date12 = perf_list[0]
date22 = perf_list[1]
date32 = perf_list[2]
date42 = perf_list[3]

date12.head(25)
date1.head(25)



date1.drop(columns=['BookDate', 'step_one', 'months_added'], inplace=True)
date2.drop(columns=['BookDate', 'step_one', 'months_added'], inplace=True)
date3.drop(columns=['BookDate', 'step_one', 'months_added'], inplace=True)
date4.drop(columns=['BookDate', 'step_one', 'months_added'], inplace=True)

len(date1)
date1.isna().sum()

date1.to_csv(pathlib.Path(datafolder / 'perf1_with_CurrentMont_2.txt'))
date2.to_csv(pathlib.Path(datafolder / 'perf2_with_CurrentMont_2.txt'))
date3.to_csv(pathlib.Path(datafolder / 'perf3_with_CurrentMont_2.txt'))
date4.to_csv(pathlib.Path(datafolder / 'perf4_with_CurrentMont_2.txt'))
