# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py

# %% data files
datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
apps = datafolder / 'VT_Applications_11262019.txt'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'


# %% data clean
#apps = pd.read_csv(apps, sep=',', low_memory=False)
origin = pd.read_csv(origination_file, sep=',', low_memory=False)
origin.drop(columns=['State', 'AmountFinanced', 'TotalNote', 'CashToCustomer', 'Segment', 'IP_Unique_ContractID', 'RegularPayment', 'Unique_ApplicationID'], inplace=True)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))

origin['BookDate'] = pd.to_datetime(origin['BookDate'])
portfolio = origin.merge(perf, on='Unique_ContractID')


portfolio['months_added'] = pd.to_timedelta(portfolio['MonthsOnBook'], 'M')
portfolio['CurrentMonth'] = portfolio['BookDate'] + portfolio['months_added']
portfolio['CurrentMonth'] = portfolio['CurrentMonth'].dt.strftime('%Y-%m-%d')
portfolio['CurrentMonth'].dt.normalize()
portfolio['CurrentMonth'] = pd.to_datetime(portfolio['CurrentMonth'])
portfolio = portfolio.loc[(portfolio['CurrentMonth'] >= dt.datetime(2018,1,1)) & (portfolio['CurrentMonth'] < dt.datetime(2019,10,1))]
portfolio.drop(columns=['Unique_BranchID_y', 'Unique_CustomerID_y'], inplace=True)
portfolio.rename(columns={'Unique_BranchID_x':'Unique_BranchID', 'Unique_CustomerID_x':'Unique_CustomerID'})
portfolio.loc[portfolio['ProductType'] != 'LC', 'ProductType'] = 'DL'
portfolio['CurrentMonth'].unique()
outflow = portfolio.loc[portfolio['ProcessStatus'] == 'Closed'].groupby(['ProductType', 'CurrentMonth'])['Unique_ContractID'].count().to_frame().reset_index()
plt.figure(figsize=(20,11))
sns.lineplot(x='CurrentMonth', y='Unique_ContractID', hue='ProductType', data=outflow)
outflow.to_csv(outputfolder / 'outflow_by_type.csv')
