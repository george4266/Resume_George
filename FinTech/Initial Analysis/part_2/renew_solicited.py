import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import swifter

outputfolder = pathlib.Path.cwd() / 'Outputs'
datafolder = pathlib.Path.cwd().parent.parent.parent / 'Data'
perffile1 = datafolder / 'VT_Performance_1.txt'
perffile2 = datafolder / 'VT_Performance_2.txt'
perffile3 = datafolder / 'VT_Performance_3.txt'
perffile4 = datafolder / 'VT_Performance_4.txt'
originfile = datafolder / 'VT_Originations_11012019.txt'

perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))
origin = pd.read_csv(originfile, sep=',', low_memory=False)

perf = perf[['Unique_ContractID', 'MonthsOnBook', 'Solicitation_Memos', 'ProcessStatus']]
origin = origin[['Unique_ContractID', 'CashToCustomer', 'BookDate']]

combined = origin.merge(perf, on='Unique_ContractID', how='left')
combined['BookDate'] = pd.to_datetime(combined['BookDate'])

combined[['MonthsOnBook', 'Solicitation_Memos']] = combined[['MonthsOnBook', 'Solicitation_Memos']].fillna(value=0,  axis=1)
combined = combined.loc[(combined['BookDate'] >= dt.datetime(2018,1,1)) & (combined['BookDate'] < dt.datetime(2019,4,1))]
combined['Renewed?'] = 0
combined.loc[combined['ProcessStatus'] == 'Renewed', 'Renewed?'] = 1
combined['CurrentMonth'] = combined[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)

solicit = combined.groupby(combined.CurrentMonth.dt.to_period('M'))['Solicitation_Memos'].sum().reset_index()
renewed = combined.groupby(combined.CurrentMonth.dt.to_period('M'))['Renewed?'].sum().reset_index()

solicit.head()
solicit_vs_renewed = solicit
solicit_vs_renewed.rename(columns={'Solicitation_Memos':'Ratio'}, inplace=True)
solicit_vs_renewed['Ratio'] = renewed['Renewed?']/solicit_vs_renewed['Ratio']

sns.barplot(data=solicit_vs_renewed, x='CurrentMonth', y='Ratio')
solicit_vs_renewed.to_csv(outputfolder / 'solicit_vs_renewed.csv')

total_from_renewed = combined[['BookDate', 'CashToCustomer']]
total_from_renewed = total_from_renewed.groupby(total_from_renewed.BookDate.dt.to_period('M'))['CashToCustomer'].sum().reset_index()
sns.barplot(data=total_from_renewed, x='BookDate', y='CashToCustomer')

total_from_renewed.to_csv(outputfolder / 'total_cash_from_renewed.csv')
