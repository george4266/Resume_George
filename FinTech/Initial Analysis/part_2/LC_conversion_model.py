import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

outputfolder = pathlib.Path.cwd() / 'Outputs'
datafolder = pathlib.Path.cwd().parent.parent.parent / 'Data'
originfile = datafolder / 'VT_Originations_11012019.txt'
perffile1 = datafolder / 'VT_Performance_1.txt'
perffile2 = datafolder / 'VT_Performance_2.txt'
perffile3 = datafolder / 'VT_Performance_3.txt'
perffile4 = datafolder / 'VT_Performance_4.txt'

origin = pd.read_csv(originfile, sep=',', low_memory=False)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))

perf[['Approved_Apps','Declined_Apps','Solicitation_Memos', 'Contacted_Memos']].fillna(value=0, inplace=True)
origin['BookDate'] = pd.to_datetime(origin['BookDate'])
origin['IP_Unique_ContractID'].fillna(value=0, inplace=True)
origin['Contract_Rank'] = 1
orig = origin
conversions = orig.merge(orig,how='left',left_on='Unique_ContractID',right_on='IP_Unique_ContractID')
conversions['Converted?'] = 0 #Helper calculated column
conversions.loc[~pd.isnull(conversions.Unique_ContractID_y), 'Converted?'] = 1
conversions = conversions[['Unique_ContractID_x','Converted?']].rename(columns={'Unique_ContractID_x':'Unique_ContractID'})
conversions = conversions.merge(origin, how='left',on='Unique_ContractID')
conversions.rename(columns={'Converted?_x':'Converted?'}, inplace=True)
single = conversions[conversions['IP_Unique_ContractID'] == 0]

single_new = single[['Unique_ContractID']]
single_new.rename(columns={'Unique_ContractID':'IP_Unique_ContractID'}, inplace=True)

count = 2
final = single
final_new = single_new
while len(final_new) != 0:
    x = pd.merge(final_new, origin, on='IP_Unique_ContractID')
    x['Contract_Rank'] = count
    final = final.append(x, ignore_index=True,sort=True)
    final_new = x[['Unique_ContractID']]
    final_new.rename(columns={'Unique_ContractID':'IP_Unique_ContractID'}, inplace=True)
    count = count + 1

perf.groupby('ProcessStatus').mean()
perf.groupby('ProcessStatus').max()
perf.groupby('ProcessStatus').min()

perf['RenewedMarker'] = 0
perf.loc[perf['ProcessStatus'] == 'Renewed', 'RenewedMarker'] = 1

combined = final.merge(perf,on='Unique_ContractID',how='left' )

combined = combined.loc[combined['BookDate'] > dt.datetime(2017,12,31)]

combined.groupby('Contacted_Memos').mean()
combined.columns
combined.drop(columns=['Unique_BranchID_y'], axis=1, inplace=True)
combined.rename(columns={'Unique_BranchID_x':'Unique_BranchID'}, inplace=True)

x = combined['Solicitation_Memos'].sum()
y = combined['RenewedMarker'].sum()
y/x
combined.isna().sum()

combined[['Solicitation_Memos', 'Contacted_Memos', 'Approved_Apps', 'Declined_Apps']].fillna(value=0, inplace=True)

total_from_renewed = combined[['BookDate', 'CashToCustomer', 'Contract_Rank']]
total_from_renewed = total_from_renewed.loc[total_from_renewed['BookDate'] < dt.datetime(2019,4,1)]
total_from_renewed = total_from_renewed.loc[total_from_renewed['Contract_Rank'] >= 2]
total_from_renewed = total_from_renewed.groupby(total_from_renewed.BookDate.dt.to_period('M'))['CashToCustomer'].sum().reset_index()
sns.barplot(data=total_from_renewed, x='BookDate', y='CashToCustomer')

total_from_renewed.to_csv(outputfolder / 'total_cash_from_renewed.csv')
