# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.utils import compute_sample_weight
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'

perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))
origin = pd.read_csv(origination_file, sep=',', low_memory=False)

portfolio = origin.merge(perf, on='Unique_ContractID', how='left')
portfolio['BookDate'] = pd.to_datetime(portfolio['BookDate'])
portfolio['months_added'] = pd.to_timedelta(portfolio['MonthsOnBook'], 'M')
portfolio['step_one'] = portfolio['BookDate'] + portfolio['months_added']
portfolio['CurrentMonth'] = portfolio['step_one'].dt.strftime('%m/%Y')
portfolio['CurrentMonth'] = pd.to_datetime(portfolio['CurrentMonth'])

portfolio2 = portfolio[['CurrentMonth', 'ProcessStatus', 'Unique_BranchID', 'MonthsOnBook', 'Term']]
portfolio2.dropna(inplace=True)
portfolio2['counter'] = 1
portfolio2 = portfolio2.loc[portfolio2['CurrentMonth'] < dt.datetime(2019,1,1)]


# %% CALCULATE RENEWAL RATES


renewals_dl = portfolio2.loc[(portfolio2['ProcessStatus'] == 'Renewed') & (portfolio2['ProductType'] !='LC')].groupby(['Unique_BranchID',portfolio2.CurrentMonth.dt.to_period('M')])['counter'].sum().to_frame().reset_index()
renewals_lc = portfolio2.loc[(portfolio2['ProcessStatus'] == 'Renewed') & (portfolio2['ProductType'] =='LC')].groupby(['Unique_BranchID',portfolio2.CurrentMonth.dt.to_period('M')])['counter'].sum().to_frame().reset_index()

non_renewals_dl = portfolio2.loc[(portfolio2['ProcessStatus'] != 'Renewed') & (portfolio2['ProductType'] !='LC')].groupby(portfolio2.CurrentMonth.dt.to_period('M'))['counter'].count().to_frame().reset_index()
non_renewals_lc = portfolio2.loc[(portfolio2['ProcessStatus'] != 'Renewed') & (portfolio2['ProductType'] =='LC')].groupby(portfolio2.CurrentMonth.dt.to_period('M'))['counter'].count().to_frame().reset_index()


renewal_rate_dl = renewals_dl
renewal_rate_dl['rr'] = renewals_dl['counter']/non_renewals_dl['counter']
renewal_rate_dl['Date'] = renewal_rate_dl['CurrentMonth'].dt.to_timestamp()
sns.lineplot(x='Date', y='rr', data=renewal_rate_dl)

renewal_rate_lc = renewals_lc
renewal_rate_lc['rr'] = renewals_lc['counter']/non_renewals_lc['counter']
renewal_rate_lc['Date'] = renewal_rate_lc['CurrentMonth'].dt.to_timestamp()
sns.lineplot(x='Date', y='rr', data=renewal_rate_lc)

# %% CALCULATE ATTRITION RATES

attrition = portfolio.loc[(portfolio['ProcessStatus'] == 'Closed') & (portfolio['Term'] > portfolio['MonthsOnBook'])].groupby(['Unique_BranchID',portfolio.CurrentMonth.dt.to_period('M')])['counter'].sum().to_frame().reset_index()

non_attrition = portfolio2.loc[(portfolio2['ProcessStatus'] != 'Closed') | (portfolio2['ProcessStatus'] == 'Closed') & (portfolio2['Term'] == portfolio2['MonthsOnBook'])].groupby(portfolio2.CurrentMonth.dt.to_period('M'))['counter'].count().to_frame().reset_index()
