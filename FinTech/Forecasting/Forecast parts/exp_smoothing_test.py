###########################
# %% Imports and file load
###########################
import pathlib
import datetime as dt
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt


outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'output'
datafolder = pathlib.Path.cwd().parent / 'Data'

Markfile = datafolder / 'VT_Marketing_11012019.txt'
mktg = pd.read_csv(Markfile, sep=',', low_memory=False)

branchfile = datafolder / 'VT_Branches.txt'
branch = pd.read_csv(branchfile, sep=',', low_memory=False)

SeasonalFactorsfile = datafolder / 'seasonal_factors.csv'
szn_facts = pd.read_csv(SeasonalFactorsfile, sep=',')

mktg = mktg[pd.notnull(mktg['CashDate'])]
mktg['CashDate'] = pd.to_datetime(mktg['CashDate'])
mktg = mktg[['CashDate', 'Cashings', 'Unique_BranchID','State']]
#mark1 = mark1.dropna()
mktg['CashDate'].dropna(inplace=True)
mktg['Cashings'].fillna(value=0,inplace=True)
mktg2 = mktg


branch = branch[['Unique_BranchID', 'BranchOpenDate']]
branch['BranchOpenDate'] = pd.to_datetime(branch['BranchOpenDate'])
branch = branch.loc[branch['BranchOpenDate'] >= dt.datetime(2014,1,1)]

cash = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2014,1,1))]
cash = cash.groupby(['Unique_BranchID',cash.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cash.index = cash.index.set_levels([cash.index.levels[0], cash.index.levels[1].to_timestamp()])
cash = cash.reset_index()
cashed = cash.loc[cash['Unique_BranchID'].isin(branch['Unique_BranchID'].unique())]

cash_actuals = mktg.loc[mktg['CashDate'] >= dt.datetime(2014,1,1)]
cash_actuals = cash_actuals.groupby(['Unique_BranchID',cash_actuals.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cash_actuals.index = cash_actuals.index.set_levels([cash_actuals.index.levels[0], cash_actuals.index.levels[1].to_timestamp()])
cash_actuals = cash_actuals.reset_index()

cashed = cash.merge(mktg2[['Unique_BranchID','State']].drop_duplicates(), how='left',on='Unique_BranchID')
statecashings = cashed.groupby(['State','CashDate'])['Cashings'].sum().reset_index()


# %%


model = Holt(np.asarray(cashed.loc[cashed['Unique_BranchID'] == 164]['Cashings'].values), damped=False)
model2 = ExponentialSmoothing(np.asarray(cashed.loc[cashed['Unique_BranchID'] == 164]['Cashings'].values), trend='mul', seasonal=None, damped=True)


fit1 = model.fit(smoothing_level=0.2, smoothing_slope=0.05)
#fit2 = model.fit(smoothing_level=0.2)
fit3 = model.fit(smoothing_level=0.5)
pred1 = fit1.forecast(7)
#pred2 = fit2.forecast(7)
pred3 = fit3.forecast(7)
plot_actuals = cash_actuals.loc[(cash_actuals['CashDate'] >= dt.datetime(2019,1,1)) & (cash_actuals['CashDate'] < dt.datetime(2019,8,1)) & (cash_actuals['Unique_BranchID'] == 164)]['Cashings'].values

plt.plot(pred1, color='blue')
#plt.plot(pred2, color='red')
plt.plot(pred3, color='green')
plt.plot(plot_actuals, color='black')
plt.show()

# %%
