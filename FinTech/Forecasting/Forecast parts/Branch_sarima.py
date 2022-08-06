# %% Imports and file load
import pathlib
import datetime as dt
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
from statsmodels.tsa.stattools import adfuller
import warnings
import itertools
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from pylab import rcParams

outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'
datafolder = pathlib.Path.cwd().parent / 'Data'

Markfile = datafolder / 'VT_Marketing_11012019.txt'
mark = pd.read_csv(Markfile, sep=',', low_memory=False)

mark['CashDate'] = pd.to_datetime(mark['CashDate'])
mark1 = mark[['CashDate', 'Cashings', 'Unique_BranchID']]
#mark1 = mark1.dropna()
mark1['CashDate'].dropna(inplace=True)
mark1['Cashings'].fillna(value=0,inplace=True)


# %% lets do this by branches

branch_frame = mark1.sort_values(by='Unique_BranchID', ascending=True)
branches = {}

for b in branch_frame['Unique_BranchID'].unique():
    branch = branch_frame.loc[(branch_frame['Unique_BranchID'] == b) & (branch_frame['CashDate'] < dt.datetime(2019,10,1)) & (branch_frame['CashDate'] >= dt.datetime(2015,1,1))].groupby(branch_frame.CashDate.dt.to_period('M'))['Cashings'].sum().to_frame()
    branch.index = branch.index.to_timestamp()
    branches[b] = branch
branches

# %% time to forecast them branches

agg_branch_predictions = {}
for branch in branches:
    try:
        mod = sm.tsa.statespace.SARIMAX(branches[branch],
                                        order=(1, 0, 1),
                                        seasonal_order=(1, 1, 0, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()
        #print(results.summary().tables[1])
        #results.plot_diagnostics(figsize=(18, 8))
        #plt.show()
        pred = results.get_prediction(start=pd.to_datetime('2019-10-01'), end=dt.datetime(2020,10,1))
        agg_branch_predictions[branch] = pred.predicted_mean
        print(branch)
        print(pred.predicted_mean)
    except:
        pass
agg_branch_predictions

# %%
pred_ci = pred.conf_int()
ax = actuals['2015':].plot(label='observed',linewidth=4, fontsize=15)
pred.predicted_mean.plot(ax=ax, label='Forecasted', alpha=0.7, figsize=(15, 5),linewidth=4, fontsize=15)
print(pred.predicted_mean)
ax.set_xlabel('Date',fontsize=20)
ax.set_ylabel('Cashed units',fontsize=20)
plt.legend()
plt.show()











# %% test sets for trying out sarima
#cashings = mark1.loc[mark1['CashDate'] <= dt.datetime(2019,3,1)].groupby([mark1.CashDate.dt.to_period('M'), mark1.Unique_BranchID])['Cashings'].sum().to_frame()
#cashings.index=cashings.index.to_timestamp()
cashed = mark1.loc[(mark1['CashDate'] < dt.datetime(2019,3,1)) & (mark1['CashDate'] >= dt.datetime(2015,1,1))]
cashed = cashed.groupby(cashed.CashDate.dt.to_period('M'))['Cashings'].sum().to_frame()

cashed.index=cashed.index.to_timestamp()
cashed
actuals = mark1.groupby(mark1.CashDate.dt.to_period('M'))['Cashings'].sum().to_frame()
actuals.index = actuals.index.to_timestamp()
#cashings


# %%  Testing out sarima
mod = sm.tsa.statespace.SARIMAX(branches[69],
                                order=(1, 0, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(18, 8))
plt.show()


pred = results.get_prediction(start=pd.to_datetime('2019-3-01'), end=dt.datetime(2019,10,1))
print(pred.predicted_mean)
pred_ci = pred.conf_int()
ax = actuals['2015':].plot(label='observed',linewidth=4, fontsize=15)
pred.predicted_mean.plot(ax=ax, label='Forecasted', alpha=0.7, figsize=(15, 5),linewidth=4, fontsize=15)
print(pred.predicted_mean)
ax.set_xlabel('Date',fontsize=20)
ax.set_ylabel('Cashed units',fontsize=20)
plt.legend()
plt.show()

#actuals = mark1.loc[(mark1['CashDate'] >= dt.datetime(2019,3,1)) & (mark1['CashDate'] < dt.datetime(2019,10,1))].groupby(mark1.CashDate.dt.to_period('M'))['Cashings'].sum().to_frame().reset_index()
#actuals
#pred.predicted_mean

predictions = pd.DataFrame(pred.predicted_mean)
predictions.to_csv(outputfolder / 'sarimax_predictions.csv')
actuals.to_csv(outputfolder / 'actuals_for_sarimax.csv')
