# %% Imports and file load
import pathlib
import datetime as dt
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pmdarima import auto_arima

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from pylab import rcParams

# %% DATA LOAD

datafolder = pathlib.Path.cwd().parent.parent.parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'


mktgfile = datafolder / 'VT_Marketing_11012019.txt'
mktg = pd.read_csv(mktgfile, sep=',', low_memory=False)
appfile = datafolder / 'VT_Applications_11262019.txt'
app = pd.read_csv(appfile, sep=',', low_memory=False)

mktg['CashDate'] = pd.to_datetime(mktg['CashDate'])
mktg2 = mktg
mktg = mktg[['CashDate', 'Cashings', 'Unique_BranchID']]
mktg = mktg.dropna()

app['AppCreatedDate'] = pd.to_datetime(app['AppCreatedDate'])
app = app[['AppCreatedDate', 'Booked_Indicator', 'Unique_BranchID','Unique_ApplicationID','Application_Source']]
app = app.dropna()

cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1))]
cashed = cashed.groupby(['Unique_BranchID',cashed.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cashed.index = cashed.index.set_levels([cashed.index.levels[0], cashed.index.levels[1].to_timestamp()])
cashed = cashed.reset_index()

cash_actuals = mktg.loc[mktg['CashDate'] >= dt.datetime(2015,1,1)]
cash_actuals = cash_actuals.groupby(['Unique_BranchID',cash_actuals.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cash_actuals.index = cash_actuals.index.set_levels([cash_actuals.index.levels[0], cash_actuals.index.levels[1].to_timestamp()])
cash_actuals = cash_actuals.reset_index()

newandformer = app.loc[((app['Application_Source']=='New Customer')|(app['Application_Source']=='Former Customer'))]
newandformer = newandformer.groupby(['Unique_BranchID', newandformer.AppCreatedDate.dt.to_period('M')])['Booked_Indicator'].sum().to_frame()
newandformer.index = newandformer.index.set_levels([newandformer.index.levels[0], newandformer.index.levels[1].to_timestamp()])
newandformer = newandformer.reset_index()

cashed = cashed.merge(mktg2[['Unique_BranchID','State']].drop_duplicates(), how='left',on='Unique_BranchID')
statecashings = cashed.groupby(['State','CashDate'])['Cashings'].sum().reset_index()

newandformer.groupby('AppCreatedDate')['Booked_Indicator'].sum()


# %% additional analysis

for state in statecashings['State'].unique():
    decomp = seasonal_decompose(statecashings.loc[statecashings.State == state][['CashDate','Cashings']].set_index('CashDate'), model='additive',freq=12)
    decomp.plot()
    plt.title(state)

# %% LIVE CHECK CASHING PREDICTIONS
LC_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
for branch in cashed['Unique_BranchID'].unique():
    branchcashed = cashed.loc[cashed.Unique_BranchID==branch][['Cashings']].set_index(cashed.loc[cashed.Unique_BranchID==branch]['CashDate'])

    try:
        mod = sm.tsa.statespace.SARIMAX(branchcashed,
                                            order=(1, 0, 1),
                                            seasonal_order=(1, 1, 0, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
        results = mod.fit()
        #print(results.summary().tables[1])
        #results.plot_diagnostics(figsize=(10, 6))
        #plt.show()

        pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), end=dt.datetime(2020,1,1))
        #pred_ci = pred.conf_int()
        temp = {'PredMonth':pred.predicted_mean.index,'Prediction':pred.predicted_mean.values}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        LC_predictions = LC_predictions.append(temp)

    except:
        pass

LC_predictions

#LC_predictions.to_csv(outputfolder / 'LC_branch_sarimax_predictions.csv')
#cash_actuals.to_csv(outputfolder / 'LC_branch_actuals_for_sarimax.csv')

# %% DIRECT LOAN APPLICATION PREDICTIONS
DL_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
for branch in newandformer['Unique_BranchID'].unique():
    branchDLs = newandformer.loc[newandformer.Unique_BranchID==branch][['Booked_Indicator']].set_index(newandformer.loc[newandformer.Unique_BranchID==branch]['AppCreatedDate'])

    try:
        mod = sm.tsa.statespace.SARIMAX(branchDLs,
                                            order=(1, 0, 1),
                                            seasonal_order=(1, 1, 0, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
        results = mod.fit()
        #print(results.summary().tables[1])
        #results.plot_diagnostics(figsize=(10, 6))
        #plt.show()

        pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), end=dt.datetime(2020,1,1))
        #pred_ci = pred.conf_int()
        temp = {'PredMonth':pred.predicted_mean.index,'Prediction':pred.predicted_mean.values}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        DL_predictions = DL_predictions.append(temp)

    except:
        pass

DL_predictions
#DL_predictions.to_csv(outputfolder / 'DL_branch_sarimax_predictions.csv')
#newandformer.to_csv(outputfolder / 'DL_branch_actuals_for_sarimax.csv')
