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
from statsmodels.tsa.seasonal import seasonal_decompose
from random import random
import warnings
import itertools
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from pylab import rcParams
import math
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt


###############
# %% DATA LOAD
###############


outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'output'
datafolder = pathlib.Path.cwd().parent / 'Data'

mktgfile = datafolder / 'VT_Marketing_11012019.txt'
mktg = pd.read_csv(mktgfile, sep=',', low_memory=False)
appfile = datafolder / 'VT_Applications_11262019.txt'
app = pd.read_csv(appfile, sep=',', low_memory=False)

m = mktg[['IssueDate', 'CashDate', 'Cashings', 'Mailings', 'Unique_BranchID', 'State']]


m['IssueDate'] = pd.to_datetime(m['IssueDate'])
m['CashDate'] = pd.to_datetime(m['CashDate'])

m['Adjusted_Issue_Date'] = m['IssueDate'] + dt.timedelta(days = 15)

m['Date'] = pd.to_datetime(m['Adjusted_Issue_Date']).dt.to_period('M')
m = m.loc[m['Date'] <= dt.datetime(2019,10,1)]
issues = m[['Mailings', 'Date']]
cashes = m[['Cashings','Date']]

sum_issues = issues.groupby(['Date'])['Mailings'].sum()
cash_sums = cashes.groupby(['Date'])['Cashings'].sum()

rr = cash_sums/sum_issues
rr = rr.to_frame().reset_index()
rr['Date'] = rr.Date.dt.to_timestamp()
rr
sns.lineplot(x='Date', y=0, data=rr)


##################
# %% for RESULTS
##################


cash_actuals = m.loc[(m['CashDate'] >= dt.datetime(2019,1,1)) & (m['CashDate'] < dt.datetime(2019,8,1))]
cash_actuals = cash_actuals.groupby(['Unique_BranchID',cash_actuals.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cash_actuals.index = cash_actuals.index.set_levels([cash_actuals.index.levels[0], cash_actuals.index.levels[1].to_timestamp()])
cash_actuals = cash_actuals.reset_index()

mail_actuals = m.loc[(m['Date'] >= dt.datetime(2019,1,1)) & (m['Date'] < dt.datetime(2019,8,1))]
mail_actuals = mail_actuals.groupby(['Unique_BranchID',mail_actuals.IssueDate.dt.to_period('M')])['Mailings'].sum().to_frame()
mail_actuals.index = mail_actuals.index.set_levels([mail_actuals.index.levels[0], mail_actuals.index.levels[1].to_timestamp()])
mail_actuals = mail_actuals.reset_index()

mail_actuals


###################################################
# %% Response rate by state and by state by branch
###################################################


issues = m[['Mailings', 'Date', 'State']]
cashes = m[['Cashings','Date', 'State']]

sum_issues = issues.groupby([ 'State','Date'])['Mailings'].sum()
cash_sums = cashes.groupby(['State','Date'])['Cashings'].sum()

rr = cash_sums/sum_issues
rr = rr.to_frame().reset_index()
rr.Date = rr.Date.dt.to_timestamp()

fig, axs = plt.subplots(3,3,figsize=(20,10),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in rr.State.unique():
    ax = sns.lineplot(x='Date', y=0, data=rr.loc[(rr.State == state)], ax=axs[num])
    ax.set_title('State - {}'.format(state))
    ax.set_xlabel('Date')
    ax.set_ylabel('Cashing Response Rate')
    num+=1
fig.show()




issues = m[['Mailings', 'Date', 'State', 'Unique_BranchID']]
cashes = m[['Cashings','Date', 'State', 'Unique_BranchID']]

sum_issues = issues.groupby([ 'State','Unique_BranchID','Date'])['Mailings'].sum()
cash_sums = cashes.groupby(['State','Unique_BranchID','Date'])['Cashings'].sum()
sum_issues.index = sum_issues.index.set_levels([sum_issues.index.levels[0], sum_issues.index.levels[1], sum_issues.index.levels[2].to_timestamp()])
cash_sums.index = cash_sums.index.set_levels([cash_sums.index.levels[0], cash_sums.index.levels[1], cash_sums.index.levels[2].to_timestamp()])
rr = cash_sums/sum_issues
rr = rr.to_frame().reset_index()
rr
fig, axs = plt.subplots(3,3,figsize=(40,20),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in rr.State.unique():
    ax = sns.lineplot(x='Date', y=0, hue='Unique_BranchID', data=rr.loc[(rr.State == state)], ax=axs[num])
    ax.set_title('State - {}'.format(state))
    ax.set_xlabel('Date')
    ax.set_ylabel('Cashing Response Rate')
    num+=1
fig.show()

################################
# %% calculate seasonal factors
################################


issues = m[['Mailings', 'Date', 'State', 'Unique_BranchID']]
cashes = m[['Cashings','Date', 'State', 'Unique_BranchID']]

sum_issues = issues.groupby([ 'State','Date'])['Mailings'].sum()
cash_sums = cashes.groupby(['State','Date'])['Cashings'].sum()
sum_issues.index = sum_issues.index.set_levels([sum_issues.index.levels[0], sum_issues.index.levels[1].to_timestamp()])
cash_sums.index = cash_sums.index.set_levels([cash_sums.index.levels[0], cash_sums.index.levels[1].to_timestamp()])
rr = cash_sums/sum_issues
rr = rr.to_frame().reset_index()
rr.rename(columns={0:'ResponseRate'}, inplace=True)
rr = rr.loc[rr['Date'] <= dt.datetime(2019,1,1)]
state_seasonals = {}
for state in rr.State.unique():
    try:
        decomp = seasonal_decompose(rr.loc[rr.State == state][['Date','ResponseRate']].set_index('Date'), model='multiplicative',freq=12)
        dszn = decomp.seasonal
        dszn = pd.DataFrame(dszn)
        dszn.reset_index(inplace=True)
        state_seasonals[state] = dszn.iloc[0:12]['ResponseRate']
    except:
        pass
szn_facts = pd.DataFrame(state_seasonals)
szn_facts['MO'] = szn_facts['AL']
szn_facts['TX'] = szn_facts['SC']
szn_facts['Month'] = [1,2,3,4,5,6,7,8,9,10,11,12]

#state_seasonals.to_csv(outputfolder / 'seasonal_factors.csv')

###################################
# %% lets try some forecasting now
###################################


issues = m[['Mailings', 'Date', 'State', 'Unique_BranchID']]
cashes = m[['Cashings','Date', 'State', 'Unique_BranchID']]

sum_issues = issues.groupby(['State','Unique_BranchID','Date'])['Mailings'].sum()
cash_sums = cashes.groupby(['State','Unique_BranchID','Date'])['Cashings'].sum()
sum_issues.index = sum_issues.index.set_levels([sum_issues.index.levels[0], sum_issues.index.levels[1], sum_issues.index.levels[2].to_timestamp()])
cash_sums.index = cash_sums.index.set_levels([cash_sums.index.levels[0], cash_sums.index.levels[1], cash_sums.index.levels[2].to_timestamp()])
rr = cash_sums/sum_issues
rr = rr.to_frame().reset_index()
rr.rename(columns={0:'RR'}, inplace=True)

rr = create_deseasonal(rr)

num_forecasts = 7
prediction = []
predictions = {}
for b in rr['Unique_BranchID'].unique():
    if b < 164:
        x = rr.loc[rr['Unique_BranchID'] == b]
        state = x.iloc[0]['State']
        rr
        rolling_pred = rolling_average(b, num_forecasts)
        exp_predictions = exponetial_smoothing_predictions(b, num_forecasts)
        for num in range(1,num_forecasts+1):
            exp_num = szn_facts.loc[szn_facts['Month'] == num][state].values[0]*exp_predictions[num-1]
            rolling_num = szn_facts.loc[szn_facts['Month'] == num][state].values[0]*rolling_pred[num-1]
            prediction.append((exp_num+rolling_num)/2)
            #prediction.append((rolling_num))
        predictions[b] = prediction
        prediction = []
    else:
        pass
predictions
predictions = pd.DataFrame(predictions)
forecast = {}
cant_predict = []
for b in rr['Unique_BranchID'].unique():
    if b < 164:
        try:
            forecast[b] = mail_actuals.loc[mail_actuals['Unique_BranchID'] == b]['Mailings'].values*predictions[b].values
        except:
            cant_predict.append(b)
            #pass

forecast = pd.DataFrame(forecast)

forecast[forecast < 5] = 10
forecast[forecast > 100] = 100


#forecast.to_csv(outputfolder / 'cashings_new_by_mailings.csv')
forecast = forecast.sum(axis=1, skipna=True).to_frame()
cash_actuals_plot = cash_actuals.loc[(cash_actuals['Unique_BranchID'] < 164) & (cash_actuals['Unique_BranchID'] != 133) & (cash_actuals['CashDate'] >= dt.datetime(2018,1,1)) & (cash_actuals['CashDate'] < dt.datetime(2019,8,1))].groupby(cash_actuals.CashDate.dt.to_period('M'))['Cashings'].sum().to_frame().reset_index()

actuals_like_forecast = {}
for b in cash_actuals['Unique_BranchID'].unique():
    if b < 164 and b != 133:
        actuals_like_forecast[b] = cash_actuals.loc[cash_actuals['Unique_BranchID'] == b]['Cashings'].values
actuals_like_forecast = pd.DataFrame(actuals_like_forecast)
#actuals_like_forecast.to_csv(outputfolder / 'cashings_new_actuals.csv')


############################
# %% RESULTS FROM MAIN CODE
############################

plt.plot(forecast[0].values)
plt.plot(cash_actuals_plot.Cashings.values, color='red')
plt.title('Predicting using state aggreagated by Nearest Neighbor MonthsOpen \n and Month of year nearest neighbor (1/19-9/19)')
plt.show()

MSE = mean_squared_error(cash_actuals_plot.Cashings.values[0:7], forecast[0].values)
MSE
math.sqrt(MSE)
ME = np.mean(forecast[0].values - cash_actuals_plot.Cashings.values[0:7])
ME
ME*7/np.sum(cash_actuals_plot.Cashings.values[0:7])*100
MAPE = np.mean(np.abs((cash_actuals_plot.Cashings.values[0:7] - forecast[0].values) / cash_actuals_plot.Cashings.values[0:7])) * 100
MAPE

np.sum(forecast[0].values)
np.sum(cash_actuals_plot.Cashings.values[0:7])
np.sum(forecast[0].values) - np.sum(cash_actuals_plot.Cashings.values[0:7])

###############
# %% functions
###############


def cashed_months_open(rr):
    open_count = 1
    for b in rr['Unique_BranchID'].unique():
        for i in rr.loc[rr['Unique_BranchID'] == b].index.unique():
            rr.loc[(rr['Unique_BranchID'] == b) & (rr.index == i), 'MonthsOpen'] = open_count
            open_count = open_count + 1
        open_count = 1
    return rr


def create_deseasonal(rr):
    rr = cashed_months_open(rr)
    rr.Date = pd.to_datetime(rr.Date)
    rr['Month'] = pd.DatetimeIndex(rr.Date).month
    for b in rr['Unique_BranchID'].unique():
        temp = rr.loc[rr['Unique_BranchID'] == b]
        for m in temp['MonthsOpen'].unique():
            rr.loc[(rr['MonthsOpen'] == m) & (rr['Unique_BranchID'] == b), 'Dszn'] = temp.loc[temp['MonthsOpen'] == m]['RR'].values[0]/szn_facts.loc[szn_facts['Month'] == temp.loc[temp['MonthsOpen'] == m]['Month'].values[0]][temp['State'].values[0]].values[0]
    return rr

def exponetial_smoothing_predictions(b, num_forecasts):
    X = np.asarray(rr.loc[rr['Unique_BranchID'] == b]['Dszn'].values)
    predictions = []
    for i in range(1,num_forecasts+1):
        model = Holt(X)
        fit = model.fit(smoothing_level=0.5, smoothing_slope=0.1)
        prediction = fit.forecast(1)
        predictions.append(prediction[0])
        X = list(X)
        X.append(prediction[0])
        X = np.asarray(X)
    return predictions

def rolling_average(b, num_forecasts):
    X = rr.loc[rr['Unique_BranchID'] == b].Dszn.values
    window = 3
    history = [X[i] for i in range(window)]
    test = [X[i] for i in range(window, len(X))]
    predictions = list()
    # walk forward over time steps in test
    for t in range(len(test)+num_forecasts+3):
        if t >= len(test):
            length = len(history)
            yhat = np.mean([history[i] for i in range(length-window,length)])
            predictions.append(yhat)
            history
        else:
        	length = len(history)
        	yhat = np.mean([history[i] for i in range(length-window,length)])
        	obs = test[t]
        	predictions.append(yhat)
        	history.append(obs)
    return(predictions[-num_forecasts:])
