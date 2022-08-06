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

outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'output'
datafolder = pathlib.Path.cwd().parent / 'Data'

Markfile = datafolder / 'VT_Marketing_11012019.txt'
mktg = pd.read_csv(Markfile, sep=',', low_memory=False)

branchfile = datafolder / 'VT_Branches.txt'
branch = pd.read_csv(branchfile, sep=',', low_memory=False)

appfile = datafolder / 'VT_Applications_11262019.txt'
app = pd.read_csv(appfile, sep=',', low_memory=False)

SeasonalFactorsfile = datafolder / 'seasonal_factors.csv'
szn_facts = pd.read_csv(SeasonalFactorsfile, sep=',')

app['AppCreatedDate'] = pd.to_datetime(app['AppCreatedDate'])
app = app[['AppCreatedDate', 'Booked_Indicator', 'Unique_BranchID','Unique_ApplicationID','Application_Source']]
app = app.dropna()

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


#####################################
# %% check branch cashings by states
#####################################


cash = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2014,1,1))]
cash = cash.groupby(['Unique_BranchID','State', cash.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cash.index = cash.index.set_levels([cash.index.levels[0], cash.index.levels[1], cash.index.levels[2].to_timestamp()])
cash = cash.reset_index()


fig, axs = plt.subplots(3,3,figsize=(50,35),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in cash.State.unique():
    ax = sns.lineplot(x='CashDate', y='Cashings', hue='Unique_BranchID', data=cash.loc[(cash.State == state)], ax=axs[num])
    ax.set_title('State - {}'.format(state))
    ax.set_xlabel('Date')
    ax.set_ylabel('Cashings')
    num+=1
fig.show()



########################################
# %% PREPARE FUNCTIONS BEFORE MAIN CODE
########################################


cashed = create_deseasonal(cashed)

#for st in cashed.State.unique():
    #for monthopen in cashed.MonthsOpen.unique():
    #    cashed.loc[(cashed['State'] == st) & (cashed['MonthsOpen'] == monthopen), 'MonthsOpenAvg'] = cashed.loc[(cashed['State'] == st) & (cashed['MonthsOpen'] == monthopen)]['Cashings'].mean()
    #for month in cashed.Month.unique():
        #cashed.loc[(cashed['State'] == st) & (cashed['Month'] == month), 'MonthsAvg'] = cashed.loc[(cashed['State'] == st) & (cashed['Month'] == month)]['Cashings'].mean()

num_forecasts = 7
prediction = []
predictions = {}
for b in cashed['Unique_BranchID'].unique():
    if b < 164 and b != 133:
        x = cashed.loc[cashed['Unique_BranchID'] == b]
        state = x.iloc[0]['State']
        current_month = x.iloc[-1].MonthsOpen
        #month_open_b = find_nearest_neighbor(b)
        month_b = find_nearest_neighbor_season(b)
        rolling_pred = rolling_average(b, num_forecasts)
        exp_predictions = exponetial_smoothing_predictions(b, num_forecasts)
        for num in range(1,num_forecasts+1):
            predict_month = num + current_month
            #MonthOpen_num = cashed.loc[(cashed['MonthsOpen'] == predict_month) & (cashed['State'] == state)]['MonthsOpenAvg'].mean()
            #MonthOpen_num = cashed.loc[(cashed['Unique_BranchID'] == month_open_b) & (cashed['MonthsOpen'] == predict_month)]['Cashings'].values[0]
            if state == 'TX':
                exp_num = szn_facts.loc[szn_facts['Month'] == num]['KY'].values[0]*exp_predictions[num-1]
                rolling_num = szn_facts.loc[szn_facts['Month'] == num]['KY'].values[0]*rolling_pred[num-1]
            else:
                exp_num = szn_facts.loc[szn_facts['Month'] == num][state].values[0]*exp_predictions[num-1]
                rolling_num = szn_facts.loc[szn_facts['Month'] == num][state].values[0]*rolling_pred[num-1]
            #Month_num = cashed.loc[(cashed['Unique_BranchID'] == month_b) & (cashed['Month'] == num)]['Cashings'].values[0]
            #Month_num = cashed.loc[(cashed['Month'] == num) & (cashed['State'] == state)]['MonthsAvg'].mean()
            #prediction.append((MonthOpen_num+Month_num)/2)
            prediction.append((((rolling_num+exp_num)/2) + month_b)/2)
        predictions[b] = prediction
        prediction = []
    else:
        pass

predictions = pd.DataFrame(predictions)
predictions.to_csv(outputfolder / 'cashings_new_expo.csv')
predict_values = predictions.sum(axis=1, skipna=True).to_frame()
predict_values
cash_actuals_plot = cash_actuals.loc[(cash_actuals['Unique_BranchID'] < 164) & (cash_actuals['CashDate'] >= dt.datetime(2019,1,1)) & (cash_actuals['CashDate'] < dt.datetime(2019,8,1))].groupby(cash_actuals.CashDate.dt.to_period('M'))['Cashings'].sum().to_frame().reset_index()



############################
# %% RESULTS FROM MAIN CODE
############################


plt.plot(predict_values[0].values)
plt.plot(cash_actuals_plot.Cashings.values, color='red')
plt.title('Predicting using state aggreagated by Nearest Neighbor MonthsOpen \n and Month of year nearest neighbor (1/19-9/19)')
plt.show()
MSE = mean_squared_error(cash_actuals_plot.Cashings.values[0:7], predict_values[0].values)
MSE
math.sqrt(MSE)
ME = np.mean(predict_values[0].values - cash_actuals_plot.Cashings.values[0:7])
ME
ME*7/np.sum(cash_actuals_plot.Cashings.values[0:7])*100
MAPE = np.mean(np.abs((cash_actuals_plot.Cashings.values[0:7] - predict_values[0].values) / cash_actuals_plot.Cashings.values[0:7])) * 100
MAPE

np.sum(predict_values[0].values)
np.sum(cash_actuals_plot.Cashings.values[0:7])
np.sum(predict_values[0].values) - np.sum(cash_actuals_plot.Cashings.values[0:7])


predictions.to_csv(outputfolder / 'cashing_predictions_new_branches_12019_72019.csv')
cash_actuals.loc[(cash_actuals['Unique_BranchID'] >= 164) & (cash_actuals['Unique_BranchID'] <= 207) & (cash_actuals['CashDate'] >= dt.datetime(2019,1,1))].to_csv(outputfolder / 'cashing_actual_new_branches_12019_72019.csv')

cash_actuals2 = cashed_months_open(cashed)
cash_actuals2 = cash_actuals2.groupby('MonthsOpen')['Cashings'].sum().to_frame()
cash_actuals_plot2 = cash_actuals2['Cashings'].values
plt.figure(figsize=(20,10))
plt.plot(cash_actuals_plot2)
plt.show()


#######################
# %% FUNCTIONS SECTION
#######################


def cashed_months_open(cashed):
    open_count = 1
    for b in cashed['Unique_BranchID'].unique():
        for i in cashed.loc[cashed['Unique_BranchID'] == b].index.unique():
            cashed.loc[(cashed['Unique_BranchID'] == b) & (cashed.index == i), 'MonthsOpen'] = open_count
            open_count = open_count + 1
        open_count = 1
    return cashed




def find_nearest_neighbor_season(b):
    check_branches = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163]
    cashed_2 = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2014,1,1))]
    cashed_2 = cashed_2.groupby(['State','Unique_BranchID',cashed_2.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
    cashed_2.index = cashed_2.index.set_levels([cashed_2.index.levels[0], cashed_2.index.levels[1],cashed_2.index.levels[2].to_timestamp()])
    cashed_2 = cashed_2.reset_index()

    cashed_2['Month'] = pd.DatetimeIndex(cashed.CashDate).month
    cashed_2 = cashed_2.groupby(['Unique_BranchID','Month'])['Cashings'].mean().to_frame().reset_index()
    temp = cashed_2.loc[cashed_2['Unique_BranchID'] == b]
    mape_dict = {}
    num = 0
    temp['Month'].unique()

    cashed_2
    for m in temp['Month'].unique():
        for br in check_branches:
            mape = np.abs((temp.loc[temp['Month'] == m]['Cashings'].values[0] - cashed_2.loc[(cashed_2['Unique_BranchID'] == br) & (cashed_2['Month'] == m)]['Cashings'].values[0])/temp.loc[temp['Month'] == m]['Cashings'].values[0])*100
            if num == 0:
                mape_dict[br] = []
                mape_dict[br].append(mape)
            else:
                mape_dict[br].append(mape)
        num+=1
    num = 0
    mape_dict = pd.DataFrame(mape_dict)
    mape_dict
    mean = mape_dict.mean(axis=0).to_frame()
    min = mape_dict.mean(axis=0).to_frame().min().values
    return mean.loc[mean[0] == min[0]].index.values[0]


def find_nearest_neighbor(b):
    check_branches = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163]

    temp = cashed.loc[cashed['Unique_BranchID'] == b]
    mape_dict = {}
    num = 0
    for mo in temp['MonthsOpen'].unique():
        for br in check_branches:
            mape = np.abs((temp.loc[temp['MonthsOpen'] == mo]['Dszn'].values - cashed.loc[(cashed['Unique_BranchID'] == br) & (cashed['MonthsOpen'] == mo)]['Dszn'].values)/temp.loc[temp['MonthsOpen'] == mo]['Dszn'].values)*100
            if num == 0:
                mape_dict[br] = []
                mape_dict[br].append(mape)
            else:
                mape_dict[br].append(mape)
        num+=1
    num = 0
    mape_dict = pd.DataFrame(mape_dict)
    mape_dict
    mean = mape_dict.mean(axis=0).to_frame()
    min = mape_dict.mean(axis=0).to_frame().min().values
    min[0]
    return mean.loc[mean[0] == min[0]].index.values[0]


def create_deseasonal(cashed):
    cashed = cashed_months_open(cashed)
    cashed.CashDate = pd.to_datetime(cashed.CashDate)
    cashed['Month'] = pd.DatetimeIndex(cashed.CashDate).month
    for b in cashed['Unique_BranchID'].unique():
        temp = cashed.loc[cashed['Unique_BranchID'] == b]
        for m in temp['MonthsOpen'].unique():
            if temp['State'].values[0] == 'TX':
                cashed.loc[(cashed['MonthsOpen'] == m) & (cashed['Unique_BranchID'] == b), 'Dszn'] = temp.loc[temp['MonthsOpen'] == m]['Cashings'].values[0]/szn_facts.loc[szn_facts['Month'] == temp.loc[temp['MonthsOpen'] == m]['Month'].values[0]]['KY'].values[0]
            else:
                cashed.loc[(cashed['MonthsOpen'] == m) & (cashed['Unique_BranchID'] == b), 'Dszn'] = temp.loc[temp['MonthsOpen'] == m]['Cashings'].values[0]/szn_facts.loc[szn_facts['Month'] == temp.loc[temp['MonthsOpen'] == m]['Month'].values[0]][temp['State'].values[0]].values[0]

    return cashed


def rolling_average(b, num_forecasts):
    X = cashed.loc[cashed['Unique_BranchID'] == b].Dszn.values
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


def exponetial_smoothing_predictions(b, num_forecasts):
    X = np.asarray(cashed.loc[cashed['Unique_BranchID'] == b]['Dszn'].values)
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
