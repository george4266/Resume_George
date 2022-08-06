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
# %% Load Data
###############


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
app.loc[(app['AppCreatedDate'] >= dt.datetime(2019,3,1)) & (app['AppCreatedDate'] < dt.datetime(2019,4,1)) & (app['Application_Source'] == 'Direct Loan Renewal'), 'Application_Source'] = 'N'
app.loc[(app['AppCreatedDate'] >= dt.datetime(2019,3,1)) & (app['AppCreatedDate'] < dt.datetime(2019,4,1)) & (app['Application_Source'] == 'New Customer'), 'Application_Source'] = 'Direct Loan Renewal'
app.loc[(app['AppCreatedDate'] >= dt.datetime(2019,3,1)) & (app['AppCreatedDate'] < dt.datetime(2019,4,1)) & (app['Application_Source'] == 'N'), 'Application_Source'] = 'New Customer'

app_edit = app.loc[app['AppCreatedDate'] >= dt.datetime(2019,1,1)]
app_edit.groupby(['Application_Source', app_edit.AppCreatedDate.dt.to_period('M')])['Booked_Indicator'].sum().to_frame().reset_index()



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

newandformer = app.loc[((app['Application_Source']=='New Customer')|(app['Application_Source']=='Former Customer'))]
newandformer = newandformer.groupby(['Unique_BranchID', newandformer.AppCreatedDate.dt.to_period('M')])['Booked_Indicator'].sum().to_frame()
newandformer.index = newandformer.index.set_levels([newandformer.index.levels[0], newandformer.index.levels[1].to_timestamp()])
newandformer = newandformer.reset_index()
newandformer = newandformer.merge(mktg2[['Unique_BranchID','State']].drop_duplicates(), how='left',on='Unique_BranchID')



#################
# %% forecasting
#################


newandformer2 = newandformer.loc[newandformer['AppCreatedDate'] < dt.datetime(2019,1,1)]
newandformer2 = create_deseasonal(newandformer2)
newandformer2

num_forecasts = 6
prediction = []
predictions = {}
for b in newandformer2['Unique_BranchID'].unique():
    if b >= 164 and b < 203:
        x = newandformer2.loc[newandformer2['Unique_BranchID'] == b]
        state = x.iloc[0]['State']
        current_month = x.iloc[-1].MonthsOpen
        #month_open_b = find_nearest_neighbor(b)
        #month_b = find_nearest_neighbor_season(b)
        rolling_pred = rolling_average(b, num_forecasts)
        exp_predictions = exponetial_smoothing_predictions(b, num_forecasts)
        print(b)
        for num in range(1,num_forecasts+1):
            predict_month = num + current_month
            #MonthOpen_num = newandformer2.loc[(newandformer2['MonthsOpen'] == predict_month) & (newandformer2['State'] == state)]['MonthsOpenAvg'].mean()
            #MonthOpen_num = newandformer2.loc[(newandformer2['Unique_BranchID'] == month_open_b) & (newandformer2['MonthsOpen'] == predict_month)]['Booked_Indicator'].values[0]
            if state == 'TX':
                exp_num = szn_facts.loc[szn_facts['Month'] == num]['KY'].values[0]*exp_predictions[num-1]
                rolling_num = szn_facts.loc[szn_facts['Month'] == num]['KY'].values[0]*rolling_pred[num-1]
            else:
                exp_num = szn_facts.loc[szn_facts['Month'] == num][state].values[0]*exp_predictions[num-1]
                rolling_num = szn_facts.loc[szn_facts['Month'] == num][state].values[0]*rolling_pred[num-1]
            #Month_num = newandformer2.loc[(newandformer2['Unique_BranchID'] == month_b) & (newandformer2['Month'] == num)]['Cashings'].values[0]
            #Month_num = newandformer2.loc[(newandformer2['Month'] == num) & (newandformer2['State'] == state)]['MonthsAvg'].mean()
            #prediction.append((MonthOpen_num+Month_num)/2)
            prediction.append((rolling_num+exp_num+MonthOpen_num)/3)
        predictions[b] = prediction
        prediction = []
    else:
        pass

predictions = pd.DataFrame(predictions)
#predictions.to_csv(outputfolder / 'dl_new_expo.csv')
predictions
predict_values = predictions.sum(axis=1, skipna=True).to_frame()

newandformer_actuals = newandformer.loc[(newandformer['AppCreatedDate'] >= dt.datetime(2019,1,1)) & (newandformer['AppCreatedDate'] < dt.datetime(2019,7,1)) & (newandformer['Unique_BranchID'] >= 164) & (newandformer['Unique_BranchID'] < 203)]
newandformer_actuals_plot = newandformer_actuals.groupby(newandformer_actuals.AppCreatedDate.dt.to_period('M'))['Booked_Indicator'].sum().to_frame().reset_index()

actuals_like_forecast = {}
for b in newandformer_actuals['Unique_BranchID'].unique():
    if b >= 164 and b < 203:
        actuals_like_forecast[b] = newandformer_actuals.loc[newandformer_actuals['Unique_BranchID'] == b]['Booked_Indicator'].values
actuals_like_forecast = pd.DataFrame(actuals_like_forecast)

#actuals_like_forecast.to_csv(outputfolder / 'NandF_old_actuals.csv')


#############
# %% RESULTS
#############

plt.plot(predict_values[0].values)
plt.plot(newandformer_actuals_plot.Booked_Indicator.values, color='red')
plt.title('Predicting using state aggreagated by Nearest Neighbor MonthsOpen \n and Month of year nearest neighbor (1/19-9/19)')
plt.show()

MSE = mean_squared_error(newandformer_actuals_plot.Booked_Indicator.values[0:7], predict_values[0].values)
MSE
math.sqrt(MSE)
ME = np.mean(predict_values[0].values - newandformer_actuals_plot.Booked_Indicator.values[0:7])
ME
ME*7/np.sum(newandformer_actuals_plot.Booked_Indicator.values[0:7])*100
MAPE = np.mean(np.abs((newandformer_actuals_plot.Booked_Indicator.values[0:7] - predict_values[0].values) / newandformer_actuals_plot.Booked_Indicator.values[0:7])) * 100
MAPE

np.sum(predict_values[0].values)
np.sum(newandformer_actuals_plot.Booked_Indicator.values[0:7])
np.sum(predict_values[0].values) - np.sum(newandformer_actuals_plot.Booked_Indicator.values[0:7])

###############
# %% FUNCTIONS
###############


def months_open(newandformer2):
    open_count = 1
    for b in newandformer2['Unique_BranchID'].unique():
        for i in newandformer2.loc[newandformer2['Unique_BranchID'] == b].index.unique():
            newandformer2.loc[(newandformer2['Unique_BranchID'] == b) & (newandformer2.index == i), 'MonthsOpen'] = open_count
            open_count = open_count + 1
        open_count = 1
    return newandformer2



def create_deseasonal(newandformer2):
    newandformer2 = months_open(newandformer2)
    newandformer2.AppCreatedDate = pd.to_datetime(newandformer2.AppCreatedDate)
    newandformer2['Month'] = pd.DatetimeIndex(newandformer2.AppCreatedDate).month
    for b in newandformer2['Unique_BranchID'].unique():
        temp = newandformer2.loc[newandformer2['Unique_BranchID'] == b]
        for m in temp['MonthsOpen'].unique():
            if temp['State'].values[0] == 'TX':
                newandformer2.loc[(newandformer2['MonthsOpen'] == m) & (newandformer2['Unique_BranchID'] == b), 'Dszn'] = temp.loc[temp['MonthsOpen'] == m]['Booked_Indicator'].values[0]/szn_facts.loc[szn_facts['Month'] == temp.loc[temp['MonthsOpen'] == m]['Month'].values[0]]['KY'].values[0]
            else:
                newandformer2.loc[(newandformer2['MonthsOpen'] == m) & (newandformer2['Unique_BranchID'] == b), 'Dszn'] = temp.loc[temp['MonthsOpen'] == m]['Booked_Indicator'].values[0]/szn_facts.loc[szn_facts['Month'] == temp.loc[temp['MonthsOpen'] == m]['Month'].values[0]][temp['State'].values[0]].values[0]

    return newandformer2


def rolling_average(b, num_forecasts):
    X = newandformer2.loc[newandformer2['Unique_BranchID'] == b].Dszn.values
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
    X = np.asarray(newandformer2.loc[newandformer2['Unique_BranchID'] == b]['Dszn'].values)
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

newandformer2

def find_nearest_neighbor(b):
    check_branches = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163]

    temp = newandformer2.loc[newandformer2['Unique_BranchID'] == 164]
    mape_dict = {}
    num = 0
    for mo in temp['MonthsOpen'].unique():
        for br in check_branches:
            mape = np.abs((temp.loc[temp['MonthsOpen'] == mo]['Dszn'].values - newandformer2.loc[(newandformer2['Unique_BranchID'] == br) & (newandformer2['MonthsOpen'] == mo)]['Dszn'].values)/temp.loc[temp['MonthsOpen'] == mo]['Dszn'].values)*100
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
    mean.loc[mean[0] == min[0]].index.values[0]
    return mean.loc[mean[0] == min[0]].index.values[0]
