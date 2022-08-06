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
from statsmodels.tsa.stattools import adfuller
import warnings
import itertools

from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from pylab import rcParams

# %% DATA LOAD

datafolder = pathlib.Path.cwd().parent.parent.parent / 'Data'
outputfolder = pathlib.Path.cwd().parent.parent / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'

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
LC_predictions = pd.DataFrame(columns=['Unique_BranchID','CashDate','Prediction'])

LC_predictions
cashed




##########################################################
#George's new part
#############################


########## removing branches 174, 177, 178, 181, 182, 183, 203, 204, 205, 206, and 207  for now

Newcashed=cashed
badBranches = [174, 177, 178, 181, 182, 183, 203, 204, 205, 206, 207]

for i in badBranches:
    # Get names of indexes for which column Age has value 30
    indexNames = Newcashed[ Newcashed['Unique_BranchID'] == i ].index

    # Delete these row indexes from dataFrame
    Newcashed.drop(indexNames , inplace=True)
#Newcashed
#########################################


cashedGrouped=Newcashed.drop(columns=['State']).groupby('Unique_BranchID')
cashedGrouped.groups.keys()


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
# print('Examples of parameter for SARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))



############################################
#trying to fix the error
# branches  174, 177, 178, 181, 182, 183, 203, 204, 205, 206, and 207 have problems with the indeces

#cashedGrouped.get_group(173)
#cashedGrouped.get_group(177)



#print(len(cashedGrouped.get_group(177)['Cashings']))
#print(len(cashedGrouped.get_group(2)['Cashings']))


warnings.filterwarnings("ignore")
minimumsofar=float('inf')
paramDF= pd.DataFrame(columns=['branch', 'paramofminimum', 'param_seasonalofminimum', 'minimumsofar'])
x=190
while x < 203:
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            mod = sm.tsa.statespace.SARIMAX(cashedGrouped.get_group(x)['Cashings'],order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit(disp=0)
            if(minimumsofar>=results.aic):
                minimumsofar=results.aic
                paramofminimum=param
                param_seasonalofminimum=param_seasonal


    paramDF = paramDF.append({'branch':x, 'paramofminimum':paramofminimum, 'param_seasonalofminimum':param_seasonalofminimum, 'minimumsofar':minimumsofar }, ignore_index=True)
    print(x)
    print('ARIMA{}x{}12 - AIC:{}'.format(paramofminimum,param_seasonalofminimum,minimumsofar))
    x +=1



paramDF

###############
#this gives the optimal parameters for each branch

warnings.filterwarnings("ignore")
paramDFfull= pd.DataFrame(columns=['branch','paramofminimum', 'param_seasonalofminimum', 'minimumsofar'])

for branch in cashedGrouped.groups.keys():

    minimumsofar=float('inf')
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            mod = sm.tsa.statespace.SARIMAX(cashedGrouped.get_group(branch)['Cashings'],order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit(disp=0)
            if(minimumsofar>=results.aic):
                minimumsofar=results.aic
                paramofminimum=param
                param_seasonalofminimum=param_seasonal
    paramDFfull = paramDFfull.append({'branch':branch, 'paramofminimum':paramofminimum, 'param_seasonalofminimum':param_seasonalofminimum, 'minimumsofar':minimumsofar }, ignore_index=True)
    print(branch)
    print('ARIMA{}x{}12 - AIC:{}'.format(paramofminimum,param_seasonalofminimum,minimumsofar))
        # except:
        #     continue


paramDFfull


############################################################
# cashed

#newandformer



y=0
#for branch in cashed['Unique_BranchID'].unique():
for branch in cashedGrouped.groups.keys():
    branchcashed = cashed.loc[cashed.Unique_BranchID==branch][['Cashings']].set_index(cashed.loc[cashed.Unique_BranchID==branch]['CashDate'])
#branchcashed
    try:
        # mod = sm.tsa.statespace.SARIMAX(branchcashed,
        #                                     order=(1, 0, 1),
        #                                     seasonal_order=(1, 1, 0, 12),
        #                                     enforce_stationarity=False,
        #                                     enforce_invertibility=False)
        mod = sm.tsa.statespace.SARIMAX(branchcashed,
                                        order=paramDFfull.loc[y,'paramofminimum'],
                                        seasonal_order=paramDFfull.loc[y,'param_seasonalofminimum'],
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
        y+=1

    except:
        pass

LC_predictions

LC_predictions.to_csv(outputfolder / 'George_LC_branch_sarimax_predictions.csv')
cash_actuals.to_csv(outputfolder / 'George_LC_branch_actuals_for_sarimax.csv')







predic_new=LC_predictions
predic_new


predic_new['CashDate'] = pd.to_datetime(predic_new['PredMonth'])
badBranches2 = [174, 177, 178, 181, 182, 183, 203,204, 205, 206,207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 217, 218, 219, 220, 221, 222, 223, 224]

for i in badBranches2:
    # Get names of indexes for which column Age has value 30
    indexNames = predic_new[ predic_new['Unique_BranchID'] == i ].index

    # Delete these row indexes from dataFrame
    predic_new.drop(indexNames , inplace=True)
predic_new = predic_new.loc[(predic_new['PredMonth'] > dt.datetime(2018,12,1))]
predic_new = predic_new.loc[(predic_new['PredMonth'] < dt.datetime(2019,7,1))]


predic_new



cash_actuals



cashed_new=cash_actuals
badBranches2 = [174, 177, 178, 181, 182, 183, 203,204, 205, 206,207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 217, 218, 219, 220, 221, 222, 223, 224]

for i in badBranches2:
    # Get names of indexes for which column Age has value 30
    indexNames = cashed_new[ cashed_new['Unique_BranchID'] == i ].index

    # Delete these row indexes from dataFrame
    cashed_new.drop(indexNames , inplace=True)

cashed_new['CashDate'] = pd.to_datetime(cashed_new['CashDate'])

cashed_new = cashed_new.loc[(cashed_new['CashDate'] > dt.datetime(2018,12,1))]
cashed_new = cashed_new.loc[(cashed_new['CashDate'] < dt.datetime(2019,7,1))]


cashed_new





result = cashed_new
result
result.insert(2,"Prediction",[0]*len(result['CashDate']))



for i in result.index:
     loc= predic_new.loc[(predic_new['Unique_BranchID']==result['Unique_BranchID'][i])]
     loc=loc.loc[(loc['CashDate']==result['CashDate'][i])]
     result['Prediction'][i]=loc['Prediction'].iloc[0]

result
result.to_csv(outputfolder / 'George_comparison2.csv')



mse = mean_squared_error(result['Prediction'], result['Cashings'])
mse
mae = metrics.mean_absolute_error(result['Prediction'], result['Cashings'])
mae
rmse = np.sqrt(metrics.mean_squared_error(result['Prediction'], result['Cashings']))
rmse

##############################################
#still have to work on this part



# %% DIRECT LOAN APPLICATION PREDICTIONS
DL_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
z=0
#for branch in newandformer['Unique_BranchID'].unique():
for branch in cashedGrouped.groups.keys():
    branchDLs = newandformer.loc[newandformer.Unique_BranchID==branch][['Booked_Indicator']].set_index(newandformer.loc[newandformer.Unique_BranchID==branch]['AppCreatedDate'])
#branchDLs
    try:
        # mod = sm.tsa.statespace.SARIMAX(branchDLs,
        #                                     order=(1, 0, 1),
        #                                     seasonal_order=(1, 1, 0, 12),
        #                                     enforce_stationarity=False,
        #                                     enforce_invertibility=False)
        mod = sm.tsa.statespace.SARIMAX(branchDLs,
                                        order=paramDFfull.loc[y,'paramofminimum'],
                                        seasonal_order=paramDFfull.loc[y,'param_seasonalofminimum'],
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
        z+=1

    except:
        pass



DL_predictions

#DL_predictions.to_csv(outputfolder / 'George_DL_branch_sarimax_predictions.csv')
#newandformer.to_csv(outputfolder / 'George_DL_branch_actuals_for_sarimax.csv')
