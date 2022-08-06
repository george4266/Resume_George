################################################################################

# %% forecasting_old_branches.py

# DESCRIPTION:
# File generates forecasts for live check cashings and new incoming direct loans
# for branches 1-163. Also generates forecasts for live check conversions, direct
# loan renewals, and attritions (early exit from portfolio) for all branches.

# FILE OUTPUTS:
# Separate csv files for each forecast type, with following columns -
# ['Unique_BranchID','PredMonth','Prediciton']
# !!!!!!!!!!! csv files labeled by prediction window

################################################################################
import datetime as dt, numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
import time, pathlib, math
from fbprophet import Prophet

# FORECASTING MONTH START AND WINDOW
#month = input('Input current month (MM-DD-YYYY format): ')
#window = input('Input number of months to forecast: ')
month = '12-01-2019'
dtmonth = pd.to_datetime(month)
window = 6

# DATA FOLDER AND OUTPUT LOCATIONS
outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'output'
datafolder = pathlib.Path.cwd().parent / 'Data'

# Marketing table operations, including adjusting issue date for estimated two-week
# offset for a live check cashing
mktgfile = datafolder / 'VT_Marketing_11012019.txt'
mktg = pd.read_csv(mktgfile, sep=',', low_memory=False)
mktg['CashDate'] = pd.to_datetime(mktg['CashDate'], errors='coerce')
mktg['IssueDate'] = pd.to_datetime(mktg['IssueDate'], errors='coerce')
mktg = mktg[['CashDate','State','Cashings','Mailings','Unique_BranchID','IssueDate']]
mktg['Adjusted_Issue_Date'] = (mktg['IssueDate'] + dt.timedelta(days = 15)) + pd.DateOffset(day=1)

appfile = datafolder / 'VT_Applications_11262019.txt'
app = pd.read_csv(appfile, sep=',', low_memory=False)

# Performance table operations
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))
perf = perf[['Unique_ContractID','Unique_BranchID','MonthsOnBook','ProcessStatus']]

# Origination table operations
origination_file = datafolder/ 'VT_Originations_11262019.txt'
origin = pd.read_csv(origination_file, sep=',', low_memory=False)
origin = origin[['Unique_ContractID','Unique_BranchID','BookDate','ProductType','Term']]

################################################################################

# %% GENERATING TRAINING DATASETS

################################################################################

mktg['CashDate'] = pd.to_datetime(mktg['CashDate'], errors='coerce')
mktg['IssueDate'] = pd.to_datetime(mktg['IssueDate'], errors='coerce')
mktg = mktg[['CashDate','State','Cashings','Mailings','Unique_BranchID','IssueDate']]
mktg['Adjusted_Issue_Date'] = (mktg['IssueDate'] + dt.timedelta(days = 15)) + pd.DateOffset(day=1)
responserates = mktg.groupby(['Unique_BranchID','Adjusted_Issue_Date']).sum().reset_index()
responserates['ResponseRate'] = responserates['Cashings'] / responserates['Mailings']
mktg = mktg.dropna()

app['AppCreatedDate'] = pd.to_datetime(app['AppCreatedDate'])
app = app[['AppCreatedDate', 'Booked_Indicator', 'Unique_BranchID','Unique_ApplicationID','Application_Source']]
app = app.dropna()

cashed = mktg.loc[(mktg['CashDate'] < (dtmonth + pd.DateOffset(months=1))) & (mktg['CashDate'] >= dt.datetime(2015,1,1))]
cashed = cashed.groupby(['State','Unique_BranchID',cashed.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame().reset_index()
cashed['CashDate'] = cashed.CashDate.dt.to_timestamp()

newandformer = app.loc[((app['Application_Source']=='New Customer')|(app['Application_Source']=='Former Customer'))]
newandformer = newandformer.groupby(['Unique_BranchID', newandformer.AppCreatedDate.dt.to_period('M')])['Booked_Indicator'].sum().to_frame().reset_index()
newandformer['AppCreatedDate'] = newandformer.AppCreatedDate.dt.to_timestamp()

RR_train = responserates.loc[responserates.Adjusted_Issue_Date < (dtmonth + pd.DateOffset(months=1))]
LC_train = cashed.loc[cashed.CashDate<(dtmonth + pd.DateOffset(months=1))]
DL_train = newandformer.loc[newandformer.AppCreatedDate<(dtmonth + pd.DateOffset(months=1))]
R_LC_train = renewals_lc.loc[renewals_lc['CurrentMonth']< (dtmonth + pd.DateOffset(months=1))]
R_DL_train = renewals_lc.loc[renewals_dl['CurrentMonth']< (dtmonth + pd.DateOffset(months=1))]
AT_train = attrition.loc[attrition['CurrentMonth']< (dtmonth + pd.DateOffset(months=1))]

# %% PROPHET DL RENEWALS
R_DL_start = time.time()
R_DL_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
ds = ['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019']
future = pd.DataFrame(ds,columns=['ds'])

for branch in R_DL_train['Unique_BranchID'].unique():
    branch_train = R_DL_train.loc[R_DL_train.Unique_BranchID==branch][['CurrentMonth','RenewalRate']].rename(columns={'CurrentMonth':'ds','RenewalRate':'y'})

    try:
        prophet = Prophet(seasonality_mode='multiplicative',daily_seasonality=False,weekly_seasonality=False)
        prophet.fit(branch_train)
        #future = prophet.make_future_dataframe(periods=6,freq='M')
        pred = prophet.predict(future)
        pred = pred.loc[(pred.ds<='09-01-2019')&(pred.ds>='01-01-2019')][['ds','yhat']]

        temp = {'PredMonth':pred.ds,'Prediction':pred.yhat}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        R_DL_predictions = R_DL_predictions.append(temp)

    except:
        pass
R_DL_end = time.time()

# %% PROPHET LC CONVERSIONS
R_LC_start = time.time()
R_LC_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
ds = ['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019']
future = pd.DataFrame(ds,columns=['ds'])

for branch in R_LC_train['Unique_BranchID'].unique():
    branch_train = R_LC_train.loc[R_LC_train.Unique_BranchID==branch][['CurrentMonth','RenewalRate']].rename(columns={'CurrentMonth':'ds','RenewalRate':'y'})

    try:
        prophet = Prophet(seasonality_mode='multiplicative',daily_seasonality=False,weekly_seasonality=False)
        prophet.fit(branch_train)
        #future = prophet.make_future_dataframe(periods=6,freq='M')
        pred = prophet.predict(future)
        pred = pred.loc[(pred.ds<='09-01-2019')&(pred.ds>='01-01-2019')][['ds','yhat']]

        temp = {'PredMonth':pred.ds,'Prediction':pred.yhat}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        R_LC_predictions = R_LC_predictions.append(temp)

    except:
        pass
R_LC_end = time.time()

# %% PROPHET ATTRITION
AT_start = time.time()
AT_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
ds = ['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019']
future = pd.DataFrame(ds,columns=['ds'])

for branch in AT_train['Unique_BranchID'].unique():
    branch_train = AT_train.loc[AT_train.Unique_BranchID==branch][['CurrentMonth','AttritionRate']].rename(columns={'CurrentMonth':'ds','AttritionRate':'y'})

    try:
        prophet = Prophet(seasonality_mode='multiplicative',daily_seasonality=False,weekly_seasonality=False)
        prophet.fit(branch_train)
        #future = prophet.make_future_dataframe(periods=6,freq='M')
        pred = prophet.predict(future)
        pred = pred.loc[(pred.ds<='09-01-2019')&(pred.ds>='01-01-2019')][['ds','yhat']]

        temp = {'PredMonth':pred.ds,'Prediction':pred.yhat}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        AT_predictions = AT_predictions.append(temp)

    except:
        pass
AT_end = time.time()

# %% PROPHET RESPONSE RATES
RR_start = time.time()
RR_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
ds = ['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019']
future = pd.DataFrame(ds,columns=['ds'])

for branch in RR_train['Unique_BranchID'].unique():
    branch_train = RR_train.loc[RR_train.Unique_BranchID==branch][['Adjusted_Issue_Date','ResponseRate']].rename(columns={'Adjusted_Issue_Date':'ds','ResponseRate':'y'})

    try:
        prophet = Prophet(seasonality_mode='multiplicative',daily_seasonality=False,weekly_seasonality=False)
        prophet.fit(branch_train)
        #future = prophet.make_future_dataframe(periods=6,freq='M')
        pred = prophet.predict(future)
        pred = pred.loc[(pred.ds<='09-01-2019')&(pred.ds>='01-01-2019')][['ds','yhat']]

        temp = {'PredMonth':pred.ds,'Prediction':pred.yhat}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        RR_predictions = RR_predictions.append(temp)

    except:
        pass
RR_end = time.time()

# %% DIRECT LOAN APPLICATION PROPHET PREDICTIONS
DL_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
for branch in DL_train['Unique_BranchID'].unique():
    branchDLs = DL_train.loc[DL_train.Unique_BranchID==branch][['AppCreatedDate','Booked_Indicator']].rename(columns={'AppCreatedDate':'ds','Booked_Indicator':'y'})

    try:
        prophet = Prophet(seasonality_mode='multiplicative',daily_seasonality=False, weekly_seasonality=False)
        prophet.fit(branchDLs)
        #future = prophet.make_future_dataframe(periods=6,freq='M')
        ds = ['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019']
        future = pd.DataFrame(ds,columns=['ds'])
        pred = prophet.predict(future)
        pred = pred.loc[(pred.ds<='09-01-2019')&(pred.ds>='01-01-2019')][['ds','yhat']]

        temp = {'PredMonth':pred.ds,'Prediction':pred.yhat}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        DL_predictions = DL_predictions.append(temp)

    except:
        pass


# %% OUTPUT PREDICITION FILES
R_LC_predictions.to_csv(outputfolder/pathlib.Path('LC_conversion_predictions_Jan2019_Mar2019.csv'))
R_DL_predictions.to_csv(outputfolder/pathlib.Path('DL_renewal_predictions_Jan2019_Mar2019.csv'))
AT_predictions.to_csv(outputfolder/pathlib.Path('attrition_predictions_Jan2019_Mar2019.csv'))
