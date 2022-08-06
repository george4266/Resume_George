# %% Imports and file load
import pathlib
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from pmdarima import auto_arima
import itertools
import time
import math

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import metrics

from fbprophet import Prophet

# %% DATA LOAD

outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'output'
datafolder = pathlib.Path.cwd().parent / 'Data'

mktgfile = datafolder / 'VT_Marketing_11012019.txt'
mktg = pd.read_csv(mktgfile, sep=',', low_memory=False)
appfile = datafolder / 'VT_Applications_11262019.txt'
app = pd.read_csv(appfile, sep=',', low_memory=False)

origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'

perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))

perf = perf[['Unique_ContractID','Unique_BranchID','MonthsOnBook','ProcessStatus']]
origin = pd.read_csv(origination_file, sep=',', low_memory=False)
origin = origin[['Unique_ContractID','Unique_BranchID','BookDate','ProductType','Term']]

# %% Generating datasets

portfolio = origin.merge(perf, on=['Unique_ContractID','Unique_BranchID'], how='left')
portfolio['BookDate'] = pd.to_datetime(portfolio['BookDate'])
portfolio['months_added'] = pd.to_timedelta(portfolio['MonthsOnBook'], 'M')
portfolio['CurrentMonth'] = portfolio['BookDate'] + portfolio['months_added']
portfolio['CurrentMonth'] = portfolio.CurrentMonth.dt.to_period('M').dt.to_timestamp()
portfolio2 = portfolio[['CurrentMonth', 'ProcessStatus', 'Unique_BranchID', 'MonthsOnBook', 'Term','ProductType']]
portfolio2.dropna(inplace=True)
#portfolio2 = portfolio2.loc[portfolio2['CurrentMonth'] < dt.datetime(2019,1,1)]
portfolio2.head()

renewals_dl = portfolio2.loc[(portfolio2['ProcessStatus'] == 'Renewed') & (portfolio2['ProductType'] !='LC')].groupby(['Unique_BranchID',portfolio2.CurrentMonth.dt.to_period('M')])['ProcessStatus'].count().to_frame().reset_index().rename(columns={'ProcessStatus':'Renewals'})
renewals_lc = portfolio2.loc[(portfolio2['ProcessStatus'] == 'Renewed') & (portfolio2['ProductType'] =='LC')].groupby(['Unique_BranchID',portfolio2.CurrentMonth.dt.to_period('M')])['ProcessStatus'].count().to_frame().reset_index().rename(columns={'ProcessStatus':'Renewals'})

non_renewals_dl = portfolio2.loc[(portfolio2['ProcessStatus'] != 'Renewed') & (portfolio2['ProductType'] !='LC')].groupby(['Unique_BranchID',portfolio2.CurrentMonth.dt.to_period('M')])['ProcessStatus'].count().to_frame().reset_index().rename(columns={'ProcessStatus':'NonRenewals'})
non_renewals_lc = portfolio2.loc[(portfolio2['ProcessStatus'] != 'Renewed') & (portfolio2['ProductType'] =='LC')].groupby(['Unique_BranchID',portfolio2.CurrentMonth.dt.to_period('M')])['ProcessStatus'].count().to_frame().reset_index().rename(columns={'ProcessStatus':'NonRenewals'})

renewals_dl = renewals_dl.merge(non_renewals_dl,on=['Unique_BranchID','CurrentMonth'],how='right')
renewals_dl.loc[pd.isnull(renewals_dl.Renewals),'Renewals'] = 0
renewals_dl['RenewalRate'] = renewals_dl['Renewals']/(renewals_dl['NonRenewals'] + renewals_dl['Renewals'])
renewals_dl['CurrentMonth'] = renewals_dl['CurrentMonth'].dt.to_timestamp()
#sns.lineplot(x='CurrentMonth', y='RenewalRate', data=renewals_dl)

renewals_lc = renewals_lc.merge(non_renewals_lc,on=['Unique_BranchID','CurrentMonth'],how='right')
renewals_lc.loc[pd.isnull(renewals_lc.Renewals),'Renewals'] = 0
renewals_lc['RenewalRate'] = renewals_lc['Renewals']/(renewals_lc['NonRenewals'] + renewals_lc['Renewals'])
renewals_lc['CurrentMonth'] = renewals_lc['CurrentMonth'].dt.to_timestamp()
#sns.lineplot(x='CurrentMonth', y='RenewalRate', data=renewals_lc)

attrition = portfolio.loc[(portfolio['ProcessStatus'] == 'Closed') & (portfolio['Term'] > portfolio['MonthsOnBook'])].groupby(['Unique_BranchID',portfolio.CurrentMonth.dt.to_period('M')])['ProcessStatus'].count().to_frame().reset_index().rename(columns={'ProcessStatus':'Attritions'})
non_attrition = portfolio2.loc[(portfolio2['ProcessStatus'] != 'Closed') | (portfolio2['ProcessStatus'] == 'Closed') & (portfolio2['Term'] == portfolio2['MonthsOnBook'])].groupby(['Unique_BranchID',portfolio2.CurrentMonth.dt.to_period('M')])['ProcessStatus'].count().to_frame().reset_index()
non_attrition = non_attrition.rename(columns={'ProcessStatus':'NonAttritions'})

attrition = attrition.merge(non_attrition,on=['Unique_BranchID','CurrentMonth'],how='right')
attrition.loc[pd.isnull(attrition.Attritions),'Attritions'] = 0
attrition['AttritionRate'] = attrition['Attritions']/(attrition['NonAttritions']+attrition['Attritions'])
attrition['CurrentMonth'] = attrition['CurrentMonth'].dt.to_timestamp()

mktg['CashDate'] = pd.to_datetime(mktg['CashDate'], errors='coerce')
mktg['IssueDate'] = pd.to_datetime(mktg['IssueDate'], errors='coerce')
mktg = mktg[['CashDate','State','Cashings','Mailings','Unique_BranchID','IssueDate']]
mktg['Adjusted_Issue_Date'] = mktg['IssueDate'] + dt.timedelta(days = 15)
mktg['Adjusted_Issue_Date'] = pd.to_datetime(mktg['Adjusted_Issue_Date']).dt.to_period('M')
responserates = mktg.groupby(['Unique_BranchID','Adjusted_Issue_Date']).sum().reset_index()
responserates['ResponseRate'] = responserates['Cashings'] / responserates['Mailings']
responserates['Adjusted_Issue_Date'] = responserates.Adjusted_Issue_Date.dt.to_timestamp()
mktg = mktg.dropna()

app['AppCreatedDate'] = pd.to_datetime(app['AppCreatedDate'])
app = app[['AppCreatedDate', 'Booked_Indicator', 'Unique_BranchID','Unique_ApplicationID','Application_Source']]
app = app.dropna()

cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1))]
cashed = cashed.groupby(['State','Unique_BranchID',cashed.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame().reset_index()
cashed['CashDate'] = cashed.CashDate.dt.to_timestamp()

newandformer = app.loc[((app['Application_Source']=='New Customer')|(app['Application_Source']=='Former Customer'))]
newandformer = newandformer.groupby(['Unique_BranchID', newandformer.AppCreatedDate.dt.to_period('M')])['Booked_Indicator'].sum().to_frame().reset_index()
newandformer['AppCreatedDate'] = newandformer.AppCreatedDate.dt.to_timestamp()

# %% Test groups

RR_train = responserates.loc[responserates.Adjusted_Issue_Date < pd.to_datetime('01-01-2019')]
LC_train = cashed.loc[cashed.CashDate<pd.to_datetime('01-01-2019')]
DL_train = newandformer.loc[newandformer.AppCreatedDate<pd.to_datetime('01-01-2019')]
R_LC_train = renewals_lc.loc[renewals_lc['CurrentMonth']< pd.to_datetime('01-01-2019')]
R_DL_train = renewals_lc.loc[renewals_dl['CurrentMonth']< pd.to_datetime('01-01-2019')]
AT_train = attrition.loc[attrition['CurrentMonth']< pd.to_datetime('01-01-2019')]

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

# %% TESTING RESULTS
def testing_LC(LC_predictions,start,end):
    table = cash_actuals.merge(prophet_LC_predictions, left_on=['Unique_BranchID','Date'],right_on=['Unique_BranchID','PredMonth'],how='inner').drop(columns='Date')
    table = table.loc[table['Unique_BranchID']<=163]
    table = table.loc[table['PredMonth']<pd.to-datetime('07-01-19')]

    table_agg = table.groupby(['PredMonth']).sum()

    table['Error'] = table.Prediction - table.Cashings
    table['AbsError'] = abs(table.Prediction - table.Cashings)
    table['AbsPerError'] = abs(table.Prediction - table.Cashings)/table.Cashings
    table['SquaredError'] = (table.Prediction - table.Cashings)**2
    table_agg['Error'] = table_agg.Prediction - table_agg.Cashings
    table_agg['AbsError'] = abs(table_agg.Prediction - table_agg.Cashings)
    table_agg['AbsPerError'] = abs(table_agg.Prediction - table_agg.Cashings)/table_agg.Cashings
    table_agg['SquaredError'] = (table_agg.Prediction - table_agg.Cashings)**2

    print('Runtime: {}s\n'.format(start-end))
    print('Aggregates:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table_agg['Error'].mean(),table_agg['AbsError'].mean(),table_agg['AbsPerError'].mean(),table_agg['SquaredError'].mean(),math.sqrt(table_agg['SquaredError'].mean())))
    print('Branch Averages:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table['Error'].mean(),table['AbsError'].mean(),table['AbsPerError'].mean(),table['SquaredError'].mean(),math.sqrt(table['SquaredError'].mean())))
    print(table_agg)
    print('\n')
    print(table.groupby(['Unique_BranchID']).mean().head())
def testing_RR(RR_predictions,start,end):
    RR_predictions['PredMonth'] = pd.to_datetime(RR_predictions['PredMonth'])
    table = responserates.merge(RR_predictions, left_on=['Unique_BranchID','Adjusted_Issue_Date'],right_on=['Unique_BranchID','PredMonth'],how='inner').drop(columns='Adjusted_Issue_Date')
    #table = table.loc[table['Unique_BranchID']<=163]
    #table = table.loc[table['PredMonth'] != '2019-03-01']
    table['PredictedCashings'] = table['Prediction'] * table['Mailings']

    table_agg = table.groupby('PredMonth').sum()

    table ['Error'] = table.PredictedCashings - table.Cashings
    table ['AbsError'] = abs(table.PredictedCashings - table.Cashings)
    table ['AbsPerError'] = abs(table.PredictedCashings - table.Cashings)/table.Cashings
    table['SquaredError'] = (table.PredictedCashings - table.Cashings)**2
    table_agg['Error'] = table_agg.PredictedCashings - table_agg.Cashings
    table_agg['AbsError'] = abs(table_agg.PredictedCashings - table_agg.Cashings)
    table_agg['AbsPerError'] = abs(table_agg.PredictedCashings - table_agg.Cashings)/table_agg.Cashings
    table_agg['SquaredError'] = (table_agg.PredictedCashings - table_agg.Cashings)**2

    print('Runtime: {}s\n'.format(start-end))
    print('Aggregates:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table_agg['Error'].mean(),table_agg['AbsError'].mean(),table_agg['AbsPerError'].mean(),table_agg['SquaredError'].mean(),math.sqrt(table_agg['SquaredError'].mean())))
    print('Branch Averages:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table['Error'].mean(),table['AbsError'].mean(),table['AbsPerError'].mean(),table['SquaredError'].mean(),math.sqrt(table['SquaredError'].mean())))
    print(table_agg)
    print('\n')
    print(table.groupby(['Unique_BranchID']).mean().head())
def testing_DL(DL_predictions,start,end):
    DL_predictions['PredMonth'] = pd.to_datetime(DL_predictions['PredMonth'])
    table = newandformer.merge(DL_predictions, left_on=['Unique_BranchID','AppCreatedDate'],right_on=['Unique_BranchID','PredMonth'],how='inner').drop(columns='AppCreatedDate')
    table = table.loc[table['Unique_BranchID']<=163]
    table = table.loc[table['PredMonth'] < pd.to_datetime('2019-03-01')]

    table_agg = table.groupby('PredMonth').sum()

    table ['Error'] = table.Prediction - table.Booked_Indicator
    table ['AbsError'] = abs(table.Prediction - table.Booked_Indicator)
    table ['AbsPerError'] = abs(table.Prediction - table.Booked_Indicator)/table.Booked_Indicator
    table['SquaredError'] = (table.Prediction - table.Booked_Indicator)**2
    table_agg['Error'] = table_agg.Prediction - table_agg.Booked_Indicator
    table_agg['AbsError'] = abs(table_agg.Prediction - table_agg.Booked_Indicator)
    table_agg['AbsPerError'] = abs(table_agg.Prediction - table_agg.Booked_Indicator)/table_agg.Booked_Indicator
    table_agg['SquaredError'] = (table_agg.Prediction - table_agg.Booked_Indicator)**2

    print('Runtime: {}s\n'.format(start-end))
    print('Aggregates:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table_agg['Error'].mean(),table_agg['AbsError'].mean(),table_agg['AbsPerError'].mean(),table_agg['SquaredError'].mean(),math.sqrt(table_agg['SquaredError'].mean())))
    print('Branch Averages:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table['Error'].mean(),table['AbsError'].mean(),table['AbsPerError'].mean(),table['SquaredError'].mean(),math.sqrt(table['SquaredError'].mean())))
    print(table_agg)
    print('\n')
    print(table.groupby(['Unique_BranchID']).mean().head())
def testing_R_LC(R_LC_predictions,start,end):
    R_LC_predictions['PredMonth'] = pd.to_datetime(R_LC_predictions['PredMonth'])
    table = renewals_lc.merge(R_LC_predictions, left_on=['Unique_BranchID','CurrentMonth'],right_on=['Unique_BranchID','PredMonth'],how='inner').drop(columns='CurrentMonth')
    #table = table.loc[table['Unique_BranchID']<=163]
    table['PredictedRenewals'] = table['Prediction'] * (table['Renewals']+table['NonRenewals'])

    table_agg = table.groupby('PredMonth').sum()

    table['Error'] = table.PredictedRenewals - table.Renewals
    table['AbsError'] = abs(table.PredictedRenewals - table.Renewals)
    table['AbsPerError'] = abs(table.PredictedRenewals - table.Renewals)/table.Renewals
    table['SquaredError'] = (table.PredictedRenewals - table.Renewals)**2
    table_agg['Error'] = table_agg.PredictedRenewals - table_agg.Renewals
    table_agg['AbsError'] = abs(table_agg.PredictedRenewals - table_agg.Renewals)
    table_agg['AbsPerError'] = abs(table_agg.PredictedRenewals - table_agg.Renewals)/table_agg.Renewals
    table_agg['SquaredError'] =  (table_agg.PredictedRenewals - table_agg.Renewals)**2

    print('Runtime: {}s\n'.format(start-end))
    print('Aggregates:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table_agg['Error'].mean(),table_agg['AbsError'].mean(),table_agg['AbsPerError'].mean(),table_agg['SquaredError'].mean(),math.sqrt(table_agg['SquaredError'].mean())))
    print('Branch Averages:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table['Error'].mean(),table['AbsError'].mean(),table['AbsPerError'].mean(),table['SquaredError'].mean(),math.sqrt(table['SquaredError'].mean())))
    print(table_agg)
    print('\n')
    print(table.groupby(['Unique_BranchID']).mean().head())
def testing_R_DL(R_DL_predicitions,start,end):
    table = renewals_dl.merge(R_DL_predictions, left_on=['Unique_BranchID','CurrentMonth'],right_on=['Unique_BranchID','PredMonth'],how='inner').drop(columns='CurrentMonth')
    #table = table.loc[table['Unique_BranchID']<=163]
    table['PredictedRenewals'] = table['Prediction'] * (table['Renewals']+table['NonRenewals'])

    table_agg = table.groupby('PredMonth').sum()

    table['Error'] = table.PredictedRenewals - table.Renewals
    table['AbsError'] = abs(table.PredictedRenewals - table.Renewals)
    table['AbsPerError'] = abs(table.PredictedRenewals - table.Renewals)/table.Renewals
    table['SquaredError'] = (table.PredictedRenewals - table.Renewals)**2
    table_agg['Error'] = table_agg.PredictedRenewals - table_agg.Renewals
    table_agg['AbsError'] = abs(table_agg.PredictedRenewals - table_agg.Renewals)
    table_agg['AbsPerError'] = abs(table_agg.PredictedRenewals - table_agg.Renewals)/table_agg.Renewals
    table_agg['SquaredError'] =  (table_agg.PredictedRenewals - table_agg.Renewals)**2

    print('Runtime: {}s\n'.format(start-end))
    print('Aggregates:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table_agg['Error'].mean(),table_agg['AbsError'].mean(),table_agg['AbsPerError'].mean(),table_agg['SquaredError'].mean(),math.sqrt(table_agg['SquaredError'].mean())))
    print('Branch Averages:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table['Error'].mean(),table['AbsError'].mean(),table['AbsPerError'].mean(),table['SquaredError'].mean(),math.sqrt(table['SquaredError'].mean())))
    print(table_agg)
    print('\n')
    print(table.groupby(['Unique_BranchID']).mean().head())
def testing_AT(AT_predictions,start,end):
    AT_predictions['PredMonth'] = pd.to_datetime(R_LC_predictions['PredMonth'])
    table = attrition.merge(AT_predictions, left_on=['Unique_BranchID','CurrentMonth'],right_on=['Unique_BranchID','PredMonth'],how='inner').drop(columns='CurrentMonth')
    #table = table.loc[table['Unique_BranchID']<=163]
    table['PredictedAttritions'] = table['Prediction'] * (table['Attritions']+table['NonAttritions'])
    table_agg = table.groupby('PredMonth').sum()

    table['Error'] = table.PredictedAttritions - table.Attritions
    table['AbsError'] = abs(table.PredictedAttritions - table.Attritions)
    table['AbsPerError'] = abs(table.PredictedAttritions - table.Attritions)/table.Attritions
    table['SquaredError'] = (table.PredictedAttritions - table.Attritions)**2
    table_agg['Error'] = table_agg.PredictedAttritions - table_agg.Attritions
    table_agg['AbsError'] = abs(table_agg.PredictedAttritions - table_agg.Attritions)
    table_agg['AbsPerError'] = abs(table_agg.PredictedAttritions - table_agg.Attritions)/table_agg.Attritions
    table_agg['SquaredError'] = (table_agg.PredictedAttritions - table_agg.Attritions)**2
    print('Runtime: {}s\n'.format(start-end))
    print('Aggregates:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table_agg['Error'].mean(),table_agg['AbsError'].mean(),table_agg['AbsPerError'].mean(),table_agg['SquaredError'].mean(),math.sqrt(table_agg['SquaredError'].mean())))
    print('Branch Averages:\nME - {}\nMAE - {}\nMAPE - {}\nMSE - {}\nRMSE - {}\n'.format(table['Error'].mean(),table['AbsError'].mean(),table['AbsPerError'].mean(),table['SquaredError'].mean(),math.sqrt(table['SquaredError'].mean())))
    print(table_agg)
    print('\n')
    print(table.groupby(['Unique_BranchID']).mean().head())

print('*****PROPHET LC CONVERSION PREDICTIONS*****\n')
testing_R_LC(R_LC_predictions,R_LC_start,R_LC_end)
print('\n\n')

print('*****PROPHET DL RENEWAL PREDICTIONS*****\n')
testing_R_DL(R_DL_predictions,R_DL_start,R_DL_end)
print('\n\n')

print('*****PROPHET ATTRITION RATE PREDICTIONS*****\n')
testing_AT(AT_predictions,AT_start,AT_end)
print('\n\n')

print('*****PROPHET LC RESPONSE RATE METHOD PREDICTIONS*****\n')
testing_RR(RR_predictions,RR_start,RR_end)
print('\n\n')

DL_start =1
DL_end=1
print('*****PROPHET NEW LOAN PREDICTIONS*****\n')
testing_DL(DL_predictions,DL_start,DL_end)
print('\n\n')

# %%

print('*****PROPHET LC CASHING PREDICTIONS*****\n')
testing_LC(prophet_LC_predictions,prophet_LC_start,prophet_LC_end)
print('\n\n')

print('*****AUTOARIMA LC CASHING PREDICTIONS*****\n')
testing_LC(aarima_LC_predictions,aarima_LC_start,prophet_LC_end)
print('\n\n')

print('*****ORIGINAL LC CASHING PREDICTIONS*****\n')
testing_LC(sarima_LC_predictions,sarima_LC_start,prophet_LC_end)
print('\n\n')

# %% OUTPUT PREDICITION FILES
R_LC_predictions.to_csv(outputfolder/pathlib.Path('LC_conversion_predictions.csv'))
R_DL_predictions.to_csv(outputfolder/pathlib.Path('DL_renewal_predictions.csv'))
AT_predictions.to_csv(outputfolder/pathlib.Path('attrition_predictions.csv'))
RR_predictions = responserates.merge(RR_predictions, left_on=['Unique_BranchID','Adjusted_Issue_Date'],right_on=['Unique_BranchID','PredMonth'],how='inner').drop(columns='Adjusted_Issue_Date')
RR_predictions['Prediction'] = RR_predictions['Prediction'] * RR_predictions['Mailings']
RR_predictions = RR_predictions[['Unique_BranchID','PredMonth','Prediction']]
RR_predictions.to_csv(outputfolder/pathlib.Path('LC_cashing_predictions.csv'))
DL_predictions.to_csv(outputfolder/pathlib.Path('DL_newformer_predictions.csv '))

###############################################################################
#
# NON-PROHPET, AKA HERETIC MATERIAL BELOW
#
################################################################################

# %% ORIGINAL SARIMAX
sarima_LC_start = time.time()
sarima_LC_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
for branch in LC_train['Unique_BranchID'].unique():
    branch_train = LC_train.loc[LC_train.Unique_BranchID==branch][['Cashings']].set_index(LC_train.loc[LC_train.Unique_BranchID==branch]['Date'])

    try:
        mod = SARIMAX(branch_train.asfreq('MS'),
                                            order=(1, 0, 1),
                                            seasonal_order=(1, 1, 0, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                            suppress_warnings=True)
        results = mod.fit()

        pred = results.forecast(steps=9)
        temp = {'PredMonth':['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019'],'Prediction':list(pred)}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        sarima_LC_predictions = sarima_LC_predictions.append(temp,sort=True)

    except:
        pass
sarima_LC_end = time.time()

# %% GRIDSEARCH SARIMAX
grid_LC_start = time.time()
grid_LC_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
for branch in LC_train['Unique_BranchID'].unique():
    branch_train = LC_train.loc[LC_train.Unique_BranchID==branch][['Cashings']].set_index(LC_train.loc[LC_train.Unique_BranchID==branch]['Date'])

    try:
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        aic = 10000

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = SARIMAX(branch_train.asfreq('MS'),
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False,
                                                    suppress_warnings=True)
                    results = mod.fit()
                    if results.aic < aic:
                        fin_param = param
                        fin_seasonal = param_seasonal
                        aic = results.aic
                except:
                    continue

        mod = SARIMAX(branch_train.asfreq('MS'),
                                        order=fin_param,
                                        seasonal_order=fin_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                        suppress_warnings=True)
        results = mod.fit()

        pred = results.forecast(steps=9)
        temp = {'PredMonth':['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019'],'Prediction':list(pred)}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        grid_LC_predictions = grid_LC_predictions.append(temp,sort=True)

    except:
        pass
grid_LC_end = time.time()

# %% AUTO ARIMA LIVE CHECK CASHING PREDICTIONS
aarima_LC_start = time.time()
aarima_LC_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
for branch in LC_train['Unique_BranchID'].unique():
    branch_train = LC_train.loc[LC_train.Unique_BranchID==branch][['Cashings']].set_index(LC_train.loc[LC_train.Unique_BranchID==branch]['Date'])

    try:
        mod = auto_arima(branch_train,seasonal=True,error_action='ignore',m=12,suppress_warnings=True)
        results = mod.fit(branch_train)
        pred = results.predict(n_periods=9)
        temp = {'PredMonth':['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019'],'Prediction':list(pred)}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch
        aarima_LC_predictions = aarima_LC_predictions.append(temp)
    except:
        pass
aarima_LC_end = time.time()

# %% DIRECT LOAN APPLICATION AARIMA PREDICTIONS
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

# %% PROPHET LIVE CHECK CASHING PREDICTIONS
LC_start = time.time()
LC_predictions = pd.DataFrame(columns=['Unique_BranchID','PredMonth','Prediction'])
ds = ['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019']
future = pd.DataFrame(ds,columns=['ds'])

for branch in LC_train['Unique_BranchID'].unique():
    branch_train = LC_train.loc[LC_train.Unique_BranchID==branch][['Adjusted_Issue_Date','Cashings']].rename(columns={'Adjusted_Issue_Date':'ds','Cashings':'y'})

    try:
        prophet = Prophet(seasonality_mode='multiplicative',daily_seasonality=False,weekly_seasonality=False)
        prophet.fit(branch_train)
        #future = prophet.make_future_dataframe(periods=6,freq='M')
        pred = prophet.predict(future)
        pred = pred.loc[(pred.ds<='09-01-2019')&(pred.ds>='01-01-2019')][['ds','yhat']]

        temp = {'PredMonth':pred.ds,'Prediction':pred.yhat}
        temp = pd.DataFrame(temp)
        temp['Unique_BranchID'] = branch

        LC_predictions = LC_predictions.append(temp)

    except:
        pass
LC_end = time.time()
