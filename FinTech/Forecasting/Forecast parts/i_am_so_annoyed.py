# %% Imports and file load
import pathlib
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import metrics

import pyaf.ForecastEngine as autof
from fbprophet import Prophet

from pmdarima import auto_arima

# %% DATA LOAD

outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'output'
datafolder = pathlib.Path.cwd().parent / 'Data'

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

allcashed = cashed.groupby(['CashDate'])['Cashings'].sum().reset_index()
test = cash_actuals.groupby(['CashDate'])['Cashings'].sum().reset_index()

allcashed2 = allcashed
allcashed2.index = pd.DatetimeIndex(allcashed2.CashDate).to_period('M')
allcashed2 = allcashed2.drop(columns='CashDate')

allcashed
# %% pyaf testing

lEngine = autof.cForecastEngine()
lEngine.train(allcashed, 'CashDate' , 'Cashings', 12)

lEngine.getModelInfo()
pred = lEngine.forecast(allcashed, 12)

#test = test.loc[(test.CashDate<='09-01-2019')&(test.CashDate>='01-01-2019')]
pred = pred.loc[(pred.CashDate<='09-01-2019')&(pred.CashDate>='01-01-2019')][['CashDate','Cashings_Forecast']]
#np.mean(np.abs((test.Cashings - pred.Cashings_Forecast) / test.Cashings)) * 100

pred

pred.plot.line('CashDate', ['Cashings' , 'Cashings_Forecast',
                                             'Cashings_Forecast_Lower_Bound',
                                             'Cashings_Forecast_Upper_Bound'], grid = True, figsize=(12, 8))


# %% trying prophet
df = allcashed.rename(columns={'CashDate':'ds','Cashings':'y'})

prophet = Prophet(seasonality_mode='multiplicative')
prophet.fit(df)
#future = prophet.make_future_dataframe(periods=6,freq='M')
ds = ['01-01-2019','02-01-2019','03-01-2019','04-01-2019','05-01-2019','06-01-2019','07-01-2019','08-01-2019','09-01-2019']
future = pd.DataFrame(ds,columns=['ds'])
pred = prophet.predict(future)
pred

test = test.loc[(test.CashDate<='09-01-2019')&(test.CashDate>='01-01-2019')]
pred.loc[(pred.ds<='09-01-2019')&(pred.ds>='01-01-2019')][['ds','yhat']]
np.mean(np.abs((test.Cashings - pred.yhat) / test.Cashings)) * 100

pred
test
fig = prophet.plot(pred)

prophet.plot_components(pred)

# %% sarimax
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

aic = 10000

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(allcashed2,order=param,
                seasonal_order=param_seasonal,            enforce_stationarity=False,
                enforce_invertibility=False)
            results = mod.fit()

            if results.aic < aic:
                print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
                aic = results.aic
        except:
            continue

mod = SARIMAX(allcashed2,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
            )
results = mod.fit()
pred = results.predict(start=pd.to_datetime('2019-01-01'),end=pd.to_datetime('2019-09-01'), dynamic=False)

pred

mod = auto_arima(allcashed2,seasonal=True,m=12,trace=True,error_action='ignore', suppress_warnings=True)
results = mod.fit(allcashed2)
pred = results.predict(n_periods=9)

pred
