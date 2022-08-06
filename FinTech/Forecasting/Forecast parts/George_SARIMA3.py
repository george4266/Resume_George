# %% Imports and file load
import pathlib, datetime
from collections import Counter
import numpy as np, pandas as pd, seaborn as sns, matplotlib, matplotlib.pyplot as plt

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
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'





sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent.parent.parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'

#######
#import os
#entries = os.listdir(datafolder)

#######
Markfile = datafolder / 'VT_Marketing_11012019.txt'


mark = pd.read_csv(Markfile, sep=',', low_memory=False)
#perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))




mark['CashDate'] = pd.to_datetime(mark['CashDate'])
#mark
mark1 = mark[['CashDate', 'Cashings']]
mark1.dropna()


mark1['CashDate'] = pd.to_datetime(mark1['CashDate'])
cashed = mark1[['CashDate', 'Cashings']].groupby(mark1.CashDate.dt.to_period('M'))['Cashings'].sum().to_frame()

cashed.plot(figsize=(15,5), linewidth=4, fontsize=15)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Cashing units', fontsize=15)


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
cashed.interpolate(inplace = True)
cashed.index=cashed.index.to_timestamp()
decomposition = seasonal_decompose(cashed)
fig = decomposition.plot()
#cashed

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(cashed,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except:
            continue

######### these have the lowest AIC
#ARIMA(1, 1, 0)x(1, 1, 0, 12)12 - AIC:596.9709140381999
#ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:598.9462905294739
#ARIMA(1, 0, 0)x(1, 1, 0, 12)12 - AIC:614.1241684428674
#ARIMA(1, 0, 1)x(1, 1, 0, 12)12 - AIC:615.2533266051363
#ARIMA(0, 1, 0)x(1, 1, 0, 12)12 - AIC:618.1038228343879
#ARIMA(0, 1, 1)x(1, 1, 0, 12)12 - AIC:618.2081534845761
#########

mod = sm.tsa.statespace.SARIMAX(cashed,
                                order=(1, 0, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

######## plot diagnostics of the dataset

results.plot_diagnostics(figsize=(18, 8))
plt.show()

#cashed

###########   forecast of the last 10 months compared to the truth


pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = cashed['2015':].plot(label='observed',linewidth=4, fontsize=15)
pred.predicted_mean.plot(ax=ax, label='Forecasted', alpha=0.7, figsize=(15, 5),linewidth=4, fontsize=15)
pred.predicted_mean
ax.set_xlabel('Date',fontsize=20)
ax.set_ylabel('Cashed units',fontsize=20)
plt.legend()
plt.show()



######last 10 months predicted
cashed_forecasted = pred.predicted_mean
cashed_forecasted
forecast_list = cashed_forecasted.values.tolist()
forecast_list
#x=pd.DataFrame(cashed_forecasted)
#x
#cashed_forecasted=cashed_forecasted.reset_index()
#cashed_forecasted=cashed_forecasted.rename(columns={"index":"CashDate",cashed_forecasted.columns[1]:"Cashings"})
#cashed_forecasted



######last 10 months actuals
cashed_truth = cashed['2019-01-01':]
cashed_truth
truth_list = cashed_truth.values.tolist()
truth_list

#cashed_truth=cashed_truth.reset_index()
#cashed_truth=cashed_truth.rename(columns={"index":"CashDate",cashed_truth.columns[1]:"Cashings"})
#cashed_truth


#truth_list= cashed_truth.values.tolist()
#truth_list
#xy= [cashed_forecasted, cashed_truth]
#xyz=pd.concat(xy,ignore_index=True)
#xyz
#y=pd.DataFrame(cashed_truth)
#y

############# acccuracy parameters for (1, 0, 1),(1, 1, 0, 12)
mse = mean_squared_error(cashed_forecasted, cashed_truth)
mse
mae = metrics.mean_absolute_error(cashed_forecasted, cashed_truth)
mae
rmse = np.sqrt(metrics.mean_squared_error(cashed_forecasted, cashed_truth))
rmse

from itertools import imap
from operator import sub
list(imap(sub, truth_list, forecast_list))


#np.array(truth_list)

#z= (np.array(truth_list) - np.array(forecast_list))
#z


#mse = ((y- x) ** 2).mean()
#print('The Mean Squared Error is {}'.format(round(mse, 2)))
#print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))


############# forecast of the next Year (2020)

pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = cashed.plot(label='observed', figsize=(15, 5),linewidth=4, fontsize=15)
pred_uc.predicted_mean.plot(ax=ax, label='Forecast',linewidth=4, fontsize=15)
ax.set_xlabel('Date', fontsize=15)
ax.set_ylabel('Cashings', fontsize=15)
plt.legend()
plt.show()

#y_forecasted = pred.predicted_mean
#y_forecasted.head(12)


#pred_ci.head(24)


############ forecast of the next year (2020) values
new_forecast = pred_uc.predicted_mean
new_forecast.head(12)
