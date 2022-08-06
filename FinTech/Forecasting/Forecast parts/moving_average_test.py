# %% Imports and file load
import pathlib
import datetime as dt
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# %%  Create data frames

outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'output'
datafolder = pathlib.Path.cwd().parent / 'Data'

mktgfile = datafolder / 'VT_Marketing_11012019.txt'
mktg = pd.read_csv(mktgfile, sep=',', low_memory=False)
appfile = datafolder / 'VT_Applications_11262019.txt'
app = pd.read_csv(appfile, sep=',', low_memory=False)



mktg['CashDate'] = pd.to_datetime(mktg['CashDate'])
mktg = mktg[['CashDate', 'Cashings', 'Unique_BranchID', 'State']]
mktg = mktg.dropna()

app['AppCreatedDate'] = pd.to_datetime(app['AppCreatedDate'])
app = app[['AppCreatedDate', 'Booked_Indicator', 'Unique_BranchID','Unique_ApplicationID','Application_Source']]
app = app.dropna()

cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1))]
cashed = cashed.groupby(['Unique_BranchID',cashed.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cashed.index = cashed.index.set_levels([cashed.index.levels[0], cashed.index.levels[1].to_timestamp()])
cashed = cashed.reset_index()

# %% moving average test

cashed.index = cashed.CashDate
rolling_mean = cashed.loc[cashed['Unique_BranchID'] == 164].Cashings.rolling(window=3).mean().freecast

rolling_mean



cashed.loc[cashed['Unique_BranchID'] == 164]

# %% moving average predict forward

X = cashed.loc[cashed['Unique_BranchID'] == 164].Cashings.values
X
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)+9):
    if t >= len(test):
        length = len(history)
        yhat = np.mean([history[i] for i in range(length-window,length)])
        predictions.append(yhat)
        history.append(yhat)
    else:
    	length = len(history)
    	yhat = np.mean([history[i] for i in range(length-window,length)])
    	obs = test[t]
    	predictions.append(yhat)
    	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
#error = mean_squared_error(test, predictions)
#print('Test MSE: %.3f' % error)
# plot
plt.plot(X)
plt.plot(predictions, color='red')
plt.title('6 Month Prediction 6 Month Window Branch 164')
plt.show()
# zoom plot
'''plt.plot(test[0:100])
plt.plot(predictions[0:100], color='red')
plt.show()'''
