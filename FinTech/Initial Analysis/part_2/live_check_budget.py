import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import plotly.express as px
import plotly.offline as py


datafolder = pathlib.Path.cwd().parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'Budget'
marketingfile = datafolder / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(marketingfile, sep=',')

#print(marketing.isna().sum())
checks = marketing[['CashDate', 'CheckAmount']]
checks = checks.dropna()
checks['CashDate'] = pd.to_datetime(checks['CashDate'])
checks['CashDate'] = checks['CashDate'].dt.normalize()
checks = checks.loc[checks['CashDate']<dt.datetime(2019,8,31)]
checks = checks.loc[checks['CashDate']>dt.datetime(2019,1,1)]
cashed = checks[['CashDate', 'CheckAmount']].groupby(checks.CashDate.dt.to_period('M'))['CheckAmount'].sum().to_frame()
budget = [19200000, 17500000, 19000000, 24000000, 24800000, 25900000, 25600000, 24700000]
time = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
cashed['budget'] = budget
cashed['time'] = time
#cashed.to_csv('Cashed_vs_budget.csv')
melted = cashed.melt('time', var_name='cols', value_name='vals')
cashed

fig = px.line(melted, x='time', y='vals', color='cols',  title='Budget vs Actual Cashed',height=576,width=512)
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'Budget_vs_actual_cashed_2019.html').resolve().as_posix(),auto_open=False)
#actual_vs_budget_plot.savefig('cashing_actual_vs_budget.png')
mape = np.mean(np.abs(cashed['CheckAmount'] - cashed['budget'])/cashed['CheckAmount'])*100
print(mape)

cashed
#print(cashed)
