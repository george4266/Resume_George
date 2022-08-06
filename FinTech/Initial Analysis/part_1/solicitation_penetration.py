import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import csv
import seaborn as sns
import datetime as dt


marketing = pd.read_csv("VT_Marketing(2).csv", sep='\t')

checks = marketing[['CashDate', 'CheckAmount']]
checks['CashDate'] = pd.to_datetime(checks['CashDate'])
checks['CashDate'] = checks['CashDate'].dt.normalize()
checks = checks.loc[checks['CashDate']<dt.datetime(2019,8,31)]
checks = checks.loc[checks['CashDate']>dt.datetime(2019,1,1)]
cashed = checks[['CashDate', 'CheckAmount']].groupby(checks.CashDate.dt.to_period('M'))['CheckAmount'].sum().to_frame()
budget = [19500000, 17000000, 20000000, 24000000, 25000000, 25500000, 25500000, 25000000]
time = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
cashed['budget'] = budget
cashed['time'] = time
cashed = cashed.melt('time', var_name='cols', value_name='vals')
actual_vs_budget_plot = sns.factorplot(x='time', y='vals', hue='cols', data=cashed)
actual_vs_budget_plot.savefig('cashing_actual_vs_budget.png')

#print(cashed)