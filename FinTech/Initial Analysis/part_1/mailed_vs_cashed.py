import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import csv
import seaborn as sns
import datetime as dt

marketing = pd.read_csv("VT_Marketing(2).csv", sep='\t')

#originations['BookDate'] = pd.to_datetime(originations['BookDate'])

#checks_mailed = pd.merge(originations,marketing,on='Unique_ContractID',how='left')

checks = marketing[['IssueDate', 'Cashings', 'Mailings']]
checks2 = marketing[['IssueDate', 'Cashings', 'Mailings']]
checks['IssueDate'] = pd.to_datetime(checks['IssueDate'])
checks2['IssueDate'] = pd.to_datetime(checks2['IssueDate'])
checks['IssueDate'] = checks['IssueDate'].dt.normalize()
checks2['IssueDate'] = checks2['IssueDate'].dt.normalize()
checks = checks.loc[checks['IssueDate']<dt.datetime(2018,12,30)]
checks2 = checks2.loc[checks2['IssueDate']<dt.datetime(2018,12,30)]
mailed = checks[['IssueDate', 'Mailings']].groupby(checks.IssueDate.dt.to_period('Y'))['Mailings'].sum().to_frame()
cashed = checks2[['IssueDate', 'Cashings']].groupby(checks2.IssueDate.dt.to_period('Y'))['Cashings'].sum().to_frame()
a =[]
b =[]
ratio = []
time = [2015, 2016, 2017, 2018]
for index, row in cashed.iterrows():
    a.append(row['Cashings'])
    
for index, row in mailed.iterrows():    
    b.append(row['Mailings'])
for num in range(0,4):
    ratio.append(a[num]/b[num])

print(ratio)
relationship = {'time':time, 'ratio': ratio}
relationship = pd.DataFrame(relationship)


#mailed_plot = sns.barplot(data=mailed, x=mailed.index, y='Mailings').get_figure()
#mailed_plot.savefig('mailed_by_year.png')
#cashed_plot = sns.barplot(data=cashed, x=cashed.index, y='Cashings').get_figure()
#cashed_plot.savefig('cashed_by_year.png')
#relationship_plot = sns.lineplot(data=relationship, x='time', y='ratio').get_figure()
#relationship_plot.savefig('cashed_to_mailed.png')







