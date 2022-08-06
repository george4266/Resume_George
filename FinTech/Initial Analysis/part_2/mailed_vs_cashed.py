import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go

datafolder = pathlib.Path.cwd().parent
outputfolder = 'C:\\Users\\999Na\\Documents\\Senior design\\SeniorDesign2020\\Initial Analysis\\Nate\\Outputs\\mailed_cashed'
marketingfile = datafolder / 'Data' / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(marketingfile, sep=',')

marketing

checks = marketing[['IssueDate', 'Cashings', 'Mailings']]
checks2 = marketing[['IssueDate', 'Cashings', 'Mailings']]
checks = checks.dropna(subset=['IssueDate'])
checks2 = checks2.dropna(subset=['IssueDate'])
checks = checks.fillna(value=0)
checks2 = checks2.fillna(value=0)
checks['IssueDate'] = pd.to_datetime(checks['IssueDate'])
checks2['IssueDate'] = pd.to_datetime(checks2['IssueDate'])
checks = checks.loc[checks['IssueDate']<dt.datetime(2018,12,30)]
checks2 = checks2.loc[checks2['IssueDate']<dt.datetime(2018,12,30)]
checks['IssueYear'] = pd.DatetimeIndex(checks.IssueDate).year
checks2['IssueYear'] = pd.DatetimeIndex(checks2.IssueDate).year
mailed = checks[['IssueDate', 'Mailings']].groupby(checks.IssueYear)['Mailings'].sum().to_frame().reset_index()
cashed = checks2[['IssueDate', 'Cashings']].groupby(checks2.IssueYear)['Cashings'].sum().to_frame().reset_index()
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
relationship

#fig = px.bar(mailed, x='IssueYear', y='Mailings', title='Total Mailed Live Checks by Year',labels={'IssueYear':'Year','Mailings':'Mailed Live Checks'},height=576,width=512)
#py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'mailed_live_checks_by_year.html').resolve().as_posix(),auto_open=False)
#fig = px.bar(cashed, x='IssueYear', y='Cashings', title='Total Cashed Live Checks by Year',labels={'IssueYear':'Year','Cashings':'Cashed Live Checks'},height=576,width=512)
#py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'cashed_live_checks_by_year.html').resolve().as_posix(),auto_open=False)
#fig = px.line(relationship, x='time', y='ratio', title='Ratio of Mailed to Cashed Live Checks',labels={'IssueYear':'Year','Cashings':'Cashed Live Checks'},height=576,width=512)
#py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'cashed_live_checks_ratio_by_year.html').resolve().as_posix(),auto_open=False)

cashed.to_csv(outputfolder+'\\cashed_by_year.csv')
mailed.to_csv(outputfolder+'\\mailed_by_year.csv')
relationship.to_csv(outputfolder+'\\relationship.csv')
