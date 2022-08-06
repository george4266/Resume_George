import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import plotly.express as px
import plotly.offline as py
pd.options.mode.chained_assignment = None
import seaborn as sns

datafolder = pathlib.Path.cwd().parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'mailed_cashed'
marketingfile = datafolder / 'Data' / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(marketingfile, sep=',')


m = marketing[['IssueDate', 'CashDate', 'Segment', 'Cashings', 'Mailings']]
m['IssueDate'] = pd.to_datetime(m['IssueDate'])
m['CashDate'] = pd.to_datetime(m['CashDate'])
m['IssueDate'] = m['IssueDate'] + dt.timedelta(10)
m.dropna(subset=['Segment'], inplace=True)

Issued = m.loc[(m['IssueDate'] >= dt.datetime(2018,11,1)) & (m['IssueDate'] <= dt.datetime(2019,7,31))]
Cashed = Issued.loc[(Issued['CashDate'] >= dt.datetime(2018,11,1)) & (Issued['CashDate'] <= dt.datetime(2019,7,31))]
Issued = Issued.groupby([Issued.IssueDate.dt.to_period('M'),'Segment'])['Mailings'].sum().to_frame().reset_index()
Cashed = Cashed.groupby([Cashed.CashDate.dt.to_period('M'),'Segment'])['Cashings'].sum().to_frame().reset_index()
Cashed
rr = Cashed['Cashings']/Issued['Mailings']
response_rate = Issued
response_rate['Mailings'] = rr

fig = sns.barplot(x='IssueDate', y='Mailings', hue='Segment', data=Issued)
fig.show()
#py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'live_check_penetration_rate_by_year_by_month.html').resolve().as_posix(),auto_open=False)
