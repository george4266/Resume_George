import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
import json
import plotly

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Nate' / 'Outputs'
orginfile = datafolder / 'VT_Originations_11012019.txt'

orgin = pd.read_csv(orginfile, sep=',', low_memory=False)
orgin = orgin[['BookDate', 'Rescored_Tier_2017Model', 'ProductType', 'CashToCustomer']]
orgin['Rescored_Tier_2017Model'] = orgin['Rescored_Tier_2017Model'].dropna()
orgin['BookDate'] = pd.to_datetime(orgin['BookDate'])
orgin = orgin.loc[orgin['BookDate']>dt.datetime(2016,12,31)]
orgin['Rescored_Tier_2017Model'] = orgin['Rescored_Tier_2017Model'].fillna(value=0)
direct = orgin[(orgin['ProductType'] != 'LC') & (orgin['ProductType'] != 'Sales') & (orgin['ProductType'] != 'RE')]
lc = orgin[orgin['ProductType'] == 'LC']

direct_average = direct.groupby([orgin.BookDate.dt.to_period('M'), orgin.Rescored_Tier_2017Model])['CashToCustomer'].mean().to_frame().reset_index().pivot_table(index='BookDate',columns='Rescored_Tier_2017Model', values='CashToCustomer',margins=True,margins_name='Overall').drop(index=['Overall'])
direct_average = direct_average.unstack().reset_index()
direct_average = direct_average.rename(columns={0:'Average'})
direct_average

lc_average = lc.groupby([orgin.BookDate.dt.to_period('M'), orgin.Rescored_Tier_2017Model])['CashToCustomer'].mean().to_frame().reset_index().pivot_table(index='BookDate',columns='Rescored_Tier_2017Model', values='CashToCustomer',margins=True,margins_name='Overall').drop(index=['Overall'])
lc_average = lc_average.unstack().reset_index()
lc_average = lc_average.rename(columns={0:'Average'})
lc_average

fig = px.line(lc_average,x='BookDate',y='Average',title='Average LC Cash To Risk Tier Over Time',color='Rescored_Tier_2017Model',labels={'BookDate':'Time','Average':'Average Cash To RiskTier ($)'})
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'average_LC_cash_to_customer.html').resolve().as_posix())
fig = px.line(direct_average,x='BookDate',y='Average',title='Average Direct Loan Cash To Risk Tier Over Time',color='Rescored_Tier_2017Model',labels={'BookDate':'Time','Average':'Average Cash To RiskTier ($)'})
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'average_direct_loan_cash_to_customer.html').resolve().as_posix())
