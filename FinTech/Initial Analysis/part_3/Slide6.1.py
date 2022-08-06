import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import pathlib
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Eric' / 'Outputs'

VT_Marketing_11012019 = datafolder / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(VT_Marketing_11012019, sep=',')

m = marketing[['IssueDate', 'CashDate', 'Cashings', 'Mailings', 'Segment']]


m['IssueDate'] = pd.to_datetime(m['IssueDate'])
m['CashDate'] = pd.to_datetime(m['CashDate'])

m['Adjusted_Issue_Date'] = m['IssueDate'] + timedelta(days = 15)

m['Month'] = pd.to_datetime(m['Adjusted_Issue_Date']).dt.to_period('M')

issues = m[['Mailings','Segment', 'Month']]
cashes = m[['Cashings','Segment','Month']]

sum_issues = issues.groupby(['Month','Segment'])['Mailings'].sum()
cash_sums = cashes.groupby(['Month', 'Segment'])['Cashings'].sum()

response_rate = cash_sums/sum_issues

response_rate = response_rate.to_frame().reset_index()

plt.figure(figsize=(16,10))

fig = sns.barplot(x='Month', y=0, hue='Segment', data=response_rate)

fig.figure.savefig("Slide6.png")
