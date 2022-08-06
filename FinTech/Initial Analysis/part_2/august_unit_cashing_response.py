import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import seaborn as sns
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

datafolder = pathlib.Path.cwd().parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'Budget'
marketingfile = datafolder / 'Data' / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(marketingfile, sep=',')

m = marketing[['IssueDate', 'CashDate', 'Cashings', 'Mailings', 'RiskTier', 'Segment']]
m['CashDate'].fillna(value=0)
m['cashed_or_not'] = 1
m.loc[m['CashDate']==0,'cashed_or_not'] = 0

m['IssueDate'] = pd.to_datetime(m['IssueDate'])
m['CashDate'] = pd.to_datetime(m['CashDate'])
months = [2,3,4,5,6,7,8,9]
for month in months:
    month_less = month - 1
    issues = m.loc[(m['IssueDate'] < dt.datetime(2019,month,16)) & (m['IssueDate'] > dt.datetime(2019,month_less,15))]
    issues.dropna(subset=['Segment'], inplace=True)
    cashes = issues.loc[(issues['CashDate'] < dt.datetime(2019,month,16)) & (issues['CashDate'] > dt.datetime(2019,month_less,15))]

    sum_issues = issues.groupby(['RiskTier','Segment'])['Mailings'].sum()
    cash_sums = cashes.groupby(['RiskTier', 'Segment'])['Cashings'].sum()

    response_rate = cash_sums/sum_issues

    response_rate = response_rate.to_frame().reset_index()
    print('Segment resonse rate month'+str(month))
    plt.figure(month)
    sns.barplot(x='RiskTier', y=0, hue='Segment', data=response_rate)
