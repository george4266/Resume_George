import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

datafolder = pathlib.Path.cwd().parent.parent.parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'mailed_cashed'
marketingfile = datafolder / 'Data' / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(marketingfile, sep=',')

m = marketing
m['IssueDate'] = pd.to_datetime(m['IssueDate'])
#m = m.loc[m['IssueDate']>=dt.datetime(2016,12,30)]
m['IssueMonth'] = pd.DatetimeIndex(m.IssueDate).month
#m['cashdummy'] = m['CashDate']
#m['cashdummy'].fillna(value=0, inplace=True)
m['Cashings'].fillna(value=0, inplace=True)
m.dropna(subset=['CreditScore'], inplace=True)
m['cashed_or_not'] = 1
m.loc[m['cashdummy']==0,'cashed_or_not'] = 0
len(m[(m['Mailings'] >= 10) & (m['Cashings'] >= 1)])
len(m[m['Mailings']>=10])
len(m)

m[m['Mailings']>=10].groupby('cashed_or_not').mean()
new = m[m['IssueDate'] > dt.datetime(2018,12,31)]
m[(m['cashed_or_not'] == 1) & (m['Mailings'] >= 2) & (m['Cashings'] == 1)]


'''x = m['CreditScore'].str.split(pat = '-')
y = 0
new = []
for i in x:
    for z in i:
        z = int(z)
        y = y + z
    new.append(round(y/2))
    y=0
new = np.array(new)
m['CreditScore'] = new
m['cashed_or_not'] = 1
m.loc[m['cashdummy']==0,'cashed_or_not'] = 0
cash = m[m['cashed_or_not'] == 1]
len(m)
m = m[m['Mailings'] >= 10]
m.columns
m.groupby('CreditScore').mean()
m.groupby('RiskTier').mean()


m[m['cashed_or_not'] == 1].count()
len(m)
