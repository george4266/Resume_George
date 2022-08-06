# %% Imports and file load
import pathlib
import datetime as dt
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# %% DATA LOAD

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

cash_actuals = mktg.loc[mktg['CashDate'] >= dt.datetime(2015,1,1)]
cash_actuals = cash_actuals.groupby(['Unique_BranchID',cash_actuals.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
cash_actuals.index = cash_actuals.index.set_levels([cash_actuals.index.levels[0], cash_actuals.index.levels[1].to_timestamp()])
cash_actuals = cash_actuals.reset_index()

newandformer = app.loc[((app['Application_Source']=='New Customer')|(app['Application_Source']=='Former Customer'))]
newandformer = newandformer.groupby(['Unique_BranchID', newandformer.AppCreatedDate.dt.to_period('M')])['Booked_Indicator'].sum().to_frame()
newandformer.index = newandformer.index.set_levels([newandformer.index.levels[0], newandformer.index.levels[1].to_timestamp()])
newandformer = newandformer.reset_index()


cashed
# %% add current months open function

def cashed_months_open(cashed):
    open_count = 1
    for b in cashed['Unique_BranchID'].unique():
        for i in cashed.loc[cashed['Unique_BranchID'] == b].index.unique():
            cashed.loc[(cashed['Unique_BranchID'] == b) & (cashed.index == i), 'MonthsOpen'] = open_count
            open_count = open_count + 1
        open_count = 1
    return cashed

for b in newandformer['Unique_BranchID'].unique():
    for i in newandformer.loc[newandformer['Unique_BranchID'] == b].index.unique():
        newandformer.loc[(newandformer['Unique_BranchID'] == b) & (newandformer.index == i), 'MonthsOpen'] = open_count
        open_count = open_count + 1
    open_count = 1

cashed
new_cashed = cashed.loc[cashed['Unique_BranchID'] >= 164]
new_cashed['CashDate'] = new_cashed['CashDate'].dt.normalize()
new_cashed

plt.figure(figsize=(20,15))
sns.boxplot(x='MonthsOpen', y='Cashings', data=new_cashed)
fig.show()


plt.figure(figsize=(30,15))
sns.boxplot(x='CashDate', y='Cashings', data=new_cashed)
fig.show()




# %% lets try this by state GRAPHS***************


fig, axs = plt.subplots(3,2,figsize=(20,15),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in mktg.State.unique():

    cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1)) & (mktg['State'] == state)]
    cashed = cashed.groupby(['Unique_BranchID',cashed.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
    cashed.index = cashed.index.set_levels([cashed.index.levels[0], cashed.index.levels[1].to_timestamp()])
    cashed = cashed.reset_index()

    cashed = cashed_months_open(cashed)
    new_cashed = cashed.loc[cashed['Unique_BranchID'] >= 164]
    try:
        ax = sns.boxplot(x='MonthsOpen', y='Cashings',data=new_cashed, ax=axs[num])
        ax.set_title('State - {}'.format(state))
        ax.set_xlabel('Current Months Open')
        ax.set_ylabel('Cashings')
        num+=1
    except:
        pass
fig.show()
plt.savefig(outputfolder / 'newbranch_cashings_state_MonthsOpen.png')


fig, axs = plt.subplots(3,2,figsize=(50,35),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in mktg.State.unique():

    cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1)) & (mktg['State'] == state)]
    cashed = cashed.groupby(['Unique_BranchID',cashed.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
    cashed.index = cashed.index.set_levels([cashed.index.levels[0], cashed.index.levels[1].to_timestamp()])
    cashed = cashed.reset_index()

    new_cashed = cashed.loc[cashed['Unique_BranchID'] >= 164]
    try:
        ax = sns.boxplot(x='CashDate', y='Cashings',data=new_cashed, ax=axs[num])
        ax.set_title('State - {}'.format(state))
        ax.set_xlabel('Date')
        ax.set_ylabel('Cashings')
        num+=1
    except:
        pass
fig.show()
plt.savefig(outputfolder / 'newbranch_cashings_state_date.png')


cashed
fig, axs = plt.subplots(3,2,figsize=(20,15),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in mktg.State.unique():

    cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1)) & (mktg['State'] == state)]
    cashed.CashDate = pd.to_datetime(cashed.CashDate)
    cashed['month'] = pd.DatetimeIndex(cashed.CashDate).month
    cashed = cashed.groupby(['Unique_BranchID','month'])['Cashings'].sum().to_frame()
    cashed.index = cashed.index.set_levels([cashed.index.levels[0], cashed.index.levels[1]])
    cashed = cashed.reset_index()
    new_cashed = cashed.loc[cashed['Unique_BranchID'] >= 164]

    try:
        ax = sns.boxplot(x='month', y='Cashings',data=new_cashed, ax=axs[num])
        ax.set_title('State - {}'.format(state))
        ax.set_xlabel('Month')
        ax.set_ylabel('Cashings')
        num+=1
    except:
        pass
fig.show()
plt.savefig(outputfolder / 'newbranch_cashings_state_month.png')


# %% calculate historic averages for new branch statespace
new_state_list = mktg.loc[mktg['Unique_BranchID'] >= 164]['State'].unique()
fig, axs = plt.subplots(3,2,figsize=(20,15),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in new_state_list:
    cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1)) & (mktg['State'] == state)]
    cashed.CashDate = pd.to_datetime(cashed.CashDate)
    cashed['month'] = pd.DatetimeIndex(cashed.CashDate).month
    cashed = cashed.groupby(['Unique_BranchID','month'])['Cashings'].sum().to_frame()
    cashed.index = cashed.index.set_levels([cashed.index.levels[0], cashed.index.levels[1]])
    cashed = cashed.reset_index()

    ax = sns.boxplot(x='month', y='Cashings',data=cashed, ax=axs[num])
    ax.set_title('State - {}'.format(state))
    ax.set_xlabel('Month')
    ax.set_ylabel('Cashings')
    num+=1
fig.show()


fig, axs = plt.subplots(3,2,figsize=(35,15),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in new_state_list:

    cashed = mktg.loc[(mktg['CashDate'] < dt.datetime(2019,1,1)) & (mktg['CashDate'] >= dt.datetime(2015,1,1)) & (mktg['State'] == state)]
    cashed = cashed.groupby(['Unique_BranchID',cashed.CashDate.dt.to_period('M')])['Cashings'].sum().to_frame()
    cashed.index = cashed.index.set_levels([cashed.index.levels[0], cashed.index.levels[1].to_timestamp()])
    cashed = cashed.reset_index()

    cashed = cashed_months_open(cashed)

    ax = sns.boxplot(x='MonthsOpen', y='Cashings',data=cashed, ax=axs[num])
    ax.set_title('State - {}'.format(state))
    ax.set_xlabel('Current Months Open')
    ax.set_ylabel('Cashings')
    num+=1

fig.show()
