# %% import libraries and files

import pathlib, swifter
import pandas as pd, numpy as np, datetime as dt, seaborn as sns, matplotlib.pyplot as plt

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'

# %% read files and prepare data

orig = pd.read_csv(origination_file, sep=',', low_memory=False)
orig['BookDate'] = pd.to_datetime(orig['BookDate'])

orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[orig.Tier_MultipleModels == 0.5, 'Tier_MultipleModels'] = 1

orig_2018 = orig.loc[(orig['BookDate'] >= dt.datetime(2018,1,1)) & (orig['ProductType'] != 'Sales')]
cash = orig_2018[['Unique_ContractID', 'CashToCustomer', 'ProductType', 'State', 'Tier_MultipleModels']]


# %% Overall dist

plt.figure(figsize=(16,9))
plt.title('CashToCustomer Distribution After 2018')
sns.violinplot(cash['CashToCustomer'])

# %% Term distribution by Product, state, Tier_MultipleModels

fig, axs = plt.subplots(3,3,figsize=(20,10),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in cash.State.unique():
    ax = sns.violinplot(cash.loc[(cash.State == state)]['CashToCustomer'], ax=axs[num])
    ax.set_title('State - {}'.format(state))
    ax.set_xlabel('CashToCustomer')
    ax.set_ylabel('Number of Originations')
    num+=1
fig.show()


fig, axs = plt.subplots(1,3,figsize=(20,6),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for prod in cash.ProductType.unique():
    ax = sns.violinplot(cash.loc[(cash.ProductType == prod)]['CashToCustomer'], ax=axs[num])
    ax.set_title('ProductType - {}'.format(prod))
    ax.set_xlabel('CashToCustomer')
    ax.set_ylabel('Number of Originations')
    num+=1
fig.show()


fig, axs = plt.subplots(2,3,figsize=(20,9),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for tier in cash.Tier_MultipleModels.unique():
    ax = sns.violinplot(cash.loc[(cash.Tier_MultipleModels == tier)]['CashToCustomer'], ax=axs[num])
    ax.set_title('RiskTier - {}'.format(tier))
    ax.set_xlabel('CashToCustomer')
    ax.set_ylabel('Number of Originations')
    num+=1
fig.show()


# %% tables for distributions

cash_state = cash.groupby(['State', 'CashToCustomer'])['Unique_ContractID'].count().to_frame().reset_index()

cash_tier = cash.groupby(['Tier_MultipleModels', 'CashToCustomer'])['Unique_ContractID'].count().to_frame().reset_index()

cash_prod = cash.groupby(['ProductType', 'CashToCustomer'])['Unique_ContractID'].count().to_frame().reset_index()
