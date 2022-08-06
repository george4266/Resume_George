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
terms = orig_2018[['Unique_ContractID', 'Term', 'ProductType', 'State', 'Tier_MultipleModels']]


# %% Overall dist
counted = terms.groupby('Term')['Unique_ContractID'].count().to_frame().reset_index()
high_count = counted.loc[counted['Unique_ContractID'] >= 1000]

plt.figure(figsize=(16,9))
plt.title('Term Counts Over 1000 After 2018')
sns.barplot(x ='Term', y='Unique_ContractID', data=high_count)

# %% Term distribution by Product, state, Tier_MultipleModels

fig, axs = plt.subplots(3,3,figsize=(20,10),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for state in terms.State.unique():
    ax = sns.countplot(x='Term',data=terms.loc[(terms.State == state)], ax=axs[num])
    ax.set_title('State - {}'.format(state))
    ax.set_xlabel('Term')
    ax.set_ylabel('Number of Originations')
    num+=1
fig.show()
plt.savefig(pathlib.Path(outputfolder / 'term_by_state.png'))

fig, axs = plt.subplots(1,3,figsize=(20,6),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for prod in terms.ProductType.unique():
    ax = sns.countplot(x='Term',data=terms.loc[(terms.ProductType == prod)], ax=axs[num])
    ax.set_title('ProductType - {}'.format(prod))
    ax.set_xlabel('Term')
    ax.set_ylabel('Number of Originations')
    num+=1
fig.show()
plt.savefig(pathlib.Path(outputfolder / 'term_by_product.png'))


fig, axs = plt.subplots(2,3,figsize=(20,9),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for tier in terms.Tier_MultipleModels.unique():
    ax = sns.countplot(x='Term',data=terms.loc[(terms.Tier_MultipleModels == tier)], ax=axs[num])
    ax.set_title('RiskTier - {}'.format(tier))
    ax.set_xlabel('Term')
    ax.set_ylabel('Number of Originations')
    num+=1
fig.show()
plt.savefig(pathlib.Path(outputfolder / 'term_by_tier.png'))

# %% tables for distributions

term_state = terms.groupby(['State', 'Term'])['Unique_ContractID'].count().to_frame().reset_index()

term_tier = terms.groupby(['Tier_MultipleModels', 'Term'])['Unique_ContractID'].count().to_frame().reset_index()

term_prod = terms.groupby(['ProductType', 'Term'])['Unique_ContractID'].count().to_frame().reset_index()
