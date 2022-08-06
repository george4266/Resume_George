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
orig['indicator'] = 1
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[orig.Tier_MultipleModels == 0.5, 'Tier_MultipleModels'] = 1
orig.dropna(subset=['Unique_CustomerID'], inplace=True)

contract_rank = orig.groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
contract_rank.rename(columns={'indicator':'contract_rank'}, inplace=True)
contract_rank = contract_rank[['contract_rank', 'Unique_CustomerID']]
orig_rank = orig.merge(contract_rank, on='Unique_CustomerID', how='left')



# %%
orig_2018 = orig_rank.loc[(orig_rank['BookDate'] >= dt.datetime(2018,1,1)) & (orig_rank['ProductType'] != 'Sales')]

plt.figure(figsize=(16,9))
plt.title('ContractRank Distribution After 2018')
sns.countplot(orig_2018['contract_rank'])


# %% contract rank over time arrays
orig_2018 = orig.loc[(orig['BookDate'] >= dt.datetime(2018,1,1)) & (orig['ProductType'] != 'Sales')]
orig_2018['year'] = orig_2018.BookDate.dt.year
orig_2018['month'] = orig_2018.BookDate.dt.month
rank_mean = []
rank_std = []
rank_time = []
tier_1_mean = []
tier_1_std = []
tier_2_std = []
tier_2_mean = []
tier_3_mean = []
tier_3_std = []
tier_4_mean = []
tier_4_std = []
tier_5_mean = []
tier_5_std = []
orig['State'].unique()
la_mean = []
la_std = []
ms_mean = []
ms_std = []
sc_mean = []
sc_std = []
ga_mean = []
ga_std = []
tn_mean = []
tn_std = []
al_mean = []
al_std = []
ky_mean = []
ky_std = []
tx_mean = []
tx_std = []
pp_mean = []
pp_std = []
auto_mean = []
auto_std = []
lc_mean = []
lc_std = []

for year in sorted(orig_2018.year.unique()):
    for month in sorted(orig_2018.month.unique()):
        if month == 12:
            contract_rank = orig.loc[orig['BookDate'] < dt.datetime(year+1,1,1)].groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
        else:
            contract_rank = orig.loc[orig['BookDate'] < dt.datetime(year,month+1,1)].groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
        contract_rank.rename(columns={'indicator':'contract_rank'}, inplace=True)
        if month == 12:
            ranks = orig_2018.loc[(orig_2018['BookDate'] >= dt.datetime(year,month,1)) & (orig_2018['BookDate'] < dt.datetime(year+1,1,1))].merge(contract_rank, on='Unique_CustomerID', how='left')
        else:
            ranks = orig_2018.loc[(orig_2018['BookDate'] >= dt.datetime(year,month,1)) & (orig_2018['BookDate'] < dt.datetime(year,month+1,1))].merge(contract_rank, on='Unique_CustomerID', how='left')
        rank_mean.append(ranks['contract_rank'].mean())
        rank_std.append(ranks['contract_rank'].std())
        rank_time.append(str(year)+'-'+str(month))
        tier_1_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 1]['contract_rank'].mean())
        tier_1_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 1]['contract_rank'].std())
        tier_2_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 2]['contract_rank'].mean())
        tier_2_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 2]['contract_rank'].std())
        tier_3_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 3]['contract_rank'].mean())
        tier_3_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 3]['contract_rank'].std())
        tier_4_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 4]['contract_rank'].mean())
        tier_4_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 4]['contract_rank'].std())
        tier_5_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 5]['contract_rank'].mean())
        tier_5_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 5]['contract_rank'].std())
        la_mean.append(ranks.loc[ranks['State'] == 'LA']['contract_rank'].mean())
        la_std.append(ranks.loc[ranks['State'] == 'LA']['contract_rank'].std())
        ms_mean.append(ranks.loc[ranks['State'] == 'MS']['contract_rank'].mean())
        ms_std.append(ranks.loc[ranks['State'] == 'MS']['contract_rank'].std())
        sc_mean.append(ranks.loc[ranks['State'] == 'SC']['contract_rank'].mean())
        sc_std.append(ranks.loc[ranks['State'] == 'SC']['contract_rank'].std())
        tn_mean.append(ranks.loc[ranks['State'] == 'TN']['contract_rank'].mean())
        tn_std.append(ranks.loc[ranks['State'] == 'TN']['contract_rank'].std())
        al_mean.append(ranks.loc[ranks['State'] == 'AL']['contract_rank'].mean())
        al_std.append(ranks.loc[ranks['State'] == 'AL']['contract_rank'].std())
        ga_mean.append(ranks.loc[ranks['State'] == 'GA']['contract_rank'].mean())
        ga_std.append(ranks.loc[ranks['State'] == 'GA']['contract_rank'].std())
        ky_mean.append(ranks.loc[ranks['State'] == 'KY']['contract_rank'].mean())
        ky_std.append(ranks.loc[ranks['State'] == 'KY']['contract_rank'].std())
        tx_mean.append(ranks.loc[ranks['State'] == 'TX']['contract_rank'].mean())
        tx_std.append(ranks.loc[ranks['State'] == 'TX']['contract_rank'].std())
        pp_mean.append(ranks.loc[ranks['ProductType'] == 'PP']['contract_rank'].mean())
        pp_std.append(ranks.loc[ranks['ProductType'] == 'PP']['contract_rank'].std())
        auto_mean.append(ranks.loc[ranks['ProductType'] == 'Auto']['contract_rank'].mean())
        auto_std.append(ranks.loc[ranks['ProductType'] == 'Auto']['contract_rank'].std())
        lc_mean.append(ranks.loc[ranks['ProductType'] == 'LC']['contract_rank'].mean())
        lc_std.append(ranks.loc[ranks['ProductType'] == 'LC']['contract_rank'].std())

# %% total rank over time
rank_info = {'rank_mean':rank_mean, 'rank_std':rank_std, 'rank_time':rank_time}
rank_info = pd.DataFrame(rank_info)
rank_info['rank_time'] = pd.to_datetime(rank_info['rank_time'])

plt.figure(figsize=(16,9))
sns.lineplot(x='rank_time', y='rank_mean',data=rank_info)
plt.figure(figsize=(16,9))
plt.title('ContractRank Average and StandardDeviation After 2018')
sns.lineplot(x='rank_time', y='value', hue='variable', data=pd.melt(rank_info, ['rank_time']))

# %% rank tier over time

rank_tier_mean = {'tier_1_mean':tier_1_mean, 'tier_2_mean':tier_2_mean, 'tier_3_mean':tier_3_mean, 'tier_4_mean':tier_4_mean, 'tier_5_mean':tier_5_mean, 'rank_time':rank_time}
rank_tier_std = {'tier_1_std':tier_1_std, 'tier_2_std':tier_2_std, 'tier_3_std':tier_3_std, 'tier_4_std':tier_4_std, 'tier_5_std':tier_5_std, 'rank_time':rank_time}

rank_tier_mean = pd.DataFrame(rank_tier_mean)
rank_tier_mean['rank_time'] = pd.to_datetime(rank_tier_mean['rank_time'])

rank_tier_std = pd.DataFrame(rank_tier_std)
rank_tier_std['rank_time'] = pd.to_datetime(rank_tier_std['rank_time'])

plt.figure(figsize=(16,9))
plt.title('ContractRank by Tier Average After 2018')
sns.lineplot(x='rank_time', y='value', hue='variable', data=pd.melt(rank_tier_mean, ['rank_time']))
plt.savefig(pathlib.Path(outputfolder / 'Contract_Rank_by_tier.png'))

plt.figure(figsize=(16,9))
plt.title('ContractRank Average and StandardDeviation After 2018')
sns.lineplot(x='rank_time', y='value', hue='variable', data=pd.melt(rank_tier_std, ['rank_time']))

# %% state contract rank

rank_state_mean = {'la_mean':la_mean, 'al_mean':al_mean, 'ms_mean':ms_mean, 'sc_mean':sc_mean, 'tn_mean':tn_mean, 'ga_mean':ga_mean, 'ky_mean':ky_mean, 'tx_mean':tx_mean, 'rank_time':rank_time}
rank_state_std = {'la_std':la_std, 'ms_std':ms_std, 'sc_std':sc_std, 'al_std':al_std, 'tn_std':tn_std, 'ga_std':ga_std, 'ky_std':ky_std, 'tx_std':tx_std, 'rank_time':rank_time}

rank_state_mean = pd.DataFrame(rank_state_mean)
rank_state_mean['rank_time'] = pd.to_datetime(rank_state_mean['rank_time'])

rank_state_std = pd.DataFrame(rank_state_std)
rank_state_std['rank_time'] = pd.to_datetime(rank_state_std['rank_time'])

plt.figure(figsize=(16,9))
plt.title('ContractRank by State Average After 2018')
sns.lineplot(x='rank_time', y='value', hue='variable', data=pd.melt(rank_state_mean, ['rank_time']))
plt.savefig(pathlib.Path(outputfolder / 'Contract_Rank_by_state.png'))

plt.figure(figsize=(16,9))
plt.title('ContractRank Average and StandardDeviation After 2018')
sns.lineplot(x='rank_time', y='value', hue='variable', data=pd.melt(rank_state_std, ['rank_time']))

# %% product type

rank_prod_mean = {'pp_mean':pp_mean, 'auto_mean':auto_mean, 'lc_mean':lc_mean, 'rank_time':rank_time}
rank_prod_std = {'pp_std':pp_std, 'auto_std':auto_std, 'lc_std':lc_std, 'rank_time':rank_time}

rank_prod_mean = pd.DataFrame(rank_prod_mean)
rank_prod_mean['rank_time'] = pd.to_datetime(rank_prod_mean['rank_time'])

rank_prod_std = pd.DataFrame(rank_prod_std)
rank_prod_std['rank_time'] = pd.to_datetime(rank_prod_std['rank_time'])

plt.figure(figsize=(16,9))
plt.title('ContractRank by Product Type Average After 2018')
sns.lineplot(x='rank_time', y='value', hue='variable', data=pd.melt(rank_prod_mean, ['rank_time']))
plt.savefig(pathlib.Path(outputfolder / 'Contract_Rank_by_product.png'))

plt.figure(figsize=(16,9))
plt.title('ContractRank Average and StandardDeviation After 2018')
sns.lineplot(x='rank_time', y='value', hue='variable', data=pd.melt(rank_prod_std, ['rank_time']))
