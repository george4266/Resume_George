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
lc_rank = orig.loc[orig['ProductType'] == 'LC'].groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
lc_rank.rename(columns={'indicator':'lc_rank'}, inplace=True)
lc_rank = lc_rank[['lc_rank', 'Unique_CustomerID']]

orig_rank = orig.merge(contract_rank, on='Unique_CustomerID', how='left')
orig_rank = orig_rank.merge(lc_rank, on='Unique_CustomerID', how='left')
orig_rank['lc_rank'].fillna(value=0, inplace=True)
orig_rank.loc[(orig_rank.lc_rank == 0), 'lc_ratio'] = 0
orig_rank.loc[(orig_rank.lc_ratio != 0), 'lc_ratio'] = orig_rank['lc_rank']/orig_rank['contract_rank']
by_cust = orig_rank.groupby('Unique_CustomerID')['lc_ratio'].mean()
by_cust

sns.violinplot(by_cust)


# %% contract rank over time arrays
orig_2018 = orig.loc[(orig['BookDate'] >= dt.datetime(2018,1,1)) & (orig['ProductType'] != 'Sales')]
orig_2018['year'] = orig_2018.BookDate.dt.year
orig_2018['month'] = orig_2018.BookDate.dt.month
ratio_mean = []
ratio_std = []
ratio_time = []
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
        if month == 11 && year == 2019:
            break
        if month == 12:
            contract_rank = orig.loc[orig['BookDate'] < dt.datetime(year+1,1,1)].groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
            lc_rank = orig.loc[(orig['ProductType'] == 'LC') & (orig['BookDate'] < dt.datetime(year+1,1,1))].groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
        else:
            contract_rank = orig.loc[orig['BookDate'] < dt.datetime(year,month+1,1)].groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
            lc_rank = orig.loc[(orig['ProductType'] == 'LC') & (orig['BookDate'] < dt.datetime(year,month+1,1))].groupby(orig.Unique_CustomerID)['indicator'].sum().to_frame().reset_index()
        contract_rank.rename(columns={'indicator':'contract_rank'}, inplace=True)
        lc_rank.rename(columns={'indicator':'lc_rank'}, inplace=True)
        if month == 12:
            ranks = orig_2018.loc[(orig_2018['BookDate'] >= dt.datetime(year,month,1)) & (orig_2018['BookDate'] < dt.datetime(year+1,1,1))].merge(contract_rank, on='Unique_CustomerID', how='left').merge(lc_rank, on='Unique_CustomerID', how='left')
        else:
            ranks = orig_2018.loc[(orig_2018['BookDate'] >= dt.datetime(year,month,1)) & (orig_2018['BookDate'] < dt.datetime(year,month+1,1))].merge(contract_rank, on='Unique_CustomerID', how='left').merge(lc_rank, on='Unique_CustomerID', how='left')
        ranks['lc_rank'].fillna(value=0, inplace=True)
        print(ranks.head())
        ranks.loc[(ranks.lc_rank == 0), 'lc_ratio'] = 0
        ranks.loc[(ranks.lc_ratio != 0), 'lc_ratio'] = ranks['lc_rank']/ranks['contract_rank']
        ratio_mean.append(ranks['lc_ratio'].mean())
        ratio_std.append(ranks['lc_ratio'].std())
        ratio_time.append(str(year)+'-'+str(month))
        tier_1_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 1]['lc_ratio'].mean())
        tier_1_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 1]['lc_ratio'].std())
        tier_2_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 2]['lc_ratio'].mean())
        tier_2_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 2]['lc_ratio'].std())
        tier_3_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 3]['lc_ratio'].mean())
        tier_3_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 3]['lc_ratio'].std())
        tier_4_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 4]['lc_ratio'].mean())
        tier_4_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 4]['lc_ratio'].std())
        tier_5_mean.append(ranks.loc[ranks['Tier_MultipleModels'] == 5]['lc_ratio'].mean())
        tier_5_std.append(ranks.loc[ranks['Tier_MultipleModels'] == 5]['lc_ratio'].std())
        la_mean.append(ranks.loc[ranks['State'] == 'LA']['lc_ratio'].mean())
        la_std.append(ranks.loc[ranks['State'] == 'LA']['lc_ratio'].std())
        ms_mean.append(ranks.loc[ranks['State'] == 'MS']['lc_ratio'].mean())
        ms_std.append(ranks.loc[ranks['State'] == 'MS']['lc_ratio'].std())
        sc_mean.append(ranks.loc[ranks['State'] == 'SC']['lc_ratio'].mean())
        sc_std.append(ranks.loc[ranks['State'] == 'SC']['lc_ratio'].std())
        tn_mean.append(ranks.loc[ranks['State'] == 'TN']['lc_ratio'].mean())
        tn_std.append(ranks.loc[ranks['State'] == 'TN']['lc_ratio'].std())
        al_mean.append(ranks.loc[ranks['State'] == 'AL']['lc_ratio'].mean())
        al_std.append(ranks.loc[ranks['State'] == 'AL']['lc_ratio'].std())
        ga_mean.append(ranks.loc[ranks['State'] == 'GA']['lc_ratio'].mean())
        ga_std.append(ranks.loc[ranks['State'] == 'GA']['lc_ratio'].std())
        ky_mean.append(ranks.loc[ranks['State'] == 'KY']['lc_ratio'].mean())
        ky_std.append(ranks.loc[ranks['State'] == 'KY']['lc_ratio'].std())
        tx_mean.append(ranks.loc[ranks['State'] == 'TX']['lc_ratio'].mean())
        tx_std.append(ranks.loc[ranks['State'] == 'TX']['lc_ratio'].std())
        pp_mean.append(ranks.loc[ranks['ProductType'] == 'PP']['lc_ratio'].mean())
        pp_std.append(ranks.loc[ranks['ProductType'] == 'PP']['lc_ratio'].std())
        auto_mean.append(ranks.loc[ranks['ProductType'] == 'Auto']['lc_ratio'].mean())
        auto_std.append(ranks.loc[ranks['ProductType'] == 'Auto']['lc_ratio'].std())
        lc_mean.append(ranks.loc[ranks['ProductType'] == 'LC']['lc_ratio'].mean())
        lc_std.append(ranks.loc[ranks['ProductType'] == 'LC']['lc_ratio'].std())

# %% total rank over time
ratio_info = {'ratio_mean':ratio_mean, 'ratio_std':ratio_std, 'ratio_time':ratio_time}
ratio_info = pd.DataFrame(ratio_info)
ratio_info['ratio_time'] = pd.to_datetime(ratio_info['ratio_time'])

plt.figure(figsize=(16,9))
sns.lineplot(x='ratio_time', y='ratio_mean',data=ratio_info)
plt.figure(figsize=(16,9))
plt.title('LC_ratio Average and StandardDeviation After 2018')
sns.lineplot(x='ratio_time', y='value', hue='variable', data=pd.melt(ratio_info, ['ratio_time']))

# %% rank tier over time

ratio_tier_mean = {'tier_1_mean':tier_1_mean, 'tier_2_mean':tier_2_mean, 'tier_3_mean':tier_3_mean, 'tier_4_mean':tier_4_mean, 'tier_5_mean':tier_5_mean, 'ratio_time':ratio_time}
ratio_tier_std = {'tier_1_std':tier_1_std, 'tier_2_std':tier_2_std, 'tier_3_std':tier_3_std, 'tier_4_std':tier_4_std, 'tier_5_std':tier_5_std, 'ratio_time':ratio_time}

ratio_tier_mean = pd.DataFrame(ratio_tier_mean)
ratio_tier_mean['ratio_time'] = pd.to_datetime(ratio_tier_mean['ratio_time'])

ratio_tier_std = pd.DataFrame(ratio_tier_std)
ratio_tier_std['ratio_time'] = pd.to_datetime(ratio_tier_std['ratio_time'])

plt.figure(figsize=(16,9))
plt.title('LC Ratio by Tier Average After 2018')
sns.lineplot(x='ratio_time', y='value', hue='variable', data=pd.melt(ratio_tier_mean, ['ratio_time']))
plt.savefig(pathlib.Path(outputfolder / 'LC_ratio_by_tier.png'))

plt.figure(figsize=(16,9))
plt.title('ContractRank Average and StandardDeviation After 2018')
sns.lineplot(x='ratio_time', y='value', hue='variable', data=pd.melt(ratio_tier_std, ['ratio_time']))

# %% state contract rank

ratio_state_mean = {'la_mean':la_mean, 'al_mean':al_mean, 'ms_mean':ms_mean, 'sc_mean':sc_mean, 'tn_mean':tn_mean, 'ga_mean':ga_mean, 'ky_mean':ky_mean, 'tx_mean':tx_mean, 'ratio_time':ratio_time}
ratio_state_std = {'la_std':la_std, 'ms_std':ms_std, 'sc_std':sc_std, 'al_std':al_std, 'tn_std':tn_std, 'ga_std':ga_std, 'ky_std':ky_std, 'tx_std':tx_std, 'ratio_time':ratio_time}

ratio_state_mean = pd.DataFrame(ratio_state_mean)
ratio_state_mean['ratio_time'] = pd.to_datetime(ratio_state_mean['ratio_time'])

ratio_state_std = pd.DataFrame(ratio_state_std)
ratio_state_std['ratio_time'] = pd.to_datetime(ratio_state_std['ratio_time'])

plt.figure(figsize=(16,9))
plt.title('LC Ratio by State Average After 2018')
sns.lineplot(x='ratio_time', y='value', hue='variable', data=pd.melt(ratio_state_mean, ['ratio_time']))
plt.savefig(pathlib.Path(outputfolder / 'LC_ratio_by_state.png'))

plt.figure(figsize=(16,9))
plt.title('ContractRank Average and StandardDeviation After 2018')
sns.lineplot(x='ratio_time', y='value', hue='variable', data=pd.melt(ratio_state_std, ['ratio_time']))

# %% product type

ratio_prod_mean = {'pp_mean':pp_mean, 'auto_mean':auto_mean, 'lc_mean':lc_mean, 'ratio_time':ratio_time}
ratio_prod_std = {'pp_std':pp_std, 'auto_std':auto_std, 'lc_std':lc_std, 'ratio_time':ratio_time}

ratio_prod_mean = pd.DataFrame(ratio_prod_mean)
ratio_prod_mean['ratio_time'] = pd.to_datetime(ratio_prod_mean['ratio_time'])

ratio_prod_std = pd.DataFrame(ratio_prod_std)
ratio_prod_std['ratio_time'] = pd.to_datetime(ratio_prod_std['ratio_time'])

plt.figure(figsize=(16,9))
plt.title('LC ratio Average by Product Type After 2018')
sns.lineplot(x='ratio_time', y='value', hue='variable', data=pd.melt(ratio_prod_mean, ['ratio_time']))
plt.savefig(pathlib.Path(outputfolder / 'LC_ratio_by_product.png'))

plt.figure(figsize=(16,9))
plt.title('ContractRank Average and StandardDeviation After 2018')
sns.lineplot(x='ratio_time', y='value', hue='variable', data=pd.melt(ratio_prod_std, ['ratio_time']))
