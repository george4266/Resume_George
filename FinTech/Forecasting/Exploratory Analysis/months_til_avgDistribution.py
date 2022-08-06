# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'
origination_file = datafolder  / 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'


orig = pd.read_csv(origination_file, sep=',', low_memory=False)


orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[orig.Tier_MultipleModels == 0.5, 'Tier_MultipleModels'] = 1
orig.dropna(subset=['Unique_CustomerID'], inplace=True)


perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))

orig = orig[['Unique_ContractID', 'Unique_CustomerID', 'ProductType', 'State', 'Tier_MultipleModels', 'BookDate']]
perf = perf[['Unique_ContractID', 'ProcessStatus', 'MonthsOnBook']]

combined = orig.merge(perf, on='Unique_ContractID', how='left')
combined['BookDate'] = pd.to_datetime(combined['BookDate'])

combined['MonthsOnBook'].dropna(inplace=True)
combined['months_added'] = pd.to_timedelta(combined['MonthsOnBook'], 'M')
combined['CurrentMonth'] = combined['BookDate'] + combined['months_added']
combined['CurrentMonth'] = combined['CurrentMonth'].dt.strftime('%m/%Y')
combined['CurrentMonth'] = pd.to_datetime(combined['CurrentMonth'])


# %%


comb_2018 = combined.loc[(combined['CurrentMonth'] >= dt.datetime(2018,1,1)) & (combined['ProductType'] != 'Sales')]
comb_2018['year'] = comb_2018.BookDate.dt.year
comb_2018['month'] = comb_2018.BookDate.dt.month
avg_mean = []
avg_time = []
tier_1_mean = []
tier_2_mean = []
tier_3_mean = []
tier_4_mean = []
tier_5_mean = []
la_mean = []
ms_mean = []
sc_mean = []
ga_mean = []
tn_mean = []
al_mean = []
ky_mean = []
tx_mean = []
pp_mean = []
auto_mean = []
lc_mean = []

for year in sorted(comb_2018.year.unique()):
    for month in sorted(comb_2018.month.unique()):
        if month == 12:
            avg_monthonbook = combined.loc[(combined['CurrentMonth'] < dt.datetime(year+1,1,1)) & (combined['ProcessStatus'] == 'Renewed')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().to_frame().reset_index()
        else:
            avg_monthonbook = combined.loc[(combined['CurrentMonth'] < dt.datetime(year,month+1,1)) & (combined['ProcessStatus'] == 'Renewed')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().to_frame().reset_index()
        avg_monthonbook.rename(columns={'MonthsOnBook':'Avg_MonthsOnBook'}, inplace=True)
        if month == 12:
            avgs = comb_2018.loc[(comb_2018['CurrentMonth'] >= dt.datetime(year,month,1)) & (comb_2018['CurrentMonth'] < dt.datetime(year+1,1,1))].merge(avg_monthonbook, on='Unique_CustomerID', how='left')
        else:
            avgs = comb_2018.loc[(comb_2018['CurrentMonth'] >= dt.datetime(year,month,1)) & (comb_2018['CurrentMonth'] < dt.datetime(year,month+1,1))].merge(avg_monthonbook, on='Unique_CustomerID', how='left')
        print(avgs.head())
        avg_mean.append(avgs['Avg_MonthsOnBook'].mean())
        avg_time.append(str(year)+'-'+str(month))
        tier_1_mean.append(avgs.loc[avgs['Tier_MultipleModels'] == 1]['Avg_MonthsOnBook'].mean())
        tier_2_mean.append(avgs.loc[avgs['Tier_MultipleModels'] == 2]['Avg_MonthsOnBook'].mean())
        tier_3_mean.append(avgs.loc[avgs['Tier_MultipleModels'] == 3]['Avg_MonthsOnBook'].mean())
        tier_4_mean.append(avgs.loc[avgs['Tier_MultipleModels'] == 4]['Avg_MonthsOnBook'].mean())
        tier_5_mean.append(avgs.loc[avgs['Tier_MultipleModels'] == 5]['Avg_MonthsOnBook'].mean())
        la_mean.append(avgs.loc[avgs['State'] == 'LA']['Avg_MonthsOnBook'].mean())
        ms_mean.append(avgs.loc[avgs['State'] == 'MS']['Avg_MonthsOnBook'].mean())
        sc_mean.append(avgs.loc[avgs['State'] == 'SC']['Avg_MonthsOnBook'].mean())
        tn_mean.append(avgs.loc[avgs['State'] == 'TN']['Avg_MonthsOnBook'].mean())
        al_mean.append(avgs.loc[avgs['State'] == 'AL']['Avg_MonthsOnBook'].mean())
        ga_mean.append(avgs.loc[avgs['State'] == 'GA']['Avg_MonthsOnBook'].mean())
        ky_mean.append(avgs.loc[avgs['State'] == 'KY']['Avg_MonthsOnBook'].mean())
        tx_mean.append(avgs.loc[avgs['State'] == 'TX']['Avg_MonthsOnBook'].mean())
        pp_mean.append(avgs.loc[avgs['ProductType'] == 'PP']['Avg_MonthsOnBook'].mean())
        auto_mean.append(avgs.loc[avgs['ProductType'] == 'Auto']['Avg_MonthsOnBook'].mean())
        lc_mean.append(avgs.loc[avgs['ProductType'] == 'LC']['Avg_MonthsOnBook'].mean())

# %% total rank over time
avg_info = {'avg_mean':avg_mean, 'avg_time':avg_time}
avg_info = pd.DataFrame(avg_info)
avg_info['avg_time'] = pd.to_datetime(avg_info['avg_time'])
avg_info = avg_info[avg_info['avg_time'] < dt.datetime(2019,10,1)]
plt.figure(figsize=(16,9))
sns.lineplot(x='avg_time', y='avg_mean',data=avg_info)
avg_info
#plt.figure(figsize=(16,9))
#plt.title('ContractRank Average and StandardDeviation After 2018')
#sns.lineplot(x='avg_time', y='value', hue='variable', data=pd.melt(rank_info, ['rank_time']))

# %% rank tier over time
avg_tier_mean = {'tier_1_mean':tier_1_mean, 'tier_2_mean':tier_2_mean, 'tier_3_mean':tier_3_mean, 'tier_4_mean':tier_4_mean, 'tier_5_mean':tier_5_mean, 'avg_time':avg_time}

avg_tier_mean = pd.DataFrame(avg_tier_mean)
avg_tier_mean['avg_time'] = pd.to_datetime(avg_tier_mean['avg_time'])

plt.figure(figsize=(16,9))
plt.title('Avg_MonthsOnBook by Tier Average After 2018')
sns.lineplot(x='avg_time', y='value', hue='variable', data=pd.melt(avg_tier_mean, ['avg_time']))
#plt.savefig(pathlib.Path(outputfolder / 'Contract_Rank_by_tier.png'))


# %% state contract rank

avg_state_mean = {'la_mean':la_mean, 'al_mean':al_mean, 'ms_mean':ms_mean, 'sc_mean':sc_mean, 'tn_mean':tn_mean, 'ga_mean':ga_mean, 'ky_mean':ky_mean, 'tx_mean':tx_mean, 'avg_time':avg_time}

avg_state_mean = pd.DataFrame(avg_state_mean)
avg_state_mean['avg_time'] = pd.to_datetime(avg_state_mean['avg_time'])

plt.figure(figsize=(16,9))
plt.title('ContractRank by State Average After 2018')
sns.lineplot(x='avg_time', y='value', hue='variable', data=pd.melt(avg_state_mean, ['avg_time']))
#plt.savefig(pathlib.Path(outputfolder / 'Contract_Rank_by_state.png'))


# %% product type

avg_prod_mean = {'pp_mean':pp_mean, 'auto_mean':auto_mean, 'lc_mean':lc_mean, 'avg_time':avg_time}

avg_prod_mean = pd.DataFrame(avg_prod_mean)
avg_prod_mean['avg_time'] = pd.to_datetime(avg_prod_mean['avg_time'])

plt.figure(figsize=(16,9))
plt.title('ContractRank by Product Type Average After 2018')
sns.lineplot(x='avg_time', y='value', hue='variable', data=pd.melt(avg_prod_mean, ['avg_time']))
plt.savefig(pathlib.Path(outputfolder / 'Contract_Rank_by_product.png'))
