# %% Import Data

import warnings
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns


datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Output'

sep_2018 = datafolder / 'feature_gen_dists_sep_2018.csv'
oct_2018 = datafolder / 'feature_gen_dists_oct_2018.csv'
nov_2018 = datafolder / 'feature_gen_dists_nov_2018.csv'
dec_2018 = datafolder / 'feature_gen_dists_dec_2018.csv'
jan_2019 = datafolder / 'feature_gen_dists_jan_2019.csv'
feb_2019 = datafolder / 'feature_gen_dists_feb_2019.csv'
mar_2019 = datafolder / 'feature_gen_dists_mar_2019.csv'
apr_2019 = datafolder / 'feature_gen_dists_apr_2019.csv'
may_2019 = datafolder / 'feature_gen_dists_may_2019.csv'
jun_2019 = datafolder / 'feature_gen_dists_jun_2019.csv'
jul_2019 = datafolder / 'feature_gen_dists_jul_2019.csv'
aug_2019 = datafolder / 'feature_gen_dists_aug_2019.csv'


data = pd.read_csv(sep_2018, sep=',', low_memory=False).append(pd.read_csv(oct_2018, sep=',', low_memory=False)).append(pd.read_csv(nov_2018, sep=',', low_memory=False)).append(pd.read_csv(dec_2018, sep=',', low_memory=False)).append(pd.read_csv(jan_2019, sep=',', low_memory=False)).append(pd.read_csv(feb_2019, sep=',', low_memory=False)).append(pd.read_csv(mar_2019, sep=',', low_memory=False)).append(pd.read_csv(apr_2019, sep=',', low_memory=False)).append(pd.read_csv(may_2019, sep=',', low_memory=False)).append(pd.read_csv(jun_2019, sep=',', low_memory=False)).append(pd.read_csv(jul_2019, sep=',', low_memory=False)).append(pd.read_csv(aug_2019, sep=',', low_memory=False))

data.loc[data['StRank'] == 100, 'State'] = 'MS'
data.loc[data['StRank'] == 76, 'State'] = 'LA'
data.loc[data['StRank'] == 56, 'State'] = 'SC'
data.loc[data['StRank'] == 36, 'State'] = 'TN'
data.loc[data['StRank'] == 50, 'State'] = 'AL'
data.loc[data['StRank'] == 24, 'State'] = 'GA'
data.loc[data['StRank'] == 8, 'State'] = 'TX'
data.loc[data['StRank'] == 18, 'State'] = 'KY'

fake_branch_id = [1,2,3,4,5,6,7,8,9]
fake_cashing_predictions = [1000, 750, 1250, 500, 1500, 800, 1200, 1000, 1000]

fake_predictions = {'Unique_BranchID':fake_branch_id, 'LC_forecast':fake_cashing_predictions}
fake_predictions = pd.DataFrame(fake_predictions)



for branch in fake_predictions['Unique_BranchID'].unique():
    get_state = data.loc[data['Unique_BranchID'] == branch].reset_index()
    fake_predictions.loc[fake_predictions['Unique_BranchID'] == branch, 'State'] = get_state.iloc[0]['State']

fake_predictions.head()


# %% random choice joint

fake_predictions.loc[fake_predictions['Unique_BranchID'] == branch, 'LC_forecast'].values[0]
count = 0
id = 1
x=1
for branch in fake_predictions['Unique_BranchID'].unique():
    id_list = list(range(x,x+fake_predictions.loc[fake_predictions['Unique_BranchID'] == branch]['LC_forecast'].values[0]))
    state = data.loc[data['Unique_BranchID'] == branch, 'State'].values[0]
    stuff = data.loc[(data['State'] == state) & (data['ProductType'] == 'LC')]
    d =stuff.Tier_MultipleModels.value_counts(normalize=True)
    tier_list = np.random.choice(d.index, size=fake_cashing_predictions[count], p=d.values)
    #tier_list = []
    #branch_list = []
    #state_list = []
    #for num in range(1,fake_predictions.loc[fake_predictions['Unique_BranchID'] == branch, 'LC_forecast'].values[0]):
    #    stuff = data.loc[(data['State'] == state) & (data['ProductType'] == 'LC')]
    #    d =stuff.Tier_MultipleModels.value_counts(normalize=True)
    #    tier = np.random.choice(d.index, p=d.values)
    #    tier_list.append(tier)
    #    branch_list.append(branch)
    #    id_list.append(id)
    #    state_list.append(state)
    #    id=id+1
    #    print(id)
    if count == 0:
        dummy_accounts = {'Unique_ContractID':id_list, 'Tier_MultipleModels':tier_list}
        dummy_accounts = pd.DataFrame(dummy_accounts)
        dummy_accounts['Unique_BranchID'] = branch
    else:
        dummies = {'Unique_ContractID':id_list, 'Tier_MultipleModels':tier_list}
        dummies = pd.DataFrame(dummies)
        dummies['Unique_BranchID'] = branch
        dummy_accounts = dummy_accounts.append(dummies)
    count=count+1
    x = fake_predictions.loc[fake_predictions['Unique_BranchID'] == branch]['LC_forecast'].values[0]+x



dummy_accounts
dummy_tier = dummy_accounts.groupby('Tier_MultipleModels')['Unique_ContractID'].count().to_frame().reset_index()
sns.barplot(data=dummy_tier, x='Tier_MultipleModels', y='Unique_ContractID')
