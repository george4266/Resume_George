# %% Import Data
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'output'

branchfile = datafolder / 'VT_Branches_01072020.csv'
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

LC_file = outputfolder / 'LC_branch_sarimax_predictions.csv'
DL_file = outputfolder / 'DL_branch_sarimax_predictions.csv'

data = pd.read_csv(sep_2018, sep=',', low_memory=False).append(pd.read_csv(oct_2018, sep=',', low_memory=False)).append(pd.read_csv(nov_2018, sep=',', low_memory=False)).append(pd.read_csv(dec_2018, sep=',', low_memory=False\
    )).append(pd.read_csv(jan_2019, sep=',', low_memory=False)).append(pd.read_csv(feb_2019, sep=',', low_memory=False)).append(pd.read_csv(mar_2019, sep=',', low_memory=False)).append(pd.read_csv(apr_2019, sep=',', low_memory=False\
    )).append(pd.read_csv(may_2019, sep=',', low_memory=False)).append(pd.read_csv(jun_2019, sep=',', low_memory=False)).append(pd.read_csv(jul_2019, sep=',', low_memory=False)).append(pd.read_csv(aug_2019, sep=',', low_memory=False))

data.loc[data['StRank'] == 100, 'State'] = 'MS'
data.loc[data['StRank'] == 76, 'State'] = 'LA'
data.loc[data['StRank'] == 56, 'State'] = 'SC'
data.loc[data['StRank'] == 36, 'State'] = 'TN'
data.loc[data['StRank'] == 50, 'State'] = 'AL'
data.loc[data['StRank'] == 24, 'State'] = 'GA'
data.loc[data['StRank'] == 8, 'State'] = 'TX'
data.loc[data['StRank'] == 18, 'State'] = 'KY'

staterank = data[['StRank','State']].drop_duplicates()

branches = pd.read_csv(branchfile)[['Unique_BranchID','State']].drop_duplicates()
branches.head()


LC_predictions = pd.read_csv(LC_file,index_col=0)
DL_predictions = pd.read_csv(DL_file,index_col=0)

forecast_month = '2019-01-01'
LC_predictions = LC_predictions.loc[LC_predictions.PredMonth == forecast_month]
LC_predictions = LC_predictions.merge(branches, on='Unique_BranchID',how='left')
LC_predictions = LC_predictions[['Unique_BranchID','State','Prediction']]
LC_predictions.loc[LC_predictions['Prediction'] < 10, 'Prediction'] = 10

DL_predictions = DL_predictions.loc[DL_predictions.PredMonth == forecast_month]
DL_predictions = DL_predictions.merge(branches, on='Unique_BranchID',how='left')
DL_predictions = DL_predictions[['Unique_BranchID','State','Prediction']]
DL_predictions.loc[DL_predictions['Prediction'] < 1, 'Prediction'] = 1

statedlratio = data.groupby(['State','ProductType'])['Unique_ContractID'].count().reset_index()
statedlratio['AutoRatio'] = 0
statedlratio['PPRatio'] = 0
for state in statedlratio.State.unique():
    sum = statedlratio.loc[(statedlratio.State == state)&(statedlratio.ProductType != 'LC'),'Unique_ContractID'].sum()
    auto = statedlratio.loc[(statedlratio.State == state)&(statedlratio.ProductType == 'Auto'),'Unique_ContractID'].sum()
    pp = statedlratio.loc[(statedlratio.State == state)&(statedlratio.ProductType == 'PP'),'Unique_ContractID'].sum()
    statedlratio.loc[statedlratio.State == state, 'AutoRatio'] = auto/sum
    statedlratio.loc[statedlratio.State == state, 'PPRatio'] = pp/sum
statedlratio = statedlratio[['State','AutoRatio','PPRatio']].drop_duplicates()

DL_predictions = DL_predictions.merge(statedlratio, how='left', on='State')

auto_predictions = DL_predictions
auto_predictions['Prediction'] = auto_predictions['Prediction'] * auto_predictions['AutoRatio']
auto_predictions = auto_predictions[['Unique_BranchID','State','Prediction']]
pp_predictions = DL_predictions
pp_predictions['Prediction'] = pp_predictions['Prediction'] * pp_predictions['PPRatio']
pp_predictions = pp_predictions[['Unique_BranchID','State','Prediction']]

pp_predictions['Prediction'] = pp_predictions['Prediction'].apply(np.ceil).astype(int)
auto_predictions['Prediction'] = auto_predictions['Prediction'].apply(np.ceil).astype(int)
LC_predictions['Prediction'] = LC_predictions['Prediction'].apply(np.ceil).astype(int)

# %% random choice

count = 0
id = 1
x=1
productlist = ['Auto','LC','PP']

for product in productlist:
    if product == 'LC':
        pred_frame = LC_predictions
    elif product == 'PP':
        pred_frame = pp_predictions
    else:
        pred_frame = auto_predictions

    for branch in pred_frame['Unique_BranchID'].unique():
        id_list = list(range(x,x+pred_frame.loc[pred_frame['Unique_BranchID'] == branch]['Prediction'].values[0]))
        state = data.loc[data['Unique_BranchID'] == branch, 'State'].values[0]
        slice = data.loc[(data['State'] == state) & (data['ProductType'] == product)]
        tier = slice.Tier_MultipleModels.value_counts(normalize=True)
        tier_list = np.random.choice(tier.index, size=pred_frame.loc[pred_frame['Unique_BranchID'] == branch]['Prediction'].values[0], p=tier.values)

        if count == 0:
            dummy_accounts = {'Unique_ContractID':id_list, 'Tier_MultipleModels':tier_list}
            dummy_accounts = pd.DataFrame(dummy_accounts)
            dummy_accounts['Unique_BranchID'] = branch
            dummy_accounts['State'] = state
            dummy_accounts['ProductType'] = product
        else:
            dummies = {'Unique_ContractID':id_list, 'Tier_MultipleModels':tier_list}
            dummies = pd.DataFrame(dummies)
            dummies['Unique_BranchID'] = branch
            dummies['State'] = state
            dummies['ProductType'] = product
            dummy_accounts = dummy_accounts.append(dummies)
        count=count+1

dummy_accounts['Unique_ContractID'] = 'DUMMYMOB1'


# %%
varlist = ['Term','LC_ratio','contract_rank','avg_monthonbook_dl_r','credit_binned','avg_monthonbook_lc_r','AmountFinanced']
mobvarlist = ['SC_ratio','30+_Indicator','greater_than?']

dummy_accounts['MonthsOnBook'] = 1
mob = 1

for state in dummy_accounts['State'].unique():
    for product in dummy_accounts.loc[dummy_accounts.State == state, 'ProductType'].unique():
        for tier in dummy_accounts.loc[(dummy_accounts.State == state)&(dummy_accounts.ProductType == product), 'Tier_MultipleModels'].unique():
            id_list = dummy_accounts.loc[(dummy_accounts.State == state) & (dummy_accounts.Tier_MultipleModels == tier) & (dummy_accounts.ProductType == product), 'Unique_ContractID']
            dataslice = data.loc[(data['State'] == state) & (data['ProductType'] == product) & (data['ProductType'] == product)]
            mobdataslice = data.loc[(data['State'] == state) & (data['ProductType'] == product) & (data['ProductType'] == product) & (data['MonthsOnBook'] == mob)]

            for var in varlist:
                varfreq = dataslice[var].value_counts(normalize=True)
                vals = np.random.choice(varfreq.index, size=id_list.count(), p=varfreq.values)

                dummy_accounts.loc[(dummy_accounts.State == state) & (dummy_accounts.Tier_MultipleModels == tier) & (dummy_accounts.ProductType == product), var] = vals

            for var in mobvarlist:
                varfreq = mobdataslice[var].value_counts(normalize=True)
                vals = np.random.choice(varfreq.index, size=id_list.count(), p=varfreq.values)

                dummy_accounts.loc[(dummy_accounts.State == state) & (dummy_accounts.Tier_MultipleModels == tier) & (dummy_accounts.ProductType == product), var] = vals


# %%
cashlist = ['GrossBalance','NetCash','NetReceivable','CashToCustomer']
for product in dummy_accounts['ProductType'].unique():
    for amountfinanced in dummy_accounts.loc[dummy_accounts.ProductType == product, 'AmountFinanced'].unique():
        id_list = dummy_accounts.loc[(dummy_accounts.AmountFinanced == amountfinanced) & (dummy_accounts.ProductType == product), 'Unique_ContractID']
        cashdataslice = data.loc[(data.ProductType == product)&(data.AmountFinanced == amountfinanced)]
        for var in cashlist:
            varfreq = cashdataslice[var].value_counts(normalize=True)
            vals = np.random.choice(varfreq.index, size=id_list.count(), p=varfreq.values)

            dummy_accounts.loc[(dummy_accounts.ProductType == product) & (dummy_accounts.AmountFinanced == amountfinanced), var] = vals


dummy_accounts.head()
dummy_accounts['HighCredit'] = 0
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 1) & (dummy_accounts['ProductType'] != 'Auto'), 'HighCredit'] = 10000
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 2) & (dummy_accounts['ProductType'] != 'Auto'), 'HighCredit'] = 7000
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 3) & (dummy_accounts['ProductType'] != 'Auto'), 'HighCredit'] = 4500
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 4) & (dummy_accounts['ProductType'] != 'Auto'), 'HighCredit'] = 3000
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 1) & (dummy_accounts['ProductType'] == 'Auto'), 'HighCredit']= 12500
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 2) & (dummy_accounts['ProductType'] == 'Auto'), 'HighCredit'] = 7700
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 3) & (dummy_accounts['ProductType'] == 'Auto'), 'HighCredit'] = 4050
dummy_accounts.loc[(dummy_accounts['Tier_MultipleModels'] == 4) & (dummy_accounts['ProductType'] == 'Auto'), 'HighCredit'] = 2250

dummy_accounts['Utilization'] = 1
dummy_accounts.loc[dummy_accounts.GrossBalance < dummy_accounts.HighCredit, 'Utilization'] = dummy_accounts.loc[dummy_accounts.GrossBalance < dummy_accounts.HighCredit,'GrossBalance'] / dummy_accounts.loc[dummy_accounts.GrossBalance < dummy_accounts.HighCredit,'HighCredit']
dummy_accounts['Utilization'] = dummy_accounts['Utilization'].replace([-np.inf,np.inf],0)
dummy_accounts['PaydownPercent'] = 0
dummy_accounts.loc[dummy_accounts.NetReceivable < dummy_accounts.AmountFinanced, 'PaydownPercent'] = 1 - (dummy_accounts.loc[dummy_accounts.NetReceivable < dummy_accounts.AmountFinanced, 'NetReceivable'] / dummy_accounts.loc[dummy_accounts.NetReceivable < dummy_accounts.AmountFinanced, 'AmountFinanced')

dummy_accounts['months_til_avg_dl_r'] = dummy_accounts['avg_monthonbook_dl_r'] - dummy_accounts['MonthsOnBook']
dummy_accounts['months_til_avg_lc_r'] = dummy_accounts['avg_monthonbook_lc_r'] - dummy_accounts['MonthsOnBook']

dummy_accounts['available_offer'] = 0
dummy_accounts.loc[dummy_accounts['HighCredit'] > dummy_accounts['NetReceivable'], 'available_offer'] = dummy_accounts.loc[dummy_accounts.HighCredit > dummy_accounts.NetReceivable,'HighCredit'] - dummy_accounts.loc[dummy_accounts.HighCredit > dummy_accounts.NetReceivable,'NetReceivable']

dummy_accounts['Months_left'] = dummy_accounts['Term'] - dummy_accounts['MonthsOnBook']
dummy_accounts.merge(data[['State','StRank']].drop_duplicates(),how='left',on='State')

# %% output
dummy_accounts.to_csv('dummy_test_forecast_Jan2019.csv',index=False)
