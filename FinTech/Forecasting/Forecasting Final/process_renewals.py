# %% Takes all accounts in a dataset and advances them to the next month
import pathlib
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import re

datafolder = pathlib.Path.cwd().parent / 'Data'

#############################################################

# RENAME ACCORDINGLY, REPLACE THIS INPUT WITH OUTPUT FROM ML MODELS

# inputfile = datafolder / 'simulation_set_begin_jan2019_month1.csv'
inputfile = datafolder / 'feature_generation_apr_2019.csv'
outputfile = datafolder / 'simulation_set_begin_jan2019_month2.csv'

##############################################################

feature_generation_files = ['feature_generation_sep_2018.csv',
                            'feature_generation_oct_2018.csv',
                            'feature_generation_nov_2018.csv',
                            'feature_generation_dec_2018.csv',
                            'feature_generation_jan_2019.csv',
                            'feature_generation_feb_2019.csv',
                            'feature_generation_mar_2019.csv',
                            'feature_generation_apr_2019.csv',
                            'feature_generation_may_2019.csv',
                            'feature_generation_jun_2019.csv',
                            'feature_generation_jul_2019.csv',
                            'feature_generation_aug_2019.csv',
                            'feature_generation_sep_2019.csv'
                            ]


data = pd.DataFrame()
data_dtypes = {'MonthsOnBook':'int16','ProductType':'category','30+_Indicator':'int8','Tier_MultipleModels':'int8','Contacted_Memos':'int16','Solicitation_Memos':'int16'}

for feat_file in feature_generation_files:
    data = data.append(pd.read_csv(pathlib.Path(datafolder / feat_file),dtype=data_dtypes,parse_dates=['CurrentMonth']))

# %% SETTING UP RENEWAL CHANGE MATRIX
renewal_matrix = pd.read_csv(pathlib.Path(datafolder / 'renewal_matrix.csv'),index_col=[0,1,2],header=[0,1])

product_list=['Auto','PP']
tier_list = ['1.0','2.0','3.0','4.0','5.0']

def assign_renewal_tier(row):
    # FUNCTION IMITATING A RANDOM SELECTION FROM A DISCRETE DISTRIBUTION, FOR NEW PRODUCT TYPE AND RISK TIER
    pivot_slice = renewal_matrix.loc[row.State,row.ProductType,row.Tier_MultipleModels,:].T
    pivot_slice.columns = pivot_slice.columns.droplevel([0,1])
    pivot_slice.columns = pivot_slice.columns.astype(str)

    cumulative_sum=0
    random_val = np.random.rand()

    for new_product in product_list:
        for new_tier in tier_list:
            cumulative_sum = cumulative_sum + pivot_slice.loc[(new_product,new_tier)][0]

            if (random_val < cumulative_sum) or ((new_product == product_list[-1]) and (new_tier == tier_list[-1])):
                row.ProductType = new_product
                row.Tier_MultipleModels = float(new_tier)
                return row

    return row
##################################################################

accounts = pd.read_csv(inputfile)

# Create renewal set and drop the old accounts from the account list
renewal_set = accounts.loc[accounts['Renewed?'] == 1]
accounts = accounts.loc[accounts['Renewed?'] != 1]

renewal_set = renewal_set.apply(assign_renewal_tier, axis=1)

# Code from dummy creation to fill in other variables
# %%
varlist = ['Term','LC_ratio','contract_rank','Solicitation_Memos','AmountFinanced']
mobvarlist = ['Solicitation_Memos','Contacted_Memos','30+_Indicator']

renewal_set['MonthsOnBook'] = 1
mob = 1

for state in renewal_set['State'].unique():
    for product in renewal_set.loc[renewal_set.State == state, 'ProductType'].unique():
        for tier in renewal_set.loc[(renewal_set.State == state)&(renewal_set.ProductType == product), 'Tier_MultipleModels'].unique():
            id_list = renewal_set.loc[(renewal_set.State == state) & (renewal_set.Tier_MultipleModels == tier) & (renewal_set.ProductType == product), 'Unique_ContractID']
            dataslice = data.loc[(data['State'] == state) & (data['ProductType'] == product) & (data['ProductType'] == product)]
            mobdataslice = data.loc[(data['State'] == state) & (data['ProductType'] == product) & (data['ProductType'] == product) & (data['MonthsOnBook'] == mob)]

            for var in varlist:
                varfreq = dataslice[var].value_counts(normalize=True)
                vals = np.random.choice(varfreq.index, size=id_list.count(), p=varfreq.values)

                renewal_set.loc[(renewal_set.State == state) & (renewal_set.Tier_MultipleModels == tier) & (renewal_set.ProductType == product), var] = vals

            for var in mobvarlist:
                varfreq = mobdataslice[var].value_counts(normalize=True)
                vals = np.random.choice(varfreq.index, size=id_list.count(), p=varfreq.values)

                renewal_set.loc[(renewal_set.State == state) & (renewal_set.Tier_MultipleModels == tier) & (renewal_set.ProductType == product), var] = vals

cashlist = ['GrossBalance','NetCash','NetReceivable','CashToCustomer']
for product in renewal_set['ProductType'].unique():
    for amountfinanced in renewal_set.loc[renewal_set.ProductType == product, 'AmountFinanced'].unique():
        id_list = renewal_set.loc[(renewal_set.AmountFinanced == amountfinanced) & (renewal_set.ProductType == product), 'Unique_ContractID']
        cashdataslice = data.loc[(data.ProductType == product)&(data.AmountFinanced == amountfinanced)]
        for var in cashlist:
            varfreq = cashdataslice[var].value_counts(normalize=True)
            vals = np.random.choice(varfreq.index, size=id_list.count(), p=varfreq.values)

            renewal_set.loc[(renewal_set.ProductType == product) & (renewal_set.AmountFinanced == amountfinanced), var] = vals

renewal_set['HighCredit'] = 0
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 1) & (renewal_set['ProductType'] != 'Auto'), 'HighCredit'] = 10000
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 2) & (renewal_set['ProductType'] != 'Auto'), 'HighCredit'] = 7000
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 3) & (renewal_set['ProductType'] != 'Auto'), 'HighCredit'] = 4500
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 4) & (renewal_set['ProductType'] != 'Auto'), 'HighCredit'] = 3000
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 1) & (renewal_set['ProductType'] == 'Auto'), 'HighCredit']= 12500
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 2) & (renewal_set['ProductType'] == 'Auto'), 'HighCredit'] = 7700
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 3) & (renewal_set['ProductType'] == 'Auto'), 'HighCredit'] = 4050
renewal_set.loc[(renewal_set['Tier_MultipleModels'] == 4) & (renewal_set['ProductType'] == 'Auto'), 'HighCredit'] = 2250

renewal_set = renewal_set.loc[renewal_set.NetReceivable >= 0]
renewal_set['Utilization'] = 1
renewal_set.loc[renewal_set.GrossBalance < renewal_set.HighCredit, 'Utilization'] = renewal_set.loc[renewal_set.GrossBalance < renewal_set.HighCredit,'GrossBalance'] / renewal_set.loc[renewal_set.GrossBalance < renewal_set.HighCredit,'HighCredit']
renewal_set['Utilization'] = renewal_set['Utilization'].replace([-np.inf,np.inf],0)
renewal_set['PaydownPercent'] = 0
renewal_set.loc[renewal_set.NetReceivable < renewal_set.AmountFinanced, 'PaydownPercent'] = 1 - (renewal_set.loc[renewal_set.NetReceivable < renewal_set.AmountFinanced, 'NetReceivable'] / renewal_set.loc[renewal_set.NetReceivable < renewal_set.AmountFinanced, 'AmountFinanced'])

renewal_set['months_til_avg_dl_r'] = renewal_set['avg_monthonbook_dl_r'] - renewal_set['MonthsOnBook']
renewal_set['months_til_avg_lc_r'] = renewal_set['avg_monthonbook_lc_r'] - renewal_set['MonthsOnBook']
renewal_set['months_til_avg_dl_c'] = renewal_set['avg_monthonbook_dl_c'] - renewal_set['MonthsOnBook']
renewal_set['months_til_avg_lc_c'] = renewal_set['avg_monthonbook_lc_c'] - renewal_set['MonthsOnBook']

renewal_set['available_offer'] = 0
renewal_set.loc[renewal_set['HighCredit'] > renewal_set['NetReceivable'], 'available_offer'] = renewal_set.loc[renewal_set.HighCredit > renewal_set.NetReceivable,'HighCredit'] - renewal_set.loc[renewal_set.HighCredit > renewal_set.NetReceivable,'NetReceivable']

renewal_set['Months_left'] = renewal_set['Term'] - renewal_set['MonthsOnBook']
renewal_set['contract_rank'] = renewal_set['contract_rank'] + 1

renewal_set['Unique_ContractID'] = 'DUMMYMOB1'
renewal_set['BookDate'] = renewal_set['CurrentMonth']
renewal_set['SC_ratio'] = 0
renewal_set.loc[(renewal_set['Contacted_Memos'] !=0) & (renewal_set['Solicitation_Memos'] != 0), 'SC_ratio'] = renewal_set['Contacted_Memos']/renewal_set['Solicitation_Memos']

# %% OUTPUT TO CSV
accounts = accounts.append(renewal_set)
accounts.to_csv(outputfile, index=False)
