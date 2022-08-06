# feature_generation.py
# Generates a csv of feature and response variables for a given month

# %% LIBRARY AND DATA IMPORT
import pandas as pd, numpy as np, datetime as dt, seaborn as sns, matplotlib.pyplot as plt
import pathlib, os
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Output'

month = '09-01-2018'
#month = input("Input month for which to generate feature set (MM-DD-YYYY format):")
dtmonth = pd.to_datetime(month)

# %% FUNCTIONS
def add_customer_history_features(origin):
    # Input origination table to add live check ratio and contract rank features.
    # Returns origination table with these columns added.
    valid_accounts = origin.loc[origin.BookDate < (dtmonth + pd.DateOffset(months=1))][['BookDate','Unique_ContractID','Unique_CustomerID','ProductType']]
    LC_originations = valid_accounts.loc[valid_accounts.ProductType=='LC'].groupby('Unique_CustomerID')['Unique_ContractID'].count().to_frame().reset_index().rename(columns={'Unique_ContractID':'prev_LCs'})
    ALL_originations = valid_accounts.groupby('Unique_CustomerID')['Unique_ContractID'].count().to_frame().reset_index().rename(columns={'Unique_ContractID':'contract_rank'})
    ALL_originations = ALL_originations.merge(LC_originations,how='left',on='Unique_CustomerID')
    ALL_originations['prev_LCs'].fillna(0,inplace=True)
    ALL_originations['LC_ratio'] = ALL_originations['prev_LCs'] / ALL_originations['contract_rank']

    origin = origin.merge(ALL_originations[['Unique_CustomerID','LC_ratio','contract_rank']], how='left',on='Unique_CustomerID')

    return origin

def add_customer_average_features(origin_and_perf):
    # Input origination and performance table to return table with average months on book,
    # and average months on book until closing/renewal for direct loans/live check
    # feature fields added.
    valid_accounts = origin_and_perf.loc[origin_and_perf['CurrentMonth'] < (dtmonth + pd.DateOffset(months=1))]
    valid_accounts = valid_accounts[['Unique_CustomerID','Unique_ContractID','ProcessStatus', 'MonthsOnBook', 'ProductType','Contacted_Memos','Solicitation_Memos','CurrentMonth']]
    valid_accounts['Contacted_Memos'].fillna(value=0, inplace=True)

    # Months on book until renewal for live checks and direct loans
    averages = valid_accounts.loc[(valid_accounts['ProcessStatus'] == 'Renewed')&(valid_accounts['ProductType'] != 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index().rename(columns={'MonthsOnBook':'avg_monthonbook_dl_r'})
    averages = averages.merge(valid_accounts.loc[(valid_accounts['ProcessStatus'] == 'Renewed') & (valid_accounts['ProductType'] == 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index().rename(columns={'MonthsOnBook':'avg_monthonbook_lc_r'}),how='outer',on='Unique_CustomerID')

    # Months on book until closing for live checks and direct loans
    averages = averages.merge(valid_accounts.loc[(valid_accounts['ProcessStatus'] == 'Closed') & (valid_accounts['ProductType'] != 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index().rename(columns={'MonthsOnBook':'avg_monthonbook_dl_c'}),how='outer',on='Unique_CustomerID')
    averages = averages.merge(valid_accounts.loc[(valid_accounts['ProcessStatus'] == 'Closed') & (valid_accounts['ProductType'] != 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index().rename(columns={'MonthsOnBook':'avg_monthonbook_lc_c'}),how='outer',on='Unique_CustomerID')

    solicit_contact = valid_accounts.groupby('Unique_CustomerID')['Contacted_Memos'].sum().reset_index()
    solicit_contact = solicit_contact.merge(valid_accounts.groupby('Unique_CustomerID')['Solicitation_Memos'].sum().reset_index(),how='outer',on='Unique_CustomerID')
    solicit_contact['SC_ratio'] = 0
    solicit_contact.loc[(solicit_contact['Contacted_Memos'] !=0) & (solicit_contact['Solicitation_Memos'] != 0), 'SC_ratio'] = solicit_contact['Contacted_Memos']/solicit_contact['Solicitation_Memos']
    solicit_contact = solicit_contact[['Unique_CustomerID', 'SC_ratio']]
    averages = averages.merge(solicit_contact,how='outer',on='Unique_CustomerID')

    current_month = valid_accounts.loc[valid_accounts['CurrentMonth'] >= dtmonth][['Unique_ContractID', 'Contacted_Memos']].rename(columns={'Contacted_Memos':'Contact_Current'})
    prev_month = valid_accounts.loc[(valid_accounts['CurrentMonth'] >= (dtmonth - pd.DateOffset(months=1))) & (valid_accounts['CurrentMonth'] < dtmonth)][['Unique_ContractID', 'Contacted_Memos']].rename(columns={'Contacted_Memos':'Contact_Prev'})
    contact_check = prev_month.merge(current_month, on='Unique_ContractID',how='outer')
    contact_check['greater_than?'] = 0
    contact_check.loc[contact_check['Contact_Current'] > contact_check['Contact_Prev'], 'greater_than?'] = 1

    origin_and_perf = origin_and_perf.merge(averages,how='left',on='Unique_CustomerID').merge(contact_check[['Unique_ContractID','greater_than?']],how='left',on='Unique_ContractID')
    origin_and_perf['avg_monthonbook_dl_r'].fillna(value=averages['avg_monthonbook_dl_r'].mean(), inplace=True)
    origin_and_perf['avg_monthonbook_lc_r'].fillna(value=averages['avg_monthonbook_lc_r'].mean(), inplace=True)
    origin_and_perf['avg_monthonbook_dl_c'].fillna(value=averages['avg_monthonbook_dl_c'].mean(), inplace=True)
    origin_and_perf['avg_monthonbook_lc_c'].fillna(value=averages['avg_monthonbook_lc_c'].mean(), inplace=True)

    return origin_and_perf

def add_credit_bin_high_credit(origin):
    # Invalid credit scores filtered out before binning
    origin = origin.loc[(origin['CreditScore']>=300) & (origin['CreditScore']<=800)]
    origin.loc[(origin['CreditScore'] >= 300) & (origin['CreditScore'] <= 566), 'credit_binned'] = 1
    origin.loc[(origin['CreditScore'] >= 567) & (origin['CreditScore'] <= 602), 'credit_binned'] = 2
    origin.loc[(origin['CreditScore'] >= 603) & (origin['CreditScore'] <= 639), 'credit_binned'] = 3
    origin.loc[(origin['CreditScore'] >= 640) & (origin['CreditScore'] <= 696), 'credit_binned'] = 4
    origin.loc[(origin['CreditScore'] >= 697) & (origin['CreditScore'] <= 800), 'credit_binned'] = 5

    origin['HighCredit'] = 0
    origin.loc[(origin['Tier_MultipleModels'] == 1) & (origin['ProductType'] != 'Auto'), 'HighCredit'] = 10000
    origin.loc[(origin['Tier_MultipleModels'] == 2) & (origin['ProductType'] != 'Auto'), 'HighCredit'] = 7000
    origin.loc[(origin['Tier_MultipleModels'] == 3) & (origin['ProductType'] != 'Auto'), 'HighCredit'] = 4500
    origin.loc[(origin['Tier_MultipleModels'] == 4) & (origin['ProductType'] != 'Auto'), 'HighCredit'] = 3000
    origin.loc[(origin['Tier_MultipleModels'] == 1) & (origin['ProductType'] == 'Auto'), 'HighCredit']= 12500
    origin.loc[(origin['Tier_MultipleModels'] == 2) & (origin['ProductType'] == 'Auto'), 'HighCredit'] = 7700
    origin.loc[(origin['Tier_MultipleModels'] == 3) & (origin['ProductType'] == 'Auto'), 'HighCredit'] = 4050
    origin.loc[(origin['Tier_MultipleModels'] == 4) & (origin['ProductType'] == 'Auto'), 'HighCredit'] = 2250

    return origin

# %% ORIGINATION AND PERFORMANCE TABLE FEATURE SETUP
origin_cols = ['Unique_ContractID','Unique_CustomerID','Unique_BranchID','BookDate','State','ProductType','CreditScore','Term','AmountFinanced','NetCash','CashToCustomer','Tier_MultipleModels']
origin_dtypes = {'State':'category','ProductType':'category','Term':'int8','CreditScore':'int16','Unique_BranchID':'int16'}
origin = pd.read_csv(origination_file, sep=',',usecols=origin_cols,dtype=origin_dtypes,parse_dates=['BookDate'])

perf_cols = ['Unique_ContractID','MonthsOnBook','30+_Indicator','GrossBalance','ProcessStatus','Solicitation_Memos','Contacted_Memos','NetReceivable']
perf_dtypes = {'MonthsOnBook':'int16','30+_Indicator':'int8','ProcessStatus':'category'}
perf = pd.read_csv(perffile1, sep=',',usecols=perf_cols,dtype=perf_dtypes).append(pd.read_csv(perffile2, sep=',',usecols=perf_cols,dtype=perf_dtypes)).append(pd.read_csv(perffile3, sep=',',usecols=perf_cols,dtype=perf_dtypes)).append(pd.read_csv(perffile4, sep=',',usecols=perf_cols,dtype=perf_dtypes))

origin = add_customer_history_features(origin)

state_rank_map = {'MS':100, # State ranks were determined by relative google trends
                    'LA':76, # popularity of search term 'republic finance' by state.
                    'SC':56, # This is a very rough indicator of RF's state reputation
                    'TN':36,
                    'AL':50,
                    'GA':24,
                    'TX':8,
                    'KY':18}
origin['StRank'] = origin['State'].map(state_rank_map)

# Live check risk tiers must be shifted
origin.loc[(origin.ProductType == 'LC') & (origin.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
origin.loc[(origin.ProductType == 'LC') & (origin.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
origin.loc[(origin.ProductType == 'LC') & (origin.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
origin.loc[(origin.ProductType == 'LC') & (origin.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
origin.loc[(origin.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1

# Other product types must be removed
origin = origin.loc[(origin.ProductType != 'Sales')&(origin.ProductType != 'MH')&(origin.ProductType != 'RE')]

# Imputing missing risk tiers with existing ones, segmented by state and product
for state in origin.State.unique():
    for product in origin.ProductType.unique():
        slice = origin.loc[(origin.State == state) & (origin.ProductType == product)&(~pd.isnull(origin.Tier_MultipleModels))]
        tier = slice.Tier_MultipleModels.value_counts(normalize=True)
        origin.loc[(origin.State == state) & (origin.ProductType == product)&(pd.isnull(origin.Tier_MultipleModels)),'Tier_MultipleModels'] = np.random.choice(tier.index, size=len(origin.loc[(origin.State == state) & (origin.ProductType == product)&(pd.isnull(origin.Tier_MultipleModels))]), p=tier.values)

origin = add_credit_bin_high_credit(origin)

# %% MERGING PERFORMANCE AND ORIGINATIONS
# creating a empty bucket to save result
origin_and_perf = pd.DataFrame(columns=(origin.columns.append(perf.columns)).unique())
origin_and_perf.to_csv('origin_and_perf.csv',index_label=False)

# deleting df2 to save memory
del(perf)

def mergeprocess(x):
    perf=pd.merge(origin,x, on='Unique_ContractID',how='left')
    perf.to_csv('origin_and_perf.csv',mode='a',header=False,index=False)

perffiles = [pd.read_csv(perffile1, sep=',',usecols=perf_cols,dtype=perf_dtypes),pd.read_csv(perffile2, sep=',',usecols=perf_cols,dtype=perf_dtypes),pd.read_csv(perffile3, sep=',',usecols=perf_cols,dtype=perf_dtypes),pd.read_csv(perffile4, sep=',',usecols=perf_cols,dtype=perf_dtypes)]

for chunk in perffiles:
    mergeprocess(chunk)

del(origin)

origin_and_perf_dtypes = {'State':'category','ProductType':'category','Term':'int8','CreditScore':'int16','Unique_BranchID':'int16','ProcessStatus':'category'}
origin_and_perf = pd.read_csv('origin_and_perf.csv',dtype=origin_and_perf_dtypes)
os.remove('origin_and_perf.csv')
origin_and_perf['CurrentMonth'] = origin_and_perf['BookDate'].values.astype('datetime64[M]') + origin_and_perf['MonthsOnBook'].values.astype('timedelta64[M]')
origin_and_perf['Contacted_Memos'].fillna(0,inplace=True)
origin_and_perf['Solicitation_Memos'].fillna(0,inplace=True)
origin_and_perf = add_customer_average_features(origin_and_perf)

# %% FINAL CALCULATED FEATURES

# Determining response variables
labels = origin_and_perf.loc[(origin_and_perf['ProcessStatus'] == 'Renewed') & (origin_and_perf['CurrentMonth'] < (dtmonth + pd.DateOffset(months=2))) & (origin_and_perf['CurrentMonth'] >= (dtmonth + pd.DateOffset(months=1)))]
labels['Renewed?'] = 1
labels = labels[['Renewed?', 'Unique_ContractID']]
origin_and_perf = origin_and_perf.merge(labels, on='Unique_ContractID', how='left')
label1 = origin_and_perf.loc[(origin_and_perf['ProcessStatus'] == 'Closed') & (origin_and_perf['CurrentMonth'] < (dtmonth + pd.DateOffset(months=2))) & (origin_and_perf['CurrentMonth'] >= (dtmonth + pd.DateOffset(months=1)))]
label1['Closed?'] = 1
label1 = label1[['Closed?', 'Unique_ContractID']]
label2 = origin_and_perf.loc[(origin_and_perf['ProcessStatus'] == 'Closed') & (origin_and_perf['MonthsOnBook'] < origin_and_perf['Term']) & (origin_and_perf['CurrentMonth'] < (dtmonth + pd.DateOffset(months=2))) & (origin_and_perf['CurrentMonth'] >= (dtmonth + pd.DateOffset(months=1)))]
label2['PaidOrCharged?'] = 1
label2 = label2[['PaidOrCharged?', 'Unique_ContractID']]
origin_and_perf = origin_and_perf.merge(label1, on='Unique_ContractID', how='left').merge(label2, on='Unique_ContractID', how='left')

origin_and_perf['Closed?'].fillna(value=0, inplace=True)
origin_and_perf['PaidOrCharged?'].fillna(value=0, inplace=True)
origin_and_perf['Renewed?'].fillna(value=0, inplace=True)
origin_and_perf['LC_ratio'].fillna(value=0, inplace=True)
origin_and_perf['SC_ratio'].fillna(value=0, inplace=True)

origin_and_perf = origin_and_perf.loc[(origin_and_perf['CurrentMonth']<(dtmonth+pd.DateOffset(months=1)))&(origin_and_perf['CurrentMonth']>=dtmonth)]
origin_and_perf = origin_and_perf[origin_and_perf['MonthsOnBook'] >= 1]

# Filling empty values
origin_and_perf[['contract_rank', 'NetReceivable', 'greater_than?','GrossBalance']].fillna(value=0,inplace=True)

# Calculated MoB dependents
origin_and_perf['months_til_avg_dl_r'] = origin_and_perf['avg_monthonbook_dl_r'] - origin_and_perf['MonthsOnBook']
origin_and_perf['months_til_avg_lc_r'] = origin_and_perf['avg_monthonbook_lc_r'] - origin_and_perf['MonthsOnBook']
origin_and_perf['months_til_avg_dl_c'] = origin_and_perf['avg_monthonbook_dl_c'] - origin_and_perf['MonthsOnBook']
origin_and_perf['months_til_avg_lc_c'] = origin_and_perf['avg_monthonbook_lc_c'] - origin_and_perf['MonthsOnBook']
origin_and_perf['Months_left'] = origin_and_perf['Term'] - origin_and_perf['MonthsOnBook']

# Determining high credit dependents
origin_and_perf['available_offer'] = 0
origin_and_perf.loc[origin_and_perf['HighCredit'] > origin_and_perf['NetReceivable'], 'available_offer'] = origin_and_perf['HighCredit'] - origin_and_perf['NetReceivable']
origin_and_perf = origin_and_perf.loc[origin_and_perf.NetReceivable >= 0]
origin_and_perf['Utilization'] = 1
origin_and_perf.loc[origin_and_perf.GrossBalance < origin_and_perf.HighCredit, 'Utilization'] = origin_and_perf['GrossBalance'] / origin_and_perf['HighCredit']
origin_and_perf['GrossBalance'].isna().sum()
origin_and_perf['PaydownPercent'] = 0
origin_and_perf.loc[origin_and_perf.NetReceivable < origin_and_perf.AmountFinanced, 'PaydownPercent'] = 1 - (origin_and_perf.NetReceivable / origin_and_perf.AmountFinanced)

# %% SLICING AND OUTPUTING DATAFRAME
origin_and_perf.to_csv('feature_generation_sep_2018.csv', index=False)

# %%
import winsound
duration = 3000  # milliseconds
freq = 800  # Hz
winsound.Beep(freq, duration)
