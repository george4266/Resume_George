# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'

# %% data clean
origin = pd.read_csv(origination_file, sep=',', low_memory=False)
origin2 = origin
origin2['counter'] = 1
origin2['BookDate'] = pd.to_datetime(origin2['BookDate'])
stuff = origin2[origin2['BookDate'] < dt.datetime(2019,9,1)]
live_checks = stuff[stuff['ProductType'] == 'LC'].groupby('Unique_CustomerID')['counter'].sum()
total_origs = stuff.groupby('Unique_CustomerID')['counter'].sum()
origs_ratio = pd.merge(live_checks, total_origs, on='Unique_CustomerID')
origs_ratio['LC_ratio'] = origs_ratio['counter_x']/origs_ratio['counter_y']
origs_ratio.reset_index(inplace=True)
origs_ratio = origs_ratio[['Unique_CustomerID', 'LC_ratio']]

origin = origin[(origin['CreditScore']>=300) & (origin['CreditScore']<=800)]
origin['StRank'] = 0
origin.loc[origin['State'] == 'MS', 'StRank'] = 100
origin.loc[origin['State'] == 'LA', 'StRank'] = 76
origin.loc[origin['State'] == 'SC', 'StRank'] = 56
origin.loc[origin['State'] == 'TN', 'StRank'] = 36
origin.loc[origin['State'] == 'AL', 'StRank'] = 50
origin.loc[origin['State'] == 'GA', 'StRank'] = 24
origin.loc[origin['State'] == 'TX', 'StRank'] = 8
origin.loc[origin['State'] == 'KY', 'StRank'] = 18

origin.loc[(origin['CreditScore'] >= 300) & (origin['CreditScore'] <= 566), 'credit_binned'] = 1
origin.loc[(origin['CreditScore'] >= 567) & (origin['CreditScore'] <= 602), 'credit_binned'] = 2
origin.loc[(origin['CreditScore'] >= 603) & (origin['CreditScore'] <= 639), 'credit_binned'] = 3
origin.loc[(origin['CreditScore'] >= 640) & (origin['CreditScore'] <= 696), 'credit_binned'] = 4
origin.loc[(origin['CreditScore'] >= 697) & (origin['CreditScore'] <= 800), 'credit_binned'] = 5

#origin.drop(columns=['OwnRent', 'State', 'AmountFinanced', 'TotalNote', 'NetCash', 'CashToCustomer', 'Segment', 'IP_Unique_ContractID', 'RegularPayment', 'Unique_ApplicationID'], inplace=True)
origin.drop(columns=['TotalNote', 'CashToCustomer', 'Segment', 'IP_Unique_ContractID', 'RegularPayment', 'Unique_ApplicationID'], inplace=True)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))
#perf.drop(columns=['30+_Indicator'], inplace=True)
origin['indicator'] = 1
origin['BookDate'] = pd.to_datetime(origin['BookDate'])


#perf.loc[perf['ProcessStatus'] == 'Renewed']['MonthsOnBook'].mean()

# %% Feature generation pt. 1 (cntract rank and creating combined df)
contract_rank = origin.loc[origin['BookDate'] < dt.datetime(2019,9,1)].groupby(origin.Unique_CustomerID)['indicator'].sum().reset_index()
contract_rank.rename(columns={'indicator':'contract_rank'}, inplace=True)
contract_rank = contract_rank[['contract_rank', 'Unique_CustomerID']]
combined = origin.merge(perf, on='Unique_ContractID', how='left')
combined.drop(columns=['Unique_CustomerID_y', 'Unique_BranchID_y'], inplace=True)
combined.rename(columns={'Unique_CustomerID_x':'Unique_CustomerID','Unique_BranchID_x':'Unique_BranchID'}, inplace=True)
combined.dropna(subset=['MonthsOnBook'], inplace=True)


# %% Super long apply function

#combined['CurrentMonth'] = combined[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)
combined['months_added'] = pd.to_timedelta(combined['MonthsOnBook'], 'M')
combined['step_one'] = combined['BookDate'] + combined['months_added']
combined['CurrentMonth'] = combined['step_one'].dt.strftime('%m/%Y')
combined['CurrentMonth'] = pd.to_datetime(combined['CurrentMonth'])


# %% More processing

approved_rank_ratio = combined.loc[combined['CurrentMonth'] < dt.datetime(2019,9,1)]
approved_rank_ratio = approved_rank_ratio[['Unique_CustomerID', 'Approved_Apps']]
approved_rank_ratio['Approved_Apps'].fillna(value=0, inplace=True)
approved_rank_ratio = approved_rank_ratio.groupby(approved_rank_ratio.Unique_CustomerID)['Approved_Apps'].sum().reset_index()


# %% Average months on book by loan type and renewal/closed

avg_monthonbook = combined.loc[combined['CurrentMonth'] < dt.datetime(2019,9,1)]
avg_monthonbook = avg_monthonbook[['Unique_CustomerID', 'ProcessStatus', 'MonthsOnBook', 'ProductType']]

avg_monthonbook_dl_r = avg_monthonbook.loc[(avg_monthonbook['ProcessStatus'] == 'Renewed') & (avg_monthonbook['ProductType'] != 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index()
ind_avg_monthonbook_dl_r = avg_monthonbook_dl_r['MonthsOnBook'].mean()
avg_monthonbook_dl_r.rename(columns={'MonthsOnBook':'avg_monthonbook_dl_r'}, inplace=True)
avg_monthonbook_dl_r['avg_monthonbook_dl_r'].fillna(value=ind_avg_monthonbook_dl_r, inplace=True)

avg_monthonbook_lc_r = avg_monthonbook.loc[(avg_monthonbook['ProcessStatus'] == 'Renewed') & (avg_monthonbook['ProductType'] == 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index()
ind_avg_monthonbook_lc_r = avg_monthonbook_lc_r['MonthsOnBook'].mean()
avg_monthonbook_lc_r.rename(columns={'MonthsOnBook':'avg_monthonbook_lc_r'}, inplace=True)
avg_monthonbook_lc_r['avg_monthonbook_lc_r'].fillna(value=ind_avg_monthonbook_lc_r, inplace=True)

avg_monthonbook_dl_c = avg_monthonbook.loc[(avg_monthonbook['ProcessStatus'] == 'Closed') & (avg_monthonbook['ProductType'] != 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index()
ind_avg_monthonbook_dl_c = avg_monthonbook_dl_c['MonthsOnBook'].mean()
avg_monthonbook_dl_c.rename(columns={'MonthsOnBook':'avg_monthonbook_dl_c'}, inplace=True)
avg_monthonbook_dl_c['avg_monthonbook_dl_c'].fillna(value=ind_avg_monthonbook_dl_c, inplace=True)

avg_monthonbook_lc_c = avg_monthonbook.loc[(avg_monthonbook['ProcessStatus'] == 'Closed') & (avg_monthonbook['ProductType'] == 'LC')].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index()
ind_avg_monthonbook_lc_c = avg_monthonbook_lc_c['MonthsOnBook'].mean()
avg_monthonbook_lc_c.rename(columns={'MonthsOnBook':'avg_monthonbook_lc_c'}, inplace=True)
avg_monthonbook_lc_c['avg_monthonbook_lc_c'].fillna(value=ind_avg_monthonbook_lc_c, inplace=True)


# %% SC_ratio, greater_than, prep for merging dfs

combined['Contacted_Memos'].fillna(value=0, inplace=True)
solicit_contact = combined[['Contacted_Memos', 'CurrentMonth', 'Solicitation_Memos', 'Unique_CustomerID']]
solicit_contact = solicit_contact[solicit_contact['CurrentMonth'] < dt.datetime(2019,9,1)]
contacts = solicit_contact.groupby('Unique_CustomerID')['Contacted_Memos'].sum().reset_index()
solicits = solicit_contact.groupby('Unique_CustomerID')['Solicitation_Memos'].sum().reset_index()
solicit_contact = solicits.merge(contacts, on='Unique_CustomerID', how='left')
solicit_contact['SC_ratio'] = 0
solicit_contact.loc[(solicit_contact['Contacted_Memos'] !=0) & (solicit_contact['Solicitation_Memos'] != 0), 'SC_ratio'] = solicit_contact['Contacted_Memos']/solicit_contact['Solicitation_Memos']
solicit_contact = solicit_contact[['Unique_CustomerID', 'SC_ratio']]

combined.isna().sum()
current_month = combined.loc[(combined['CurrentMonth'] >= dt.datetime(2019,8,1)) & (combined['CurrentMonth'] < dt.datetime(2019,9,1))][['Unique_ContractID', 'Contacted_Memos']]
current_month['current_month'] = current_month['Contacted_Memos']
current_month.drop(columns=['Contacted_Memos'], inplace=True)
prev_month = combined.loc[(combined['CurrentMonth'] >= dt.datetime(2019,7,1)) & (combined['CurrentMonth'] < dt.datetime(2019,8,1))][['Unique_ContractID', 'Contacted_Memos']]
prev_month['prev_month'] = prev_month['Contacted_Memos']
prev_month.drop(columns=['Contacted_Memos'], inplace=True)
contact_check = pd.merge(prev_month, current_month, on='Unique_ContractID')
contact_check['greater_than?'] = 0
contact_check.isna().sum()
contact_check.loc[contact_check['current_month'] > contact_check['prev_month'], 'greater_than?'] = 1


no_rank_test = combined.loc[(combined['CurrentMonth'] >= dt.datetime(2019,8,1)) & (combined['CurrentMonth'] < dt.datetime(2019,9,1))]



# %% feature finalizing

test_set = no_rank_test.merge(contract_rank, on='Unique_CustomerID', how='left')
test_set = test_set.merge(approved_rank_ratio, on='Unique_CustomerID', how='left')
test_set = test_set.merge(avg_monthonbook_dl_c, on='Unique_CustomerID', how='left')
test_set = test_set.merge(avg_monthonbook_dl_r, on='Unique_CustomerID', how='left')
test_set = test_set.merge(avg_monthonbook_lc_c, on='Unique_CustomerID', how='left')
test_set = test_set.merge(avg_monthonbook_lc_r, on='Unique_CustomerID', how='left')
test_set = test_set.merge(solicit_contact, on='Unique_CustomerID', how='left')
test_set = test_set.merge(origs_ratio, on='Unique_CustomerID', how='left')
test_set = test_set.merge(contact_check, on='Unique_ContractID', how='left')

test_set[['contract_rank', 'NetReceivable', 'greater_than?']] = test_set[['contract_rank', 'NetReceivable', 'greater_than?']].fillna(value=0)
test_set['avg_monthonbook_dl_r'].fillna(value=ind_avg_monthonbook_dl_r, inplace=True)
test_set['avg_monthonbook_lc_r'].fillna(value=ind_avg_monthonbook_lc_r, inplace=True)
test_set['avg_monthonbook_dl_c'].fillna(value=ind_avg_monthonbook_dl_c, inplace=True)
test_set['avg_monthonbook_lc_c'].fillna(value=ind_avg_monthonbook_lc_c, inplace=True)
test_set.dropna(subset=['Tier_MultipleModels'], inplace=True)

#test_set['approved_vs_rank'] = test_set['Approved_Apps'] - test_set['contract_rank']

# LC risk tiers are on a different scale than other product types
test_set.loc[(test_set.ProductType == 'LC') & (test_set.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
test_set.loc[(test_set.ProductType == 'LC') & (test_set.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
test_set.loc[(test_set.ProductType == 'LC') & (test_set.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
test_set.loc[(test_set.ProductType == 'LC') & (test_set.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
test_set.loc[(test_set.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1


test_set['HighCredit'] = 0
test_set.loc[(test_set['Tier_MultipleModels'] == 1) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 10000
test_set.loc[(test_set['Tier_MultipleModels'] == 2) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 7000
test_set.loc[(test_set['Tier_MultipleModels'] == 3) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 4500
test_set.loc[(test_set['Tier_MultipleModels'] == 4) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 3000
test_set.loc[(test_set['Tier_MultipleModels'] == 1) & (test_set['ProductType'] == 'Auto'), 'HighCredit']= 12500
test_set.loc[(test_set['Tier_MultipleModels'] == 2) & (test_set['ProductType'] == 'Auto'), 'HighCredit'] = 7700
test_set.loc[(test_set['Tier_MultipleModels'] == 3) & (test_set['ProductType'] == 'Auto'), 'HighCredit'] = 4050
test_set.loc[(test_set['Tier_MultipleModels'] == 4) & (test_set['ProductType'] == 'Auto'), 'HighCredit'] = 2250

test_set['months_til_avg_dl_r'] = test_set['avg_monthonbook_dl_r'] - test_set['MonthsOnBook']
test_set['months_til_avg_lc_r'] = test_set['avg_monthonbook_lc_r'] - test_set['MonthsOnBook']
test_set['months_til_avg_dl_c'] = test_set['avg_monthonbook_dl_c'] - test_set['MonthsOnBook']
test_set['months_til_avg_lc_c'] = test_set['avg_monthonbook_lc_c'] - test_set['MonthsOnBook']
test_set['available_offer'] = 0
test_set.isna().sum()
test_set.loc[test_set['HighCredit'] > test_set['NetReceivable'], 'available_offer'] = test_set['HighCredit'] - test_set['NetReceivable']

# %% Adding in utilization and paydown percentage
test_set = test_set.loc[test_set.NetReceivable >= 0]
test_set['Utilization'] = 1
test_set.loc[test_set.GrossBalance < test_set.HighCredit, 'Utilization'] = test_set['GrossBalance'] / test_set['HighCredit']
test_set['PaydownPercent'] = 0
test_set.loc[test_set.NetReceivable < test_set.AmountFinanced, 'PaydownPercent'] = test_set.NetReceivable / test_set.AmountFinanced

# %% To create attrition model test set
label1 = combined.loc[(combined['ProcessStatus'] == 'Closed') & (combined['CurrentMonth'] < dt.datetime(2019,10,1)) & (combined['CurrentMonth'] >= dt.datetime(2019,9,1))]
label1['Closed?'] = 1
label1 = label1[['Closed?', 'Unique_ContractID']]

label2 = combined.loc[(combined['ProcessStatus'] == 'Closed') & (combined['MonthsOnBook'] < combined['Term']) & (combined['CurrentMonth'] < dt.datetime(2019,10,1)) & (combined['CurrentMonth'] >= dt.datetime(2019,9,1))]
label2['PaidOrCharged?'] = 1
label2 = label2[['PaidOrCharged?', 'Unique_ContractID']]

test_set = test_set.merge(label1, on='Unique_ContractID', how='left').merge(label2, on='Unique_ContractID', how='left')

test_set['Months_left'] = test_set['Term'] - test_set['MonthsOnBook']
test_set[['Closed?','PaidOrCharged?','Declined_Apps', 'Solicitation_Memos', 'LC_ratio', 'SC_ratio']] = test_set[['Closed?','PaidOrCharged?','Declined_Apps', 'Solicitation_Memos', 'LC_ratio', 'SC_ratio']].fillna(value=0)

# %%
test_set = test_set[test_set['MonthsOnBook'] >= 1]
labels = combined.loc[(combined['ProcessStatus'] == 'Renewed') & (combined['CurrentMonth'] < dt.datetime(2019,10,1)) & (combined['CurrentMonth'] >= dt.datetime(2019,9,1))]
labels['Renewed?'] = 1
labels = labels[['Renewed?', 'Unique_ContractID']]

test_set = test_set.merge(labels, on='Unique_ContractID', how='left')
test_set['Months_left'] = test_set['Term'] - test_set['MonthsOnBook']


test_set[['Renewed?', 'Declined_Apps', 'Solicitation_Memos', 'LC_ratio', 'SC_ratio']] = test_set[[ 'Renewed?', 'Declined_Apps', 'Solicitation_Memos', 'LC_ratio', 'SC_ratio']].fillna(value=0)

test_set
len(test_set.loc[test_set['Tier_MultipleModels'].isnull()])
# %%
#test_set = test_set[test_set['ProductType'] != 'LC']
test_set.dropna(subset=['GrossBalance'], inplace=True)
test_set = test_set.loc[test_set['ProductType'] != 'Sales']
#test_set.drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'Tier_MultipleModels', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'Avg_MonthsOnBook', 'counter', 'Months_left', 'SC_ratio', 'StRank', 'credit_binned', 'prev_month', 'current_month', 'greater_than?'], inplace=True)
#test_set.drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'Avg_MonthsOnBook', 'counter', 'prev_month', 'current_month'], inplace=True)

#test_set.to_csv(outputfolder / 'attrition_test_set_march_1_month_both_full.csv', index=False)
#len(test_set)
#test_set['Renewed?'].mean()
test_set.to_csv(outputfolder / 'feature_gen_dists_aug_2019.csv', index=False)
test_set.columns


# %% To add in local factors, later?
