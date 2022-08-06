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
outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'

data1 = datafolder / 'test_set_march_1_month_both_full.csv'
test_set = pd.read_csv(data1, sep=',', low_memory=False)

dl_correlations = test_set.drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'credit_binned', 'GrossBalance', 'SC_ratio', 'greater_than?', 'StRank', 'NetCash', 'CreditScore', 'MonthsOnBook', 'Utilization', 'months_til_avg'])

lc_correlations = test_set.drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'GrossBalance', 'CreditScore', 'available_offer', 'Utilization', 'Months_left', 'PaydownPercent', 'MonthsOnBook'])

att_correlations = test_set.drop(columns=['RiskTier','Rescored_Tier_2018Model','Rescored_Tier_2017Model','counter','Approved_Apps_x','Approved_Apps_y','prev_month','current_month','HighCredit','indicator','BookDate','OwnRent','ProcessStatus','CurrentMonth','Unique_CustomerID','Unique_BranchID','Unique_ContractID','Tier_MultipleModels','credit_binned','Contacted_Memos','Declined_Apps','greater_than?', 'ProductType','Term','TotalOldBalance','contract_rank','LC_ratio','Solicitation_Memos','Avg_MonthsOnBook','SC_ratio','available_offer','Utilization', 'CreditScore', 'AmountFinanced', 'NetCash', 'StRank', 'Renewed?'])
#%% Heatmaps

sns.heatmap(dl_correlations.corr(), vmin=-1, vmax=1)

sns.heatmap(lc_correlations.corr(), vmin=-1, vmax=1)

sns.heatmap(att_correlations.corr(), vmin=-1, vmax=1)
