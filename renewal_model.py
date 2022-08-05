#######################
# %% import and files
#######################


import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.utils import compute_sample_weight
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


###############
# %% DATA LOAD
###############


datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'


data = datafolder / 'feature_gen_dists_apr_2019.csv'
train = pd.read_csv(data, sep=',', low_memory=False)
#train_set.drop(columns=['CreditScore'], inplace=True)


data1 = datafolder / 'feature_gen_dists_may_2019.csv'
test = pd.read_csv(data1, sep=',', low_memory=False)
#test_set_validate.drop(columns=['CreditScore'], inplace=True)

X_train_lc = train.loc[train['ProductType'] == 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'available_offer', 'Utilization', 'Months_left', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_dl_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_dl_r', 'months_added', 'step_one', 'CreditScore'])
X_train_dl = train.loc[train['ProductType'] != 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'credit_binned', 'GrossBalance', 'SC_ratio', 'greater_than?', 'StRank', 'NetCash', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_lc_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_lc_r', 'months_added', 'step_one', 'CreditScore'])
y_train_dl = train.loc[train['ProductType'] != 'LC']['Renewed?']
y_train_lc = train.loc[train['ProductType'] == 'LC']['Renewed?']


X_test_lc = test.loc[test['ProductType'] == 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'available_offer', 'Utilization', 'Months_left', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_dl_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_dl_r', 'months_added', 'step_one', 'CreditScore'])
X_test_dl = test.loc[test['ProductType'] != 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'credit_binned', 'GrossBalance', 'SC_ratio', 'greater_than?', 'StRank', 'NetCash', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_lc_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_lc_r', 'months_added', 'step_one', 'CreditScore'])
y_test_dl = test.loc[test['ProductType'] != 'LC']['Renewed?']
y_test_lc = test.loc[test['ProductType'] == 'LC']['Renewed?']


sample_weight_data_train_lc = compute_sample_weight("balanced", y_train_lc)
sample_weight_data_train_dl = compute_sample_weight("balanced", y_train_dl)


####################
# %% RENEWAL MODEL
####################


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

xgb_clf_lc = XGBClassifier(n_estimators=100, learning_rate=0.05,max_features=4, max_depth=5, random_state=0)
xgb_clf_lc.fit(X_train_lc, y_train_lc)
predictions_lc = xgb_clf_lc.predict(X_test_lc)
predictions2_lc = xgb_clf_lc.predict_proba(X_test_lc)

xgb_clf_dl = XGBClassifier(n_estimators=100, learning_rate=0.05,max_features=4, max_depth=5, random_state=0)
xgb_clf_dl.fit(X_train_dl, y_train_dl)
predictions_dl = xgb_clf_dl.predict(X_test_dl)
predictions2_dl = xgb_clf_dl.predict_proba(X_test_dl)

proba_lc = pd.DataFrame(predictions2_lc)
proba_dl = pd.DataFrame(predictions2_dl)
#pred_lc = pd.DataFrame(predictions_lc)
#pred_dl = pd.DataFrame(predictions_dl)

sim_lc = X_test_lc
sim_dl = X_test_dl


sim_lc['proba'] = proba_lc[1].values
sim_dl['proba'] = proba_dl[1].values
#sim_lc['predictions'] = pred_lc[0].values
#sim_dl['predictions'] = pred_dl[0].values

#######################
# %% SIMULATE RESULTS
#######################
proba_avg_lc = proba_lc[1].mean()
num_lc = round(proba_avg_lc*len(y_test_lc))
print(num_lc)
print(proba_avg_lc)
proba_avg_dl = proba_dl[1].mean()
num_dl = round(proba_avg_dl*len(y_test_dl))
print(num_dl)
print(proba_avg_lc)

sim_lc[['Unique_ContractID', 'Unique_BranchID']] = test.loc[test['ProductType'] == 'LC'][['Unique_ContractID', 'Unique_BranchID']]
sim_dl[['Unique_ContractID', 'Unique_BranchID']] = test.loc[test['ProductType'] != 'LC'][['Unique_ContractID', 'Unique_BranchID']]

lc_renewed_actuals = test.loc[test['ProductType'] == 'LC'][['Unique_BranchID', 'Renewed?']]
dl_renewed_actuals = test.loc[test['ProductType'] != 'LC'][['Unique_BranchID', 'Renewed?']]

actual_lc = len(lc_renewed_actuals[lc_renewed_actuals['Renewed?'] == 1])
actual_dl = len(dl_renewed_actuals[dl_renewed_actuals['Renewed?'] == 1])
print(actual_lc)
print(actual_dl)


print(((num_lc+num_dl) - (actual_lc+actual_dl))/(actual_lc+actual_dl)*100)

#actuals = pd.merge(sim_lc.loc[sim_lc['uniform'] <= sim_lc['proba']], lc_renewed_actuals.loc[lc_renewed_actuals['Renewed?'] == 1], on='Unique_BranchID')
