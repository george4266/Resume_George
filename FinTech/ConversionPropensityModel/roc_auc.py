# %% import and files
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

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'



'''data = datafolder / 'test_set_march_1_month_LC_full.csv'
test_set_lc = pd.read_csv(data, sep=',', low_memory=False)
test_set_lc.drop(columns=['CreditScore'], inplace=True)

data = datafolder / 'test_set_march_1_month_DL_full.csv'
test_set_dl = pd.read_csv(data, sep=',', low_memory=False)
test_set_dl.drop(columns=['CreditScore'], inplace=True)'''

data = datafolder / 'feature_gen_dists_mar_2019.csv'
train_set = pd.read_csv(data, sep=',', low_memory=False)
train_set.drop(columns=['CreditScore'], inplace=True)
#train_set.drop(columns=['OwnRent', 'credit_binned'], inplace=True)

#test_set_lc['OwnRent'] = test_set_lc['OwnRent'].map({'O':0, 'R':1})
train_set['OwnRent'] = train_set['OwnRent'].map({'O':0, 'R':1})
#test_set_dl.drop(columns=['OwnRent', 'credit_binned', 'StRank', 'SC_ratio'], inplace=True)

'''data1 = datafolder / 'test_set_feb_1_month_dl_full.csv'
test_set_validate_dl = pd.read_csv(data1, sep=',', low_memory=False)
test_set_validate_dl.drop(columns=['CreditScore'], inplace=True)
test_set_validate_dl.drop(columns=['StRank', 'SC_ratio', 'credit_binned', 'OwnRent'], inplace=True)

data1 = datafolder / 'test_set_feb_1_month_lc_full.csv'
test_set_validate_lc = pd.read_csv(data1, sep=',', low_memory=False)
test_set_validate_lc.drop(columns=['CreditScore'], inplace=True)
test_set_validate_lc['OwnRent'] = test_set_validate_lc['OwnRent'].map({'O':0, 'R':1})'''

data1 = datafolder / 'feature_gen_dists_apr_2019.csv'
test_set_validate = pd.read_csv(data1, sep=',', low_memory=False)
test_set_validate.drop(columns=['CreditScore'], inplace=True)
#test_set_validate_co.drop(columns=['OwnRent', 'credit_binned'], inplace=True)
test_set_validate['OwnRent'] = test_set_validate['OwnRent'].map({'O':0, 'R':1})

#train_set.columns


#%%  Generate test sets

'''features_lc = test_set_lc.drop(columns=['Renewed?'])
labels_lc = test_set_lc['Renewed?']
X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(features_lc, labels_lc, test_size=0.4, random_state=42)
X_test_lc, X_val_lc, y_test_lc, y_val_lc = train_test_split(X_test_lc, y_test_lc, test_size=0.5, random_state=42)

features_dl = test_set_dl.drop(columns=['Renewed?'])
labels_dl = test_set_dl['Renewed?']
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(features_dl, labels_dl, test_size=0.4, random_state=42)
X_test_dl, X_val_dl, y_test_dl, y_val_dl = train_test_split(X_test_dl, y_test_dl, test_size=0.5, random_state=42)'''

#features_co = train_set.drop(columns=['Renewed?'])
#labels_co = train_set['Renewed?']
train, test = train_test_split(train_set, test_size=0.3, random_state=42)
#test, val = train_test_split(test, test_size=0.5, random_state=42)
#X_test = test.drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent'])
#y_test = test['Renewed?']

#X_train_co = train.drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'Utilization'])
#y_train_co = train['Renewed?']

X_train_lc = train.loc[train['ProductType'] == 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'available_offer', 'Utilization', 'Months_left', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_dl_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_dl_r', 'months_added', 'step_one'])
X_train_dl = train.loc[train['ProductType'] != 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'credit_binned', 'GrossBalance', 'SC_ratio', 'greater_than?', 'StRank', 'NetCash', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_lc_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_lc_r', 'months_added', 'step_one'])
y_train_dl = train.loc[train['ProductType'] != 'LC']['Renewed?']
y_train_lc = train.loc[train['ProductType'] == 'LC']['Renewed?']
X_test_lc = test.loc[test['ProductType'] == 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'available_offer', 'Utilization', 'Months_left', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_dl_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_dl_r', 'months_added', 'step_one'])
X_test_dl = test.loc[test['ProductType'] != 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'credit_binned', 'GrossBalance', 'SC_ratio', 'greater_than?', 'StRank', 'NetCash', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_lc_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_lc_r', 'months_added', 'step_one'])
y_test_dl = test.loc[test['ProductType'] != 'LC']['Renewed?']
y_test_lc = test.loc[test['ProductType'] == 'LC']['Renewed?']

X_validate_dl = test_set_validate.loc[test_set_validate['ProductType'] != 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'credit_binned', 'GrossBalance', 'SC_ratio', 'greater_than?', 'StRank', 'NetCash', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_lc_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_lc_r', 'months_added', 'step_one'])
y_validate_dl = test_set_validate.loc[test_set_validate['ProductType'] != 'LC']['Renewed?']
X_validate_lc = test_set_validate.loc[test_set_validate['ProductType'] == 'LC'].drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'RiskTier', 'HighCredit', 'Approved_Apps_y', 'Approved_Apps_x','ProductType', 'Tier_MultipleModels', 'counter', 'prev_month', 'current_month', 'Renewed?', 'AmountFinanced', 'OwnRent', 'available_offer', 'Utilization', 'Months_left', 'avg_monthonbook_lc_c', 'avg_monthonbook_dl_c', 'avg_monthonbook_dl_r', 'Closed?', 'PaidOrCharged?', 'months_til_avg_dl_c', 'months_til_avg_lc_c', 'months_til_avg_dl_r', 'months_added', 'step_one'])
y_validate_lc = test_set_validate.loc[test_set_validate['ProductType'] == 'LC']['Renewed?']

X_train_lc.columns
X_train_dl.columns

sample_weight_data_train_lc = compute_sample_weight(class_weight="balanced", y=y_train_lc)
sample_weight_data_validate_lc = compute_sample_weight("balanced", y_validate_lc)
sample_weight_data_train_dl = compute_sample_weight("balanced", y_train_dl)
sample_weight_data_validate_dl = compute_sample_weight("balanced", y_validate_dl)


# %% ROC AOC curve XGB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

xgb_clf_lc = XGBClassifier(n_estimators=100, learning_rate=0.05,max_features=4, max_depth=5, random_state=0)
xgb_clf_lc.fit(X_train_lc, y_train_lc, sample_weight=sample_weight_data_train_lc)
predictions_lc = xgb_clf_lc.predict(X_validate_lc)
predictions2_lc = xgb_clf_lc.predict_proba(X_validate_lc)
fpr_lc, tpr_lc, thresholds_lc = roc_curve(y_validate_lc, xgb_clf_lc.predict_proba(X_validate_lc)[:,1])
roc_auc_lc = roc_auc_score(y_validate_lc, xgb_clf_lc.predict(X_validate_lc))

xgb_clf_dl = XGBClassifier(n_estimators=100, learning_rate=0.05,max_features=4, max_depth=5, random_state=0)
xgb_clf_dl.fit(X_train_dl, y_train_dl, sample_weight=sample_weight_data_train_dl)
predictions_dl = xgb_clf_dl.predict(X_validate_dl)
predictions2_dl = xgb_clf_dl.predict_proba(X_validate_dl)
fpr_dl, tpr_dl, thresholds_dl = roc_curve(y_validate_dl, xgb_clf_dl.predict_proba(X_validate_dl)[:,1])
roc_auc_dl = roc_auc_score(y_validate_dl, xgb_clf_dl.predict(X_validate_dl))

'''xgb_clf_co = XGBClassifier(n_estimators=50, learning_rate=0.05,max_features=4, max_depth=5, random_state=0)
xgb_clf_co.fit(X_train_co, y_train_co)

fpr_co_lc, tpr_co_lc, thresholds_co_lc = roc_curve(y_validate_lc, xgb_clf_co.predict_proba(X_validate_lc)[:,1])
roc_auc_co_lc = roc_auc_score(y_validate_lc, xgb_clf_co.predict(X_validate_lc))
fpr_co_dl, tpr_co_dl, thresholds_co_dl = roc_curve(y_validate_dl, xgb_clf_co.predict_proba(X_validate_dl)[:,1])
roc_auc_co_dl = roc_auc_score(y_validate_dl, xgb_clf_co.predict(X_validate_dl))'''

#X_train_lc.columns
#X_train_dl.columns

xgb_clf_dl.feature_importances_
X_validate_dl.columns
# %%
X_train_co.dtypes
X_train_co = X_train_co.loc[select_dtypes(np.float64)].astype(np.float32)
X_train_co.dtypes

X_train_co[['NetCash', 'credit_binned', 'MonthsOnBook', '30+_Indicator', 'GrossBalance', 'SC_ratio', 'LC_ratio', 'greater_than?', 'months_til_avg', 'available_offer', 'Utilization', 'PaydownPercent', 'Months_left']] = X_train_co[['NetCash', 'credit_binned', 'MonthsOnBook', '30+_Indicator', 'GrossBalance', 'SC_ratio', 'LC_ratio', 'greater_than?', 'months_til_avg', 'available_offer', 'Utilization', 'PaydownPercent', 'Months_left']].astype(np.float32)
X_train_co[['OwnRent', 'StRank', 'contract_rank']] = X_train_co[['OwnRent', 'StRank', 'contract_rank']].astype(np.int32)
X_train_co.dtypes
y_train_co = y_train_co.astype(np.float32)


X_train_co.isna().sum()

X_train_co.drop(columns=['PaydownPercent', 'Utilization'], inplace=True)

rf_lc = RandomForestClassifier(n_estimators=2, max_features=4, max_depth=5)
rf_lc.fit(X_train_lc, y_train_lc)
fpr_lc, tpr_lc, thresholds_lc = roc_curve(y_validate_lc, rf_lc.predict_proba(X_validate_lc)[:,1])

roc_auc_lc = roc_auc_score(y_validate_lc, rf_lc.predict(X_validate_lc))

rf_dl = RandomForestClassifier(n_estimators=2, max_features=4, max_depth=5)
rf_dl.fit(X_train_dl, y_train_dl)
fpr_dl, tpr_dl, thresholds_dl = roc_curve(y_validate_dl, rf_dl.predict_proba(X_validate_dl)[:,1])

roc_auc_dl = roc_auc_score(y_validate_dl, rf_dl.predict(X_validate_dl))

rf_co = RandomForestClassifier(n_estimators=2, max_features=4, max_depth=5)
rf_co.fit(X_train_co, y_train_co)
fpr_co, tpr_co, thresholds_co = roc_curve(y_test, rf_co.predict_proba(X_test)[:,1])

roc_auc_co = roc_auc_score(y_test, rf_co.predict(X_test))



# %% Logistic Regression test

lr_lc = LogisticRegression()
lr_lc.fit(X_train_lc, y_train_lc)
fpr_lc, tpr_lc, thresholds_lc = roc_curve(y_test_lc, lr_lc.predict_proba(X_test_lc)[:,1])

roc_auc_lc = roc_auc_score(y_test_lc, lr_lc.predict(X_test_lc))

lr_dl = LogisticRegression()
lr_dl.fit(X_train_dl, y_train_dl)
fpr_dl, tpr_dl, thresholds_dl = roc_curve(y_test_dl, lr_dl.predict_proba(X_test_dl)[:,1])

roc_auc_dl = roc_auc_score(y_test_dl, lr_dl.predict(X_test_dl))

lr_co = LogisticRegression()
lr_co.fit(X_train_co, y_train_co)
fpr_co, tpr_co, thresholds_co = roc_curve(y_test_co, lr_co.predict_proba(X_test_co)[:,1])

roc_auc_co = roc_auc_score(y_test_co, lr_co.predict(X_test_co))



# %% Plot Roc Auc

import matplotlib.pyplot as plt
plt.figure()
#plt.plot(fpr_lc, tpr_lc, label='LC (AUC = %0.3f)' % metrics.auc(fpr_lc,tpr_lc))
plt.plot(fpr_dl, tpr_dl, label='DL (AUC = %0.3f)' % metrics.auc(fpr_dl,tpr_dl))
#plt.plot(fpr_co_lc, tpr_co_lc, label='Combined LC (AUC = %0.3f)' % metrics.auc(fpr_co_lc,tpr_co_lc))
#plt.plot(fpr_co_dl, tpr_co_dl, label='Combined DL (AUC = %0.3f)' % metrics.auc(fpr_co_dl,tpr_co_dl))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1.0])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Predictions to Renew in 1 Month (trained april tested may 2019)')
plt.legend()
plt.show()
#plt.savefig(outputfolder / 'auc_roc_final_validation.jpeg')
#plt.savefig(pathlib.Path(outputfolder / 'Balanced_roc_auc_oct_nov_2018.png'))
#%%
predictions_dl = xgb_clf_dl.predict_proba(X_validate_dl)
Sales_rank_and_metrics(predictions_dl, X_validate_dl, y_validate_dl)
y_validate_dl.mean()
len(y_validate_dl)
predictions_lc = xgb_clf_lc.predict_proba(X_validate_lc)
Sales_rank_and_metrics(predictions_lc, X_validate_lc, y_validate_lc)
y_validate_lc.mean()
len(y_validate_lc)
predictions2_dl = xgb_clf_dl.predict(X_validate_dl)
metrics.accuracy_score()


metrics.confusion_matrix(y_validate_lc, predictions_lc, labels=[1,0])
metrics.classification_report(y_validate_lc, predictions_lc, labels=[1,0])
metrics.confusion_matrix(y_validate_dl, predictions_dl)
metrics.classification_report(y_validate_dl, predictions_dl)


predictions_dl[:,1].mean()
y_validate_dl.mean()
predictions_lc[:,1].mean()
y_validate_lc.mean()

xgb_clf_dl.feature_importances_
xgb_clf_lc.feature_importances_
X_validate_dl.columns
X_validate_lc.columns

def Sales_rank_and_metrics(predictions, X_test, y_test):
    results = {'actual':y_test, 'prob_not':predictions[:,0], 'prob_yes':predictions[:,1]}
    results = pd.DataFrame(results)
    renewals = results[results['actual'] == 1]
    len(renewals)
    len(renewals[renewals['prob_yes'] >= .2])
    len(results[results['prob_yes'] >= .2])
    x = round(len(results)/10)
    len(results)

    recreate = X_test
    recreate['prob_yes'] = predictions[:,1]
    recreate['actual'] = y_test
    actual_rr = recreate['actual'].mean()

    results.sort_values(by=['prob_yes'], ascending=False, inplace=True)
    results.reset_index(inplace=True)
    results.rename(columns={'index':'customer'}, inplace=True)
    results
    rank = []
    renewal_rate = []
    sales_rank_probs = []
    lengths = []
    ranks = []

    # %% Generate Sales Rank

    for num in range(1,11):
        if num == 10:
            y = results.iloc[x*(num-1):]
            rr = len(results.iloc[x*(num-1):])
        else:
            y = results.iloc[x*(num-1):x*num]
            rr = len(results.iloc[x*(num-1):x*num])
        z = y['actual'].sum()/len(y)
        sales_rank_probs.append(results.iloc[x*(num-1)]['prob_yes'])
        rank.append(num)
        renewal_rate.append(z)
        lengths.append(rr)
        y.reset_index(inplace=True)
        y.drop(columns=['index'], inplace=True)
        ranks.append(y)

    sales_rank_probs.append(0)
    sales_rank = {'rank':rank, 'renewal_rate':renewal_rate}
    sales_rank = pd.DataFrame(sales_rank)
    print(lengths)
    print(sales_rank_probs)

    print(sns.barplot(x='rank', y='renewal_rate', data=sales_rank))
    sales_rank.to_csv(outputfolder / 'sales_rank_3_19_1_month_DL.csv', index=False)

    num_called = []
    solicit_rate = [.8728, .8255, .7328, .6787, .5580, .5420, .5414, .5297, .5079, .4429]
    for num in range(0,10):
        num_called.append(round(solicit_rate[num]*lengths[num]))
    num_influenced = []
    penetration_per_call = []
    relative_penetration_improvement = []
    num_possible = []


    for num in range(0,10):
        num_influenced.append(ranks[num].iloc[0:num_called[num]]['actual'].sum())
        penetration_rate = ranks[num].iloc[0:num_called[num]]['actual'].sum()/num_called[num]
        penetration_per_call.append(penetration_rate)
        #relative_penetration = (penetration_rate - actual_rr)/actual_rr
        #relative_penetration_improvement.append(relative_penetration)
        num_possible.append(ranks[num]['actual'].sum())


    call_list_metrics = {'num_called':num_called, 'num_influenced':num_influenced, 'num_possible':num_possible, 'penetration_per_call':penetration_per_call}
    call_list_metrics = pd.DataFrame(call_list_metrics)
    '''print(call_list_metrics['penetration_per_call'].mean())
    #print(call_list_metrics['relative_penetration_improvement'].mean())
    #print((call_list_metrics['penetration_per_call'].mean() - actual_rr)/actual_rr)
    print(call_list_metrics['num_influenced'].sum()/call_list_metrics['num_possible'].sum())

    call_list_metrics'''
    call_list_metrics.to_csv(outputfolder / 'call_list_metrics.csv')
