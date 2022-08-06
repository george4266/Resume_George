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
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
pd.options.mode.chained_assignment = None
sns.set()

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
branchfile = datafolder / 'VT_Branches_01072020.csv'
outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'

# %% data clean
origin = pd.read_csv(origination_file, sep=',', low_memory=False)
origin2 = origin
origin2['counter'] = 1
origin2['BookDate'] = pd.to_datetime(origin2['BookDate'])
stuff = origin2[origin2['BookDate'] < dt.datetime(2019,3,1)]
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

'''origin.loc[(origin['CreditScore'] >= 300) & (origin['CreditScore'] <= 566), 'credit_binned'] = 1
origin.loc[(origin['CreditScore'] >= 567) & (origin['CreditScore'] <= 602), 'credit_binned'] = 2
origin.loc[(origin['CreditScore'] >= 603) & (origin['CreditScore'] <= 639), 'credit_binned'] = 3
origin.loc[(origin['CreditScore'] >= 640) & (origin['CreditScore'] <= 696), 'credit_binned'] = 4
origin.loc[(origin['CreditScore'] >= 697) & (origin['CreditScore'] <= 800), 'credit_binned'] = 5'''

origin.drop(columns=['OwnRent', 'State', 'AmountFinanced', 'TotalNote', 'NetCash', 'CashToCustomer', 'Segment', 'IP_Unique_ContractID', 'RegularPayment', 'Unique_ApplicationID'], inplace=True)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))
perf.drop(columns=['30+_Indicator'], inplace=True)
origin['indicator'] = 1
origin['BookDate'] = pd.to_datetime(origin['BookDate'])


# %% Feature generation pt. 1
contract_rank = origin.loc[origin['BookDate'] < dt.datetime(2019,3,1)].groupby(origin.Unique_CustomerID)['indicator'].sum().reset_index()
contract_rank.rename(columns={'indicator':'contract_rank'}, inplace=True)
combined = origin.merge(perf, on='Unique_ContractID', how='left')
combined.drop(columns=['Unique_CustomerID_y', 'Unique_BranchID_y'], inplace=True)
combined.rename(columns={'Unique_CustomerID_x':'Unique_CustomerID','Unique_BranchID_x':'Unique_BranchID'}, inplace=True)
combined[['MonthsOnBook', 'Solicitation_Memos']] = combined[['MonthsOnBook', 'Solicitation_Memos']].fillna(value=0,  axis=1)
combined['CurrentMonth'] = combined[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)

approved_rank_ratio = combined.loc[combined['CurrentMonth'] < dt.datetime(2019,3,1)]
approved_rank_ratio = approved_rank_ratio[['Unique_CustomerID', 'Approved_Apps']]
approved_rank_ratio['Approved_Apps'].fillna(value=0, inplace=True)
approved_rank_ratio = approved_rank_ratio.groupby(approved_rank_ratio.Unique_CustomerID)['Approved_Apps'].sum().reset_index()

avg_monthonbook = combined.loc[combined['CurrentMonth'] < dt.datetime(2019,3,1)]
avg_monthonbook = avg_monthonbook[['Unique_CustomerID', 'ProcessStatus', 'MonthsOnBook']]
avg_monthonbook = avg_monthonbook.loc[avg_monthonbook['ProcessStatus'] == 'Renewed'].groupby('Unique_CustomerID')['MonthsOnBook'].mean().reset_index()
ind_avg_monthonbook = avg_monthonbook['MonthsOnBook'].mean()
avg_monthonbook.rename(columns={'MonthsOnBook':'Avg_MonthsOnBook'}, inplace=True)
avg_monthonbook['Avg_MonthsOnBook'].fillna(value=ind_avg_monthonbook, inplace=True)

solicit_contact = combined[['Contacted_Memos', 'CurrentMonth', 'Solicitation_Memos', 'Unique_CustomerID']]
solicit_contact = solicit_contact[solicit_contact['CurrentMonth'] < dt.datetime(2019,3,1)]
contacts = solicit_contact.groupby('Unique_CustomerID')['Contacted_Memos'].sum().reset_index()
solicits = solicit_contact.groupby('Unique_CustomerID')['Solicitation_Memos'].sum().reset_index()
solicit_contact = solicits.merge(contacts, on='Unique_CustomerID', how='left')
solicit_contact['SC_ratio'] = 0
solicit_contact.loc[(solicit_contact['Contacted_Memos'] !=0) & (solicit_contact['Solicitation_Memos'] != 0), 'SC_ratio'] = solicit_contact['Contacted_Memos']/solicit_contact['Solicitation_Memos']
solicit_contact = solicit_contact[['Unique_CustomerID', 'SC_ratio']]

no_rank_test = combined.loc[(combined['CurrentMonth'] >= dt.datetime(2019,3,1)) & (combined['CurrentMonth'] < dt.datetime(2019,4,1))]

# %% feature finalizing
test_set = no_rank_test.merge(contract_rank, on='Unique_CustomerID', how='left')
test_set = test_set.merge(approved_rank_ratio, on='Unique_CustomerID', how='left')
test_set = test_set.merge(avg_monthonbook, on='Unique_CustomerID', how='left')
test_set = test_set.merge(solicit_contact, on='Unique_CustomerID', how='left')
test_set = test_set.merge(origs_ratio, on='Unique_CustomerID', how='left')
test_set['contract_rank'].fillna(value=0, inplace=True)
test_set['approved_vs_rank'] = test_set['Approved_Apps_y'] - test_set['contract_rank']
test_set.dropna(subset=['RiskTier'], inplace=True)
test_set['HighCredit'] = 0
test_set.loc[(test_set['RiskTier'] == 1) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 10000
test_set.loc[(test_set['RiskTier'] == 2) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 7000
test_set.loc[(test_set['RiskTier'] == 3) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 4500
test_set.loc[(test_set['RiskTier'] == 4) & (test_set['ProductType'] != 'Auto'), 'HighCredit'] = 3000
test_set.loc[(test_set['RiskTier'] == 1) & (test_set['ProductType'] == 'Auto'), 'HighCredit']= 12500
test_set.loc[(test_set['RiskTier'] == 2) & (test_set['ProductType'] == 'Auto'), 'HighCredit'] = 7700
test_set.loc[(test_set['RiskTier'] == 3) & (test_set['ProductType'] == 'Auto'), 'HighCredit'] = 4050
test_set.loc[(test_set['RiskTier'] == 4) & (test_set['ProductType'] == 'Auto'), 'HighCredit'] = 2250
test_set['NetReceivable'].fillna(value=0, inplace=True)
test_set['Avg_MonthsOnBook'].fillna(value=ind_avg_monthonbook, inplace=True)
test_set['months_til_avg'] = test_set['MonthsOnBook'] - test_set['Avg_MonthsOnBook']
test_set['HighCredit'].mean()
test_set['NetReceivable'].mean()
test_set['available_offer'] = 0
test_set.loc[(test_set['RiskTier'] != 5) & (test_set['HighCredit'] > test_set['NetReceivable']), 'available_offer'] = test_set['HighCredit'] - test_set['NetReceivable']
labels = combined.loc[(combined['ProcessStatus'] == 'Renewed') & (combined['CurrentMonth'] < dt.datetime(2019,6,1)) & (combined['CurrentMonth'] >= dt.datetime(2019,3,1))]
labels['Renewed?'] = 1
labels = labels[['Renewed?', 'Unique_CustomerID']]
test_set = test_set.merge(labels, on='Unique_CustomerID', how='left')
test_set['Months_left'] = test_set['Term'] - test_set['MonthsOnBook']
test_set[['Tier_MultipleModels', 'Renewed?', 'Declined_Apps', 'Solicitation_Memos', 'approved_vs_rank']] = test_set[['Tier_MultipleModels', 'Renewed?', 'Declined_Apps', 'Solicitation_Memos', 'approved_vs_rank']].fillna(value=0)
test_set = test_set[test_set['ProductType'] != 'LC']
test_set.dropna(subset=['GrossBalance', 'SC_ratio', 'LC_ratio'], inplace=True)
#test_set.drop(columns=['Unique_CustomerID', 'Unique_BranchID', 'Unique_ContractID', 'BookDate', 'Term', 'TotalOldBalance', 'Rescored_Tier_2017Model', 'Rescored_Tier_2018Model', 'indicator', 'ProcessStatus', 'NetReceivable', 'CurrentMonth', 'Solicitation_Memos', 'Contacted_Memos', 'Approved_Apps_y', 'Approved_Apps_x', 'Declined_Apps', 'Tier_MultipleModels', 'HighCredit',\
#    'ProductType', 'approved_vs_rank', 'RiskTier', 'Avg_MonthsOnBook', 'counter', 'credit_binned', 'StRank', 'SC_ratio', 'contract_rank', 'Months_left'], inplace=True)
#len(test_set)

#%% Adding branch info
branches = pd.read_csv(branchfile, sep=',')
branches = branches.drop(columns=['BranchOpenDate','Month','MonthsOpen','NumActiveEmployees','State','BrCity','BrCounty','BrZip'])
test_set = test_set.merge(branches, how = 'left', on = 'Unique_BranchID')
test_set = test_set.drop_duplicates()

#%% Creating feature plots
test_set = test_set[['CreditScore','MonthsOnBook','GrossBalance','Avg_MonthsOnBook','LC_ratio','months_til_avg','available_offer','Renewed?','Population Density (per sq. mile), 2019','% Employment, Unemployed, 2019','% Housing, Renter Occupied, 2019']]
test_set

test_set

ax = sns.violinplot(x='Renewed?',y='Nondepository credit intermediation (# Businesses) [NAICS 5222], 2019',data=test_set)
ax = sns.boxplot(x='Renewed?',y='Nondepository credit intermediation (# Businesses) [NAICS 5222], 2019',data=test_set)

#%% extra test set
reserve = test_set

#%% Removing monthsonbook 0
test_set = test_set.loc[test_set.MonthsOnBook > 0]

# ////////////////////////// MACHINE LEARNING //////////////////////////////////
# %% create sets
test_set.dropna(inplace=True)
features = test_set.drop(columns=['Renewed?'])
labels = test_set['Renewed?']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

features.columns



# %% Random Forest
'''rf = RandomForestClassifier(n_estimators=20, max_depth=2)
scores = cross_val_score(rf, X_train, y_train.values.ravel(), cv=5)
scores
parameters = {
    'n_estimators': [5, 50, 100],
    'max_depth': [2, 10, 20, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(X_train, y_train.values.ravel())
cv.cv_results_
print_results(cv)'''

rf = RandomForestClassifier(n_estimators=100, max_depth=20)
rf.fit(X_train, y_train)
scores = rf.score(X_val, y_val)
scores
predictions = rf.predict_proba(X_test)
predictions2 = rf.predict(X_test)
metrics.accuracy_score(y_test, predictions2)
metrics.classification_report(y_test, predictions2, labels=[1,0])
rf.feature_importances_
X_test.columns
metrics.confusion_matrix(y_test, predictions2, labels=[1,0])
'''x,y,z = metrics.precision_recall_curve(y_test, predictions[:,1])
len(x)
len(y)
len(z)'''

results = {'actual':y_test, 'prob_not':predictions[:,0], 'prob_yes':predictions[:,1]}
results = pd.DataFrame(results)
renewals = results[results['actual'] == 1]
len(renewals)
len(renewals[renewals['prob_yes'] >= .2])
len(results[results['prob_yes'] >= .2])
len(results)
x = round(len(results)/10)
results.sort_values(by=['prob_yes'], ascending=False, inplace=True)
results.reset_index(inplace=True)
results.rename(columns={'index':'customer'}, inplace=True)
results
rank = []
renewal_rate = []
sales_rank_probs = []

for num in range(1,11):
    if num == 10:
        y = results.iloc[x*(num-1):]
    else:
        y = results.iloc[x*(num-1):x*num]
    z = y['actual'].sum()/len(y)
    sales_rank_probs.append(results.iloc[x*(num-1)]['prob_yes'])
    rank.append(num)
    renewal_rate.append(z)
sales_rank_probs.append(0)
sales_rank = {'rank':rank, 'renewal_rate':renewal_rate}
sales_rank = pd.DataFrame(sales_rank)
sales_rank
sns.barplot(x='rank', y='renewal_rate', data=sales_rank)
#sales_rank.to_csv(outputfolder / 'sales_rank.csv', index=False)
len(X_test)



# %% Gradient Boosting

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate,max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (train): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))




def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
