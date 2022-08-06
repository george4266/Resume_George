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
test_set = pd.read_csv(data, sep=',', low_memory=False)
test_set.drop(columns=['CreditScore'], inplace=True)'''

data = datafolder / 'test_set_march_1_month_DL_full.csv'
test_set = pd.read_csv(data, sep=',', low_memory=False)
test_set.drop(columns=['CreditScore'], inplace=True)




'''test_set_2_month = datafolder / 'test_set_2_month.csv'
test_set_2_month = pd.read_csv(test_set_2_month, sep=',', low_memory=False)
test_set_2_month.drop(columns=['CreditScore'], inplace=True)

data = datafolder / 'test_set_march_1_month_LC.csv'
test_set_1_month_lc = pd.read_csv(data, sep=',', low_memory=False)
test_set_1_month_lc.drop(columns=['CreditScore', 'LC_ratio'], inplace=True)

data = datafolder / 'test_set_march_1_month_combined.csv'
test_set_1_month_combined = pd.read_csv(data, sep=',', low_memory=False)
test_set_1_month_combined.drop(columns=['CreditScore'], inplace=True)
test_set_1_month_combined.columns'''



#test_set['OwnRent'] = test_set['OwnRent'].map({'O':0, 'R':1})
test_set.drop(columns=['StRank', 'SC_ratio', 'credit_binned', 'StRank'], inplace=True)

#%%

features = test_set.drop(columns=['Renewed?'])
labels = test_set['Renewed?']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#test_set['greater_than?'].describe()



# %% Random Forest

rf = RandomForestClassifier(n_estimators=2, max_depth=5)
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


# %% Prep Sales Rank

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
lengths
sales_rank_probs

print(sns.barplot(x='rank', y='renewal_rate', data=sales_rank))
sales_rank.to_csv(outputfolder / 'sales_rank_3_19_1_month_DL.csv', index=False)
#sns.violinplot(x=recreate.loc[recreate['GrossBalance'] <= 12500]['GrossBalance'], y=recreate.loc[recreate['GrossBalance'] <= 12500]['actual'])
#sns.violinplot(x=recreate.loc[recreate['actual'] == 1]['MonthsOnBook'])
#sns.violinplot(x=recreate.loc[recreate['actual'] == 1]['months_til_avg'])

# %% Call List Metrics

num_called_DL = [2902, 2769, 2435, 2268, 1868, 1801, 1801, 1768, 1701, 1468]
num_called_LC = [1454, 1387, 1220, 1136, 936, 902, 902, 886, 852, 735]
num_influenced = []
penetration_per_call = []
relative_penetration_improvement = []
num_possible = []

for num in range(0,10):
    num_influenced.append(ranks[num].iloc[0:num_called_LC[num]]['actual'].sum())
    penetration_rate = ranks[num].iloc[0:num_called_LC[num]]['actual'].sum()/num_called_LC[num]
    penetration_per_call.append(penetration_rate)
    relative_penetration = (penetration_rate - actual_rr)/actual_rr
    relative_penetration_improvement.append(relative_penetration)
    num_possible.append(ranks[num]['actual'].sum())


call_list_metrics = {'num_called':num_called, 'num_influenced':num_influenced, 'num_possible':num_possible, 'penetration_per_call':penetration_per_call, 'relative_penetration_improvement':relative_penetration_improvement}
call_list_metrics = pd.DataFrame(call_list_metrics)
print(call_list_metrics['penetration_per_call'].mean())
print(call_list_metrics['relative_penetration_improvement'].mean())
print(call_list_metrics['num_influenced'].sum()/call_list_metrics['num_possible'].sum())

call_list_metrics
call_list_metrics.to_csv(outputfolder / 'call_list_metrics.csv')

# %% Gradient Boosting

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=2, learning_rate=learning_rate,max_features=5, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (train): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))
    #xgb_clf = XGBClassifier(n_estimators=2, learning_rate=learning_rate,max_features=2, max_depth=5, random_state=0)
    #xgb_clf.fit(X_train, y_train)
    #predictions = xgb_clf.predict_proba(X_test)
    #analysis_and_SaleRank(predictions, X_test, y_test)
    #tr_te_sp(test_set)

gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=.05,max_features=4, max_depth=5, random_state=0)
gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict_proba(X_test)

xgb_clf = XGBClassifier(n_estimators=20, learning_rate=0.1,max_features=4, max_depth=5, random_state=0)
xgb_clf.fit(X_train, y_train)
predictions = xgb_clf.predict_proba(X_test)


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

def opimize_hyperparameters(X_train, y_train):
    rf = RandomForestClassifier()
    scores = cross_val_score(rf, X_train, y_train.values.ravel(), cv=5)
    scores
    parameters = {
        'n_estimators': [5, 50, 100],
        'max_depth': [2, 10, 20, None]
    }

    cv = GridSearchCV(rf, parameters, cv=5)
    cv.fit(X_train, y_train.values.ravel())
    cv.cv_results_
    print_results(cv)
