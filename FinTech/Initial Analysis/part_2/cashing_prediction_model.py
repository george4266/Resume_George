import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'mailed_cashed'
marketingfile = datafolder / 'Data' / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(marketingfile, sep=',')

m = marketing
m['IssueDate'] = pd.to_datetime(m['IssueDate'])
m = m.loc[m['IssueDate']>=dt.datetime(2016,12,30)]
m['IssueMonth'] = pd.DatetimeIndex(m.IssueDate).month
len(m)
m['CashDate'].fillna(value=0, inplace=True)
m.dropna(subset=['CreditScore'], inplace=True)
m['CreditScore'].unique()
x = m['CreditScore'].str.split(pat = '-')
y = 0
new = []
for i in x:
    for z in i:
        z = int(z)
        y = y + z
    new.append(round(y/2))
    y=0
new = np.array(new)
m['CreditScore'] = new
m['cashed_or_not'] = 1
m.loc[m['CashDate']==0,'cashed_or_not'] = 0

m.drop(['State', 'Unique_BranchID', 'Cashings'], axis=1, inplace=True)
m['cashed_or_not'].mean()
m.groupby('cashed_or_not').mean()
m.groupby('RiskTier').mean()
m.groupby('State').mean()
m.groupby('Segment').mean()
m.groupby('OwnRent').mean()
m.groupby('CreditScore').mean()
for i, col in enumerate(['RiskTier', 'Segment', 'OwnRent', 'CreditScore', 'IssueMonth']):
    plt.figure(i)
    sns.catplot(x=col, y='cashed_or_not', data=m, kind='point', aspect=2)
fig = sns.catplot(x='Segment', y='cashed_or_not', data=m, kind='point', aspect=2).set(ylim = (0,1))
fig1 = sns.catplot(x='OwnRent', y='cashed_or_not', data=m, kind='point', aspect=2).set(ylim = (0,1))
fig
fig1

def Cash_or_not_classifier(m):
    howner_rank = {'R':0, 'O':1}
    m['OwnRent'] = m['OwnRent'].map(howner_rank)
    m.dropna(subset=['RiskTier'], inplace=True)
    predict = m.loc[(m['IssueDate'] > dt.datetime(2019,6,30)) & (m['IssueDate'] <= dt.datetime(2019,9,30))]
    train = m.loc[(m['IssueDate'] > dt.datetime(2017,12,31)) & (m['IssueDate'] <= dt.datetime(2019,6,30))]
    train.drop(['State', 'Cashings', 'Segment', 'Unique_BranchID', 'CashDate', 'IssueDate'], axis=1, inplace=True)
    predict.drop(['State', 'Cashings', 'Segment', 'Unique_BranchID', 'CashDate', 'IssueDate'], axis=1, inplace=True)


    train_features = train.drop('cashed_or_not', axis=1)
    train_labels = train['cashed_or_not']
    test_features = predict.drop('cashed_or_not', axis=1)
    test_labels  = predict['cashed_or_not']
    len(train_features)
    len(test_features)
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.3, random_state=42)
    '''rf = RandomForestClassifier()

    scores = cross_val_score(rf, X_train, y_train.values.ravel(), cv=5)
    scores
    parameters = {
        'n_estimators': [5, 50, 100],
        'max_depth': [2, 10, 20, None]
    }

    cv = GridSearchCV(rf, parameters, cv=5)
    cv.fit(X_train, y_train.values.ravel().ravel())
    cv.cv_results_
    print_results(cv)'''
    rf = RandomForestClassifier(n_estimators=100, max_depth=20)
    scores = cross_val_score(rf, X_train, y_train.values.ravel(), cv=5)
    scores
    rf.fit(X_train, y_train)
    scores = rf.score(X_val, y_val)
    scores
    predictions = rf.predict(test_features)
    metrics.accuracy_score(test_labels, predictions)
    metrics.classification_report(test_labels, predictions)
    rf.feature_importances_

    test_features
    return predictions
    '''lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in lr_list:
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.075,max_features=2, max_depth=20, random_state=0)
        gb_clf.fit(train_features, train_labels)

        print("Learning rate: ", learning_rate)
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(train_features, train_labels)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(test_features, test_labels)))

    xgb_clf = XGBClassifier(n_estimators=20, max_depth=10, max_features=2, learning_rate=.05)
    xgb_clf.fit(train_features, train_labels)

    score = xgb_clf.score(test_features, test_labels)
    predictions = xgb_clf.predict(test_features)
    score
    len(predictions)'''

def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

Cash_or_not_classifier(m)




dates = m
dates['IssueDate'] = pd.to_datetime(dates['IssueDate'])
dates['CashDate'] = pd.to_datetime(dates['CashDate'])
dates['IssueMonth'] = pd.DatetimeIndex(dates.IssueDate).month
dates['CashMonth'] = pd.DatetimeIndex(dates.CashDate).month
dates['IssueYear'] = pd.DatetimeIndex(dates.IssueDate).year
dates['CashYear'] = pd.DatetimeIndex(dates.CashDate).year
dates['IssueDay'] = pd.DatetimeIndex(dates.IssueDate).day
dates['CashDay'] = pd.DatetimeIndex(dates.CashDate).day
dates['daystocashing'] = (dates['CashYear']-dates['IssueYear'])*365 + 31*(dates['CashMonth']-dates['IssueMonth']) + dates['CashDay'] - dates['IssueDay']
len(dates)
howner_rank = {'R':0, 'O':1}
dates['OwnRent'] = dates['OwnRent'].map(howner_rank)
dates.dropna(subset=['RiskTier'], inplace=True)
train = dates.loc[dates['IssueDate'] <= dt.datetime(2018,12,31)]
test = dates.loc[dates['IssueDate'] > dt.datetime(2018,12,31)]
x = test['CashYear'] - test['IssueYear']

for i, col in enumerate(['RiskTier', 'OwnRent', 'State', 'IssueMonth', 'IssueYear', 'Cashings', 'Mailings', 'Segment']):
    plt.figure(i)
    sns.catplot(x=col, y='monthstocashing', data=dates, kind='point', aspect=2)

train.drop(['IssueDate', 'CashDate', 'CashMonth', 'IssueYear', 'State', 'Unique_BranchID', 'Cashings', 'Segment', 'CashDay', 'CashYear', 'cashed_or_not'], axis=1, inplace=True)
test.drop(['IssueDate', 'CashDate', 'CashMonth', 'IssueYear', 'State', 'Unique_BranchID', 'Cashings', 'Segment', 'CashDay', 'CashYear', 'cashed_or_not'], axis=1, inplace=True)
train_features = train.drop('daystocashing', axis=1)
train_labels = train['daystocashing']
test_features = test.drop('daystocashing', axis=1)
test_labels = test['daystocashing']

'''parameters = {
    'n_estimators': [5, 50, 100],
    'max_depth': [2, 10, 20, None]
}
rf = RandomForestRegressor()
cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(train_features, train_labels.values.ravel())
cv.cv_results_
print_results(cv)'''

rf = RandomForestRegressor(n_estimators=100, max_depth=20)
rf.fit(train_features, train_labels)
rf.feature_importances_
test_features
prediction_days = rf.predict(test_features)
prediction_days = prediction_days.round()
metrics.accuracy_score(test_labels, prediction_days)
#metrics.classification_report(test_labels, prediction_days)
prediction_days




def create_budge(m, predictions):
    budget_data = m[['CheckAmount', 'IssueDate', 'cashed_or_not', 'CashDate']]
    budget_data = budget_data.loc[(budget_data['IssueDate'] > dt.datetime(2019,6,30)) & (budget_data['IssueDate'] <= dt.datetime(2019,9,30))]
    budget_data['cashed_prediction'] = predictions
    budget_data['prediction_cash_date'] = budget_data['IssueDate']
    budget_data['CashDate'] = budget_data['IssueDate']
    budget_predictions = budget_data[budget_data['cashed_prediction'] == 1]
    budget_predictions = budget_predictions.groupby(budget_predictions.prediction_cash_date.dt.to_period('M'))['CheckAmount'].sum().to_frame()
    budget_actual = budget_data[budget_data['cashed_or_not'] == 1]
    budget_actual = budget_actual.groupby([budget_actual.CashDate.dt.to_period('M')])['CheckAmount'].sum().to_frame()
    x = budget_predictions['CheckAmount'].sum()
    y = budget_actual['CheckAmount'].sum()
    (x-y)/y
    x = (budget_predictions['CheckAmount'] - budget_actual['CheckAmount'])/budget_actual['CheckAmount']
    x
    x.abs().mean()
