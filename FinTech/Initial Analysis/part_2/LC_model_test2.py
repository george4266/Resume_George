import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

datafolder = pathlib.Path.cwd().parent.parent.parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'mailed_cashed'
marketingfile = datafolder / 'Data' / 'VT_Marketing_11012019.txt'
marketing = pd.read_csv(marketingfile, sep=',')

m = marketing
m['IssueDate'] = pd.to_datetime(m['IssueDate'])
m = m.loc[m['IssueDate']>=dt.datetime(2016,12,30)]
m['IssueMonth'] = pd.DatetimeIndex(m.IssueDate).month
m['cashdummy'] = m['CashDate']
m['cashdummy'].fillna(value=0, inplace=True)
m.dropna(subset=['CreditScore'], inplace=True)
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
m.loc[m['cashdummy']==0,'cashed_or_not'] = 0
cash = m[m['cashed_or_not'] == 1]
cash
cash['CashDate'] = pd.to_datetime(cash['CashDate'])
cash['difference'] = cash['CashDate'] - cash['IssueDate']
cash['difference'].mean()
for i in
for i, col in enumerate(['CheckAmount', 'Mailings']):
    plt.figure(i)
    sns.boxplot(x='cashed_or_not', y=col, data=m)

m['Mailings'].max()
m[(m['Mailings'] >=10) & (m['cashed_or_not'] == 1)]['cashed_or_not'].count()
m[m['Mailings'] >= 10]['cashed_or_not'].count()



howner_rank = {'R':0, 'O':1}
m['OwnRent'] = m['OwnRent'].map(howner_rank)
m.dropna(subset=['RiskTier'], inplace=True)
branch1 = m[m['Unique_BranchID'] == 1]
predict = m.loc[(m['IssueDate'] > dt.datetime(2019,6,30)) & (m['IssueDate'] <= dt.datetime(2019,9,30))]
train = m.loc[(m['IssueDate'] > dt.datetime(2017,12,31)) & (m['IssueDate'] <= dt.datetime(2019,6,30))]
train.drop(['State', 'Cashings', 'Segment', 'CashDate', 'IssueDate', 'cashdummy', 'Mailings'], axis=1, inplace=True)
predict.drop(['State', 'Cashings', 'Segment', 'CashDate', 'IssueDate', 'cashdummy', 'Mailings'], axis=1, inplace=True)


train_features = train.drop('cashed_or_not', axis=1)
train_labels = train['cashed_or_not']
test_features = predict.drop('cashed_or_not', axis=1)
test_labels  = predict['cashed_or_not']

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.3, random_state=42)



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
test_features.columns
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


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
