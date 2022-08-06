# %% Imports, setup
import pathlib
import datetime as dt, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from scipy import stats

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

# %% Data load
datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Kaitlyn' / 'Outputs'
convfile = datafolder / 'encoded_conversions.csv'

conv = pd.read_csv(convfile)
conv.head()
#conv.groupby('State')['Converted?'].mean()

# %% One-hot encoding
#y = conv.drop('State', axis = 1)
#onehot = ce.OneHotEncoder(cols = ['State'])
#conv = onehot.fit_transform(conv, y)
#conv.loc[conv['RiskTier']==0.5,'RiskTier'] = 0
conv = conv.drop('State', axis=1)

#Verifying factors are not too stongly correlated
stats.spearmanr(conv['RiskTier'], conv['CreditScore'])
#sns.distplot(conv['TotalNote'])
#sns.distplot(conv['RegularPayment'])
stats.pearsonr(conv['RegularPayment'], conv['CreditScore'])

# %% Preview conversions table again
conv = conv.loc[conv.BookYear < 2019]
conv = conv.loc[~((conv.BookYear == 2018)&(conv.BookQtr >= 3))]
conv

# %% Creating logistic regression model
#Start by splitting the non-null dataset into test and train sets
features = conv.drop(columns='Converted?')
labels = conv['Converted?']
x_train, x_test, y_train, y_test = model_selection.train_test_split(features,labels,test_size=0.2)
x_test, x_valid, y_test, y_valid = model_selection.train_test_split(x_test,y_test,test_size=0.5)

logreg = LogisticRegression()

parameters = {'C':[0.001,0.01,0.1,1,10,100,1000]}
cv = model_selection.GridSearchCV(logreg,parameters,cv=5,verbose=True,n_jobs=4)
cv.fit(x_train,y_train)
cv.cv_results_
print(cv.best_estimator_)

logreg = cv.best_estimator_
y_pred = logreg.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))

logreg.coef_

# %% Creating random forest model
#Start by splitting the non-null dataset into test and train sets
features = conv.drop(columns='Converted?')
labels = conv['Converted?']
x_train, x_test, y_train, y_test = model_selection.train_test_split(features,labels,test_size=0.2)
x_test, x_valid, y_test, y_valid = model_selection.train_test_split(x_test,y_test,test_size=0.5)

randf = RandomForestClassifier()

parameters = {'n_estimators':[10,50],'max_features':[3,7],'max_depth':[5,10,50]}
cv = model_selection.GridSearchCV(randf,parameters,cv=5,verbose=True,n_jobs=4)
cv.fit(x_train,y_train)
cv.cv_results_
print(cv.best_estimator_)

randf = cv.best_estimator_
y_pred = logreg.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
randf.feature_importances_
