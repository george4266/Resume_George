# %% import and files
import pandas as pd
import numpy as np
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from xgboost import XGBClassifier


datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'

data = datafolder / 'feature_gen_dists_mar_2019.csv'
test_set = pd.read_csv(data, sep=',', low_memory=False)

test_set = test_set.loc[test_set.Utilization >= 0]

# %%
#test_set.drop(columns=['StRank', 'credit_binned'], inplace=True)
test_set.columns
test_set.dtypes
test_set.drop(columns=['RiskTier','Rescored_Tier_2018Model','Rescored_Tier_2017Model','counter','Approved_Apps_x','Approved_Apps_y','prev_month','current_month','HighCredit','indicator','BookDate','ProcessStatus','CurrentMonth','Unique_CustomerID','Unique_BranchID','Unique_ContractID','step_one','ProductType','months_added','OwnRent','GrossBalance','Declined_Apps'],inplace=True)

test_set.drop(columns=['greater_than?','Renewed?','Tier_MultipleModels','Term','avg_monthonbook_lc_c','avg_monthonbook_dl_c','Contacted_Memos','contract_rank','avg_monthonbook_dl_r','avg_monthonbook_lc_r','LC_ratio','credit_binned','StRank','NetCash','MonthsOnBook','30+_Indicator','months_til_avg_dl_r','months_til_avg_lc_r'],inplace=True)

#test_set.drop(columns=['months_til_avg_dl_c','months_til_avg_lc_c','SC_ratio','TotalOldBalance'],inplace=True)

#%%

#features = test_set.drop(columns=['Closed?'])
#labels = test_set['Closed?']

features = test_set.drop(columns=['PaidOrCharged?','Closed?'])
labels = test_set['PaidOrCharged?']


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# %%
#test_set['greater_than?'].describe()

'''
# %% Random Forest
rf = RandomForestClassifier()
parameters = {
        'n_estimators':[20,50,100],
        'max_depth': [10,20]
        #'learning_rate':[2, 5, 10],
        #'max_features':[2, 5, 10, None]
        }

cv = GridSearchCV(rf, parameters, cv=3, scoring='f1_macro',verbose=1)
cv.fit(X_train, y_train.values.ravel(), n_jobs=3)
cv.cv_results_
cv.best_estimator_
'''
test_set['PaidOrCharged?'].mean()
predictions2.mean()

# %%
rf = RandomForestClassifier(n_estimators=100, max_depth=20)
rf.fit(X_train, y_train)
scores = rf.score(X_val, y_val)
scores
predictions = rf.predict_proba(X_test)
predictions2 = rf.predict(X_test)
print(metrics.accuracy_score(y_test, predictions2))
print(metrics.classification_report(y_test, predictions2))
df = pd.DataFrame()
df['score'] = rf.feature_importances_
df['feature'] = X_test.columns
print(df.sort_values('score',ascending=False))
print(metrics.confusion_matrix(y_test, predictions2))

df.sort_values('score',ascending=False).to_csv('temp.csv')

# %%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, rf.predict_proba(X_val)[:,1])
roc_auc = roc_auc_score(y_val, rf.predict(X_val))

# %%
plt.figure()
plt.plot(fpr, tpr, label='Attrition (AUC = %0.3f)' % metrics.auc(fpr,tpr))
#plt.plot(fpr_co, tpr_co, label='Combined LC (AUC = %0.3f)' % metrics.auc(fpr_co,tpr_co))
#plt.plot(fpr_co, tpr_co, label='Combined DL (AUC = %0.3f)' % metrics.auc(fpr_co,tpr_co))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1.0])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Predictions to End in 1 Month')
plt.legend()
plt.show()
plt.savefig(outputfolder / 'auc_roc_attrition.png')


# %% Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=.05,max_features=4, max_depth=5, random_state=0)
gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict_proba(X_test)

# %%

xgb_clf = XGBClassifier(n_estimators=20, learning_rate=0.1,max_features=4, max_depth=5, random_state=0)
xgb_clf.fit(X_train, y_train)
predictions = xgb_clf.predict_proba(X_test)
predictions2 = xgb_clf.predict(X_test)

metrics.accuracy_score(y_test, predictions2)
metrics.classification_report(y_test, predictions2, labels=[1,0])
xgb_clf.feature_importances_
X_test.columns
