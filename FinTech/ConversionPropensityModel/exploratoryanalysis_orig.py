# %% Imports
import pathlib, sklearn
import datetime as dt, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.neighbors import KNeighborsClassifier

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

# %% Data read
datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Kaitlyn' / 'Outputs'
origfile = datafolder / 'VT_Originations_11012019.txt'
orig = pd.read_csv(origfile, sep=',', low_memory=False)
livechecks = orig[['Unique_ContractID','BookDate','ProductType']].loc[orig.ProductType == 'LC']
conversions = livechecks.merge(orig[['Unique_ContractID','BookDate','ProductType','IP_Unique_ContractID']],how='left',left_on='Unique_ContractID',right_on='IP_Unique_ContractID')
conversions['Converted?'] = 0 #Helper calculated column
conversions.loc[~pd.isnull(conversions.Unique_ContractID_y), 'Converted?'] = 1
conversions = conversions[['Unique_ContractID_x','Converted?']].rename(columns={'Unique_ContractID_x':'Unique_ContractID'})
conversions = conversions.merge(orig, how='left',on='Unique_ContractID')
conversions.BookDate = conversions.BookDate.apply(pd.to_datetime)
conversions['BookYear'] = conversions.BookDate.dt.year
conversions['BookQtr'] = conversions.BookDate.dt.quarter
conversions.head()

# %% Continuous features!!!
conversions_cont = conversions[['CreditScore','AmountFinanced','CashToCustomer','NetCash','TotalNote','TotalOldBalance','RegularPayment','Converted?']]
conversions_cont.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

# %% Need to drop some values that don't seem to make sense

#Credit scores above 850?
conversions_cont.loc[conversions_cont.CreditScore > 850]
conversions_cont.loc[conversions_cont.CreditScore < 300]
conversions = conversions.loc[(conversions.CreditScore <= 850)&(conversions.CreditScore >= 300)]

# %% Feature plots
for i in conversions_cont.columns:
    sns.distplot(conversions_cont[i])
    plt.show()
# Amount financed, cash to customer, net cash, and total note follow pretty much the exact same distributions, as expected
# Everything but net cash can be dropped altogether

# %% Total old balance?
# Total old balance could potentially be swapped to a binary variable. Do they have an old balance or not?
# Because totaloldbalance has a lot of zero values, which could be interesting?
conversions_cont.loc[conversions_cont.TotalOldBalance > 0]
conversions_cont.groupby((conversions_cont.TotalOldBalance > 0)).mean()
# Nevermind, it doesn't really seem to affect conversions. Drop the whole column.
conversions = conversions.drop(columns=['AmountFinanced','CashToCustomer','NetCash','TotalOldBalance'])

# %% Categorical features!!!
conversions_cat = conversions[['Unique_BranchID','BookDate','BookYear','RiskTier','OwnRent','State','Term','Rescored_Tier_2018Model','Rescored_Tier_2017Model','Converted?']]
conversions_cat.info()

# %% Starting with consolidating risk tiers
conversions.loc[(~pd.isnull(conversions.Rescored_Tier_2018Model))&(conversions.BookDate.dt.year == 2018),'RiskTier'] = conversions.Rescored_Tier_2018Model
conversions.loc[(~pd.isnull(conversions.Rescored_Tier_2017Model))&(conversions.BookDate.dt.year <= 2017),'RiskTier'] = conversions.Rescored_Tier_2017Model
conversions.loc[(pd.isnull(conversions.RiskTier)&pd.isnull(conversions.Rescored_Tier_2018Model)&~pd.isnull(conversions.Rescored_Tier_2017Model)),'RiskTier'] = conversions.Rescored_Tier_2017Model
# A bunch of risk tier fields are missing. Should we drop null values or fill them with averages? maybe use k-NNs?
conversions.loc[(pd.isnull(conversions.RiskTier)&pd.isnull(conversions.Rescored_Tier_2018Model)&pd.isnull(conversions.Rescored_Tier_2017Model))]
conversions_cat.groupby(pd.isnull(conversions_cat.RiskTier)).mean()

# %% Trying out using KNN to fill in risk tier data
'''
#Start by splitting the non-null dataset into test and train sets
knn_features = conversions.loc[~pd.isnull(conversions.RiskTier),['CreditScore','OwnRent','State','Term','TotalNote','BookYear']]
knn_labels = conversions.loc[~pd.isnull(conversions.RiskTier),'RiskTier']

labelencoder = preprocessing.LabelEncoder()
knn_features['State'] = labelencoder.fit_transform(knn_features['State'])
knn_features['OwnRent'] = labelencoder.fit_transform(knn_features['OwnRent'])
knn_labels = labelencoder.fit_transform(knn_labels)
knn_features
x_train, x_test, y_train, y_test = model_selection.train_test_split(knn_features,knn_labels,test_size=0.2)

knn = KNeighborsClassifier()
parameters = {'n_neighbors':[5,7,21,31],'weights':'distance'}
cv = model_selection.GridSearchCV(knn,parameters,cv=5,verbose=True,n_jobs=4)
cv.fit(x_train,y_train)
cv.cv_results_
cv.best_estimator_
# Best result was with 27 nearest neighbors
knn = KNeighborsClassifier(n_neighbors = 27)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
'''

# %% KNN performance is horrendous if I use the replaced values of risk tier, so i'm just going to have to drop the missing risk tier rows or not replace the values before filling them with KNN
conversions = conversions.loc[~pd.isnull(conversions.RiskTier)]
conversions = conversions.drop(columns=['Rescored_Tier_2017Model','Rescored_Tier_2018Model','Segment','Unique_ApplicationID','ProductType','IP_Unique_ContractID'])
conversions = conversions.drop(columns=['Unique_ContractID','Unique_BranchID'])
conversions.groupby(conversions.RiskTier).mean()
conversions.groupby(conversions.BookYear).mean()
conversions

# %% Checking risk one more time
conversions.groupby(conversions.BookYear)['Unique_ContractID','RiskTier','Rescored_Tier_2017Model','Rescored_Tier_2018Model'].count()
# From this, which I should have checked way earlier, I see that years before 2015 just need to be dropped

# %% Moving on to another feature selection method
x = conversions.loc[~pd.isnull(conversions.RiskTier)][['BookYear','BookQtr','RiskTier','CreditScore','OwnRent','State','Term','TotalNote']]
y = conversions.loc[~pd.isnull(conversions.RiskTier)]['Converted?']
labelencoder = preprocessing.LabelEncoder()
x['State'] = labelencoder.fit_transform(x['State'])
x['OwnRent'] = labelencoder.fit_transform(x['OwnRent'])

bestfeatures = feature_selection.SelectKBest(score_func=feature_selection.chi2,k=5)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(10,'Score'))
