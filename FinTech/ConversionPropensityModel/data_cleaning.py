# %% Imports
import pathlib, sklearn
import datetime as dt, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn import preprocessing, feature_selection

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
conversions = conversions.loc[conversions.BookYear < 2019]
conversions = conversions.loc[~((conversions.BookYear == 2018)&(conversions.BookQtr >= 3))]

# %% Filtering out values and columns
# Years before 2015 have too many missing values
conversions = conversions.loc[conversions.BookYear >= 2015]
# Credit scores range from 300 to 850
conversions = conversions.loc[(conversions.CreditScore <= 850)&(conversions.CreditScore >= 300)]
# Consolidating risk tiers
conversions.loc[(~pd.isnull(conversions.Rescored_Tier_2018Model))&(conversions.BookDate.dt.year == 2018),'RiskTier'] = conversions.Rescored_Tier_2018Model
conversions.loc[(~pd.isnull(conversions.Rescored_Tier_2017Model))&(conversions.BookDate.dt.year <= 2017),'RiskTier'] = conversions.Rescored_Tier_2017Model
conversions.loc[(pd.isnull(conversions.RiskTier)&pd.isnull(conversions.Rescored_Tier_2018Model)&~pd.isnull(conversions.Rescored_Tier_2017Model)),'RiskTier'] = conversions.Rescored_Tier_2017Model

conversions.groupby(pd.isnull(conversions.RiskTier))['Converted?'].mean()
conversions = conversions.loc[~pd.isnull(conversions.RiskTier)]
# Dropping now-unnecessary columns
conversions = conversions.drop(columns=['Rescored_Tier_2017Model','Rescored_Tier_2018Model','Segment','Unique_ApplicationID','ProductType','IP_Unique_ContractID','Unique_ContractID','Unique_BranchID','AmountFinanced','CashToCustomer','NetCash','TotalOldBalance','BookDate'])

conversions.head()
conversions.to_csv(pathlib.Path(datafolder / 'conversions.csv'),index=False)

# %% Encoding categorical values
labelencoder = preprocessing.LabelEncoder()
encoded_conversions = conversions
encoded_conversions.OwnRent = labelencoder.fit_transform(conversions.OwnRent)
encoded_conversions.State = labelencoder.fit_transform(conversions.State)
encoded_conversions.to_csv(pathlib.Path(datafolder / 'encoded_conversions.csv'),index=False)

conversions.head()

# %% Feature comparison
conversions.groupby('RiskTier')['Converted?'].mean()
conversions.groupby('State')['Converted?'].mean()
conversions.groupby('Term')['Converted?'].mean()
conversions.groupby('BookYear')['Converted?'].mean()
conversions.groupby('BookQtr')['Converted?'].mean()

sns.boxplot(y=conversions.CreditScore, x=conversions['Converted?'], width=.4,fliersize=2)
sns.boxplot(y=conversions.TotalNote, x=conversions['Converted?'], width=.4,fliersize=2)
sns.boxplot(y=conversions.RegularPayment, x=conversions['Converted?'], width=.4,fliersize=2)
