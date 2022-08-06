# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
outputfolder = pathlib.Path.cwd()  / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'


# %% data clean
orig = pd.read_csv(origination_file, sep=',', low_memory=False)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))
orig['BookDate'] = pd.to_datetime(orig['BookDate'])

orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[orig.Tier_MultipleModels == 0.5, 'Tier_MultipleModels'] = 1


#%% combine origin and perf
combined = orig.merge(perf, on='Unique_ContractID', how='left')
combined.drop(columns=['Unique_CustomerID_y', 'Unique_BranchID_y'], inplace=True)
combined.rename(columns={'Unique_CustomerID_x':'Unique_CustomerID','Unique_BranchID_x':'Unique_BranchID'}, inplace=True)
combined.dropna(subset=['MonthsOnBook'], inplace=True)

combined.drop(columns=['Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'Approved_Apps'], inplace=True)

# %% Only if you need current month (takes forever)
combined['CurrentMonth'] = combined[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)



# %% risk tier product type relationship

risk_prod = orig.loc[(orig['BookDate'] >= dt.datetime(2018,1,1))&(orig.ProductType != 'Sales')][['Unique_ContractID', 'ProductType', 'Tier_MultipleModels']]
risk_prod.groupby('ProductType')['Unique_ContractID'].count().to_frame().reset_index()
risk_prod.groupby('Tier_MultipleModels')['Unique_ContractID'].count().to_frame().reset_index()
risk_prod.dropna(subset=['Tier_MultipleModels'],inplace=True)
len(risk_prod)

#risk_prod = risk_prod.groupby(['ProductType','RiskTier'])
fig, axs = plt.subplots(1,3,figsize=(16,9),sharex=True,sharey=True)
axs = axs.flatten()
num=0
for prod in risk_prod.ProductType.unique():
    ax = sns.countplot(x='Tier_MultipleModels',data=risk_prod.loc[(risk_prod.ProductType == prod)], ax=axs[num])
    ax.set_title('ProductType - {}'.format(prod))
    ax.set_xlabel('RiskTier')
    ax.set_ylabel('Number of Originations')
    num+=1
fig.show()

plt.savefig(pathlib.Path(outputfolder / 'PT_RT_dist_2018_forward.png'))


# %% origination correlation curiousity

orig_correlation = orig.loc[(orig['BookDate'] >= dt.datetime(2018,1,1)) & (orig['ProductType'] != 'Sales')][['Term', 'Tier_MultipleModels', 'State', 'CreditScore', 'AmountFinanced', 'CashToCustomer']]
sns.heatmap(orig_correlation.corr())
