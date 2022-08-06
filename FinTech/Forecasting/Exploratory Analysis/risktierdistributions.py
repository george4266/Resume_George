# %% Imports and file load
import pathlib, datetime
import numpy as np, pandas as pd, seaborn as sns, matplotlib, matplotlib.pyplot as plt

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'
origfile = datafolder / 'VT_Originations_11262019.txt'
orig = pd.read_csv(origfile, sep=',', low_memory=False)

# %% Data preparation
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 4), 'Tier_MultipleModels'] = 5
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 3), 'Tier_MultipleModels'] = 4
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 2), 'Tier_MultipleModels'] = 3
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 1), 'Tier_MultipleModels'] = 2
orig.loc[(orig.ProductType == 'LC') & (orig.Tier_MultipleModels == 0.5), 'Tier_MultipleModels'] = 1
orig.loc[orig.Tier_MultipleModels == 0.5, 'Tier_MultipleModels'] = 1

newloan = orig[['Unique_ContractID','Unique_BranchID','BookDate','Tier_MultipleModels','State','ProductType','IP_Unique_ContractID']]
loancomparison = orig[['Unique_ContractID','BookDate','Tier_MultipleModels','ProductType']].merge(newloan, how='right', left_on='Unique_ContractID', right_on='IP_Unique_ContractID')

orig.groupby('Tier_MultipleModels')['TotalNote'].mean()

loancomparison = loancomparison.loc[~pd.isnull(loancomparison.Unique_ContractID_x)]

loancomparison.rename(columns={'BookDate_x':'IP_BookDate','Tier_MultipleModels_x':'IP_Tier_MultipleModels','ProductType_x':'IP_ProductType','Unique_ContractID_y':'Unique_ContractID','BookDate_y':'BookDate','Tier_MultipleModels_y':'Tier_MultipleModels','ProductType_y':'ProductType'},inplace=True)

loancomparison.dropna(subset=['Tier_MultipleModels'],inplace=True)
loancomparison.dropna(subset=['IP_Tier_MultipleModels'],inplace=True)
loancomparison.drop(columns=['Unique_ContractID_x'],inplace=True)

loancomparison.BookDate = loancomparison.BookDate.apply(pd.to_datetime)
loancomparison.IP_BookDate = loancomparison.BookDate.apply(pd.to_datetime)
loancomparison['BookYear'] = loancomparison.BookDate.dt.year
loancomparison['BookQtr'] = loancomparison.BookDate.dt.quarter
loancomparison['BookMonth'] = loancomparison.BookDate.dt.month
loancomparison['IP_BookYear'] = loancomparison.IP_BookDate.dt.year
loancomparison['IP_BookQtr'] = loancomparison.IP_BookDate.dt.quarter
loancomparison['IP_BookMonth'] = loancomparison.IP_BookDate.dt.month

# %% Plotting
fig, axs = plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
axs = axs.flatten()
axs[5].set_axis_off()
for tier in sorted(loancomparison.IP_Tier_MultipleModels.unique()):
    num = int(tier - 1)
    ax = sns.countplot(x='Tier_MultipleModels',data=loancomparison.loc[loancomparison.IP_Tier_MultipleModels == tier],hue='State', ax=axs[num])
    ax.set_title('IP Risk Tier - {}'.format(tier))
    ax.set_xlabel('New Risk Tier')
    ax.set_ylabel('Number of Originations')
    if num > 0:
        ax.get_legend().remove()
fig.show()
plt.savefig(pathlib.Path(outputfolder / 'State_IP_RT_distributions.png'))

fig, axs = plt.subplots(8,5,figsize=(20,30),sharex=True,sharey=True)
axs = axs.flatten()
i = 0
for state in sorted(loancomparison.State.unique()):
    for tier in sorted(loancomparison.IP_Tier_MultipleModels.unique()):
        ax = sns.countplot(x='Tier_MultipleModels',data=loancomparison.loc[(loancomparison.IP_Tier_MultipleModels == tier)&(loancomparison.State == state)], ax=axs[i])
        ax.set_title('({}) IP Risk Tier {}'.format(state,tier))
        ax.set_xlabel('New Risk Tier')
        ax.set_ylabel('Number of Originations')
        i = i + 1
plt.savefig(pathlib.Path(outputfolder / 'state_RT_distributions.png'))

fig, axs = plt.subplots(figsize=(8,6))
sns.heatmap(loancomparison.pivot_table(index='Tier_MultipleModels',columns=['IP_Tier_MultipleModels'], aggfunc='size'),cmap='Blues',ax=axs)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

yearplot = loancomparison.pivot_table(index=['Tier_MultipleModels','IP_Tier_MultipleModels','BookYear'], aggfunc='size').to_frame().rename(columns={0:'Count'}).reset_index()
yearplot['Tier_MultipleModels'] = ["$%s$" % x for x in yearplot['Tier_MultipleModels']]
stateplot = loancomparison.pivot_table(index=['Tier_MultipleModels','IP_Tier_MultipleModels','State'], aggfunc='size').to_frame().rename(columns={0:'Count'}).reset_index()

fig, axs = plt.subplots(1,5,figsize=(16,6),sharex=True,sharey=True)
axs = axs.flatten()
for tier in sorted(yearplot.IP_Tier_MultipleModels.unique()):
    num = int(tier - 1)
    ax = sns.barplot(x='BookYear',y='Count',data=yearplot.loc[yearplot.IP_Tier_MultipleModels == tier],hue='Tier_MultipleModels',ax=axs[num])
    ax.set_title('IP Risk Tier - {}'.format(tier))
    ax.set_xlabel('Book Year')
    ax.set_ylabel('Number of Originations')
    if num != 1:
        ax.get_legend().remove()

fig, axs = plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
axs = axs.flatten()
axs[5].set_axis_off()
for tier in sorted(stateplot.IP_Tier_MultipleModels.unique()):
    num = int(tier - 1)
    ax = sns.barplot(x='State',y='Count',data=stateplot.loc[stateplot.IP_Tier_MultipleModels == tier],hue='Tier_MultipleModels',ax=axs[num])
    ax.set_title('IP Risk Tier - {}'.format(tier))
    ax.set_xlabel('State')
    ax.set_ylabel('Number of Originations')
    if num != 1:
        ax.get_legend().remove()

fig, axs = plt.subplots(1,5,figsize=(16,9),sharex=True,sharey=True)
axs = axs.flatten()
for tier in sorted(loancomparison.IP_Tier_MultipleModels.unique()):
    num = int(tier - 1)
    ax = sns.countplot(x='Tier_MultipleModels',data=loancomparison.loc[loancomparison.IP_Tier_MultipleModels == tier],hue='BookYear', ax=axs[num])
    ax.set_title('IP Risk Tier - {}'.format(tier))
    ax.set_xlabel('New Risk Tier')
    ax.set_ylabel('Number of Originations')
    if num > 0:
        ax.get_legend().remove()

# %% Curious as to which factors are important for the change in risk tier other than what Dun brought up
loancomparison.groupby('Tier_MultipleModels')['IP_Tier_MultipleModels'].mean()

loancomparison.columns

loancomparison.groupby('State')['Tier_MultipleModels'].mean()

loancomparison.groupby('IP_ProductType')['Tier_MultipleModels'].mean()
