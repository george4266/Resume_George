# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py

# %% data files
datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder/ 'VT_Originations_11262019.txt'
perffile1 = datafolder / 'VT_Performance_1_11262019.txt'
perffile2 = datafolder / 'VT_Performance_2_11262019.txt'
perffile3 = datafolder / 'VT_Performance_3_11262019.txt'
perffile4 = datafolder / 'VT_Performance_4_11262019.txt'
apps = datafolder / 'VT_Applications_11262019.txt'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Exploratory Analysis' / 'Outputs'


# %% data clean
apps = pd.read_csv(apps, sep=',', low_memory=False)
origin = pd.read_csv(origination_file, sep=',', low_memory=False)
#origin.drop(columns=['State', 'AmountFinanced', 'TotalNote', 'CashToCustomer', 'Segment', 'IP_Unique_ContractID', 'RegularPayment', 'Unique_ApplicationID'], inplace=True)
perf = pd.read_csv(perffile1, sep=',', low_memory=False).append(pd.read_csv(perffile2, sep=',', low_memory=False)).append(pd.read_csv(perffile3, sep=',', low_memory=False)).append(pd.read_csv(perffile4, sep=',', low_memory=False))


origin['BookDate'] = pd.to_datetime(origin['BookDate'])
'''combined = origin.merge(perf, on='Unique_ContractID', how='left')
combined.drop(columns=['Unique_CustomerID_y', 'Unique_BranchID_y'], inplace=True)
combined.rename(columns={'Unique_CustomerID_x':'Unique_CustomerID','Unique_BranchID_x':'Unique_BranchID'}, inplace=True)
combined.dropna(subset=['MonthsOnBook'], inplace=True)
combined.drop(columns=['Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'Approved_Apps'], inplace=True)

# %% Only if you need current month (takes forever)
combined['CurrentMonth'] = combined[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)

# %% checking closed stuff
#combined = combined.loc[(combined['CurrentMonth'] >= dt.datetime(2015,1,1)) & (combined['CurrentMonth'] < dt.datetime(2020,1,1))]
closed = perf.loc[perf['ProcessStatus'] == 'Closed']
closed_contracts = perf.loc[perf['Unique_ContractID'].isin(closed['Unique_ContractID'])]
closed_contracts
closed_contracts['30+_Indicator'].unique()
closed_contracts[closed_contracts['Unique_ContractID'] == 2117134].sort_values(by=['MonthsOnBook'], ascending=True)'''



# %% generating portfolio inflow vs time by number of apps

apps['AppCreatedDate'] = pd.to_datetime(apps['AppCreatedDate'])
inflow = apps.loc[(apps['AppStatus'] == 'Booked') & (apps['Booked_Indicator'] == 1) & (apps['AppCreatedDate'] >= dt.datetime(2018,1,1)) & (apps['AppCreatedDate'] < dt.datetime(2019,10,1))]
inflow_type = inflow[['Application_Source', 'AppCreatedDate', 'Booked_Indicator']].groupby([inflow.AppCreatedDate.dt.to_period('M'), inflow.Application_Source])['Booked_Indicator'].sum().to_frame().reset_index().pivot_table(index='AppCreatedDate',columns='Application_Source', values='Booked_Indicator')
inflow_type.unstack().reset_index()
inflow_type['AppCreatedDate'] = inflow_type['AppCreatedDate'].dt.strftime('%Y-%m')
plt.figure(figsize=(16,11))
plt.title('Inflow of Booked Applications by type (01/2018 - 09/2019)')
plt.xlabel('Number of Booked Apps')
plt.ylabel('Time')
plt.legend()
sns.lineplot(data=inflow_type.unstack().reset_index(), x='AppCreatedDate', y=0, hue='Application_Source')
#plt.savefig(pathlib.Path(outputfolder / 'portfolio_inflow_number_booked_apps.png'))
#fig = px.bar(inflow_type.unstack().reset_index(), x='AppCreatedDate', y=0, color='Application_Source', title='Portfolio Inflow By Application Source 2018 Forward')
#py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'inflow_vs_time.html').resolve().as_posix())

inflow_type.to_csv(outputfolder / 'inflow_by_source.csv')
# %% portfolio inflow bs time by $$ amount

inflow = apps.loc[(apps['AppStatus'] == 'Booked') & (apps['Booked_Indicator'] == 1) & (apps['AppCreatedDate'] >= dt.datetime(2018,1,1)) & (apps['AppCreatedDate'] < dt.datetime(2019,10,1))]
inflow = inflow.merge(origin, on='Unique_ContractID')
inflow_type = inflow[['Application_Source', 'BookDate', 'TotalNote']].groupby([inflow.BookDate.dt.to_period('M'), inflow.Application_Source])['TotalNote'].sum().to_frame().reset_index().pivot_table(index='BookDate',columns='Application_Source', values='TotalNote')
plt.figure(figsize=(16,11))
plt.title('Inflow of Originations by type (01/2018 - 09/2019)')
plt.xlabel('Time')
plt.ylabel('Sum of Total Note')
plt.legend()
sns.lineplot(data=inflow_type.unstack().reset_index(), x='BookDate', y=0, hue='Application_Source')
#plt.savefig(pathlib.Path(outputfolder / 'portfolio_inflow_totalnote.png'))
inflow_type.to_csv(outputfolder / 'inflow_by_source_dollar.csv')
