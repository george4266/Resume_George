import pandas as pd
import pathlib
import calendar
import datetime as dt
from datetime import timedelta
import numpy as np
import swifter
import plotly.express as px
import plotly.offline as py
import seaborn as sns
import matplotlib.pyplot as plt




datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Eric' / 'Outputs'


VT_Originations_11262019 = datafolder / 'VT_Originations_11262019.txt'

VT_Performance_1_11262019 = datafolder / 'VT_Performance_1_11262019.txt'
VT_Performance_2_11262019 = datafolder / 'VT_Performance_2_11262019.txt'
VT_Performance_3_11262019 = datafolder / 'VT_Performance_3_11262019.txt'
VT_Performance_4_11262019 = datafolder / 'VT_Performance_4_11262019.txt'



Originations = pd.read_csv(VT_Originations_11262019, sep=',',dtype={'Rescored_Tier_2017MOdel':'str', 'Rescored_Tier_2018MOdel':'str' ,'Segment':'str'})


OriginationsAdjusted = Originations[['Unique_ContractID','BookDate','CreditScore','ProductType' ]]



Performance = pd.read_csv(VT_Performance_1_11262019, sep=',').append(pd.read_csv(VT_Performance_2_11262019, sep=','))\
 .append(pd.read_csv(VT_Performance_3_11262019, sep=',')).append(pd.read_csv(VT_Performance_4_11262019, sep=','))

Performance.reset_index(drop=True, inplace=True)

PerformanceAdjusted = Performance[['Unique_ContractID','MonthsOnBook','Unique_BranchID','NetReceivable','GrossBalance']]



Portfolio = pd.merge(PerformanceAdjusted, OriginationsAdjusted, how='left', on='Unique_ContractID')
Portfolio['BookDate'] = pd.to_datetime(Portfolio['BookDate'])
Portfolio['months_added'] = pd.to_timedelta(Portfolio['MonthsOnBook'],'M')
Portfolio['HalfAdjustedDate'] = Portfolio['BookDate'] + Portfolio['months_added']
Portfolio['AdjustedDate'] = Portfolio['HalfAdjustedDate'].dt.strftime('%m/%Y')
Portfolio = Portfolio.dropna(subset = ['BookDate']) #not considering loans with not book date
Portfolio = Portfolio[Portfolio.ProductType != 'MH']
Portfolio = Portfolio[Portfolio.MonthsOnBook < 49] #not considering loans that have been on book for more than 4 years




sum_Portfolio = Portfolio.groupby(['AdjustedDate','ProductType'])['GrossBalance','NetReceivable'].sum()
sum_Portfolio = pd.DataFrame(sum_Portfolio)
sum_Portfolio = sum_Portfolio.sort_index()
sum_Portfolio = sum_Portfolio.reset_index(drop=False)
sum_Portfolio['AdjustedDate'] = pd.to_datetime(sum_Portfolio['AdjustedDate'])
sum_Portfolio = sum_Portfolio.sort_values('AdjustedDate')
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[0])
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[0])
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[0])
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[0])
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[-1])
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[-1])
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[-1])

LiveCheck = []
for row in sum_Portfolio['ProductType']:
    if row == "LC":    LiveCheck.append('Live Check')
    else:            LiveCheck.append('Direct Loan')
sum_Portfolio["LiveCheck"] = LiveCheck





sum_Portfolio.to_csv(outputfolder / 'Net_Receivable_by_Loan_Type.csv')






#Gross Balance

fig, ax = plt.subplots(figsize=(15,7))
sum_Portfolio.groupby(['AdjustedDate','LiveCheck']).sum()['GrossBalance'].unstack().plot(ax=ax)
plt.title('Portfolio Growth by Product Type (Gross Balance)')
plt.xlabel('Date')
plt.grid()
plt.legend()
plt.ylabel('Value in Billions')
plt.savefig('PortfolioGrowth_ProductType_GrossBalance.png')


#Net NetReceivable

fig, ax = plt.subplots(figsize=(15,7))
sum_Portfolio.groupby(['AdjustedDate','LiveCheck']).sum()['NetReceivable'].unstack().plot(ax=ax)
plt.title('Portfolio Growth by Product Type (Net Receivable)')
plt.xlabel('Date')
plt.grid()
plt.legend()
plt.ylabel('Value in Billions')
axes = plt.gca()
axes.set_ylim([-50000000,1000000000])
plt.savefig('PortfolioGrowth_ProductType_NetReceivable.png')
