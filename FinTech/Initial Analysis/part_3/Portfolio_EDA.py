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


OriginationsAdjusted = Originations[['Unique_ContractID','BookDate','CreditScore']]



Performance = pd.read_csv(VT_Performance_1_11262019, sep=',').append(pd.read_csv(VT_Performance_2_11262019, sep=','))\
 .append(pd.read_csv(VT_Performance_3_11262019, sep=',')).append(pd.read_csv(VT_Performance_4_11262019, sep=','))

Performance.reset_index(drop=True, inplace=True)

PerformanceAdjusted = Performance[['Unique_ContractID','MonthsOnBook','Unique_BranchID','NetReceivable','GrossBalance']]



Portfolio = pd.merge(PerformanceAdjusted, OriginationsAdjusted, how='left', on='Unique_ContractID')
Portfolio['BookDate'] = pd.to_datetime(Portfolio['BookDate'])
Portfolio['months_added'] = pd.to_timedelta(Portfolio['MonthsOnBook'],'M')
Portfolio['HalfAdjustedDate'] = Portfolio['BookDate'] + Portfolio['months_added']
Portfolio['AdjustedDate'] = Portfolio['HalfAdjustedDate'].dt.strftime('%m/%Y')
Portfolio = Portfolio.dropna(subset = ['BookDate'])
Portfolio = Portfolio[Portfolio.MonthsOnBook < 49] #not considering loans that have been on book for more than 4 years

sum_Portfolio = Portfolio.groupby(['AdjustedDate'])['GrossBalance','NetReceivable'].sum()
sum_Portfolio = pd.DataFrame(sum_Portfolio)
sum_Portfolio = sum_Portfolio.sort_index()
sum_Portfolio = sum_Portfolio.reset_index(drop=False)
sum_Portfolio['AdjustedDate'] = pd.to_datetime(sum_Portfolio['AdjustedDate'])
sum_Portfolio = sum_Portfolio.sort_values('AdjustedDate')
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[0])
sum_Portfolio = sum_Portfolio.drop(sum_Portfolio.index[-1])


fig, ax = plt.subplots(figsize=(15,7))
plt.plot( 'AdjustedDate', 'GrossBalance', data=sum_Portfolio, marker='', color='skyblue', linewidth=2)
plt.plot( 'AdjustedDate', 'NetReceivable', data=sum_Portfolio, marker='', color='olive', linewidth=2)
plt.title('Portfolio Growth')
plt.xlabel('Date')
plt.legend()
plt.ylabel('Value in Billions')
plt.savefig('PortfolioGrowth.png')
