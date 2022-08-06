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

PerformanceAdjusted = Performance[['Unique_ContractID','MonthsOnBook','Unique_BranchID','NetReceivable','GrossBalance','ProcessStatus']]

Portfolio = pd.merge(PerformanceAdjusted, OriginationsAdjusted, how='left', on='Unique_ContractID')
Portfolio['BookDate'] = pd.to_datetime(Portfolio['BookDate'])
Portfolio['months_added'] = pd.to_timedelta(Portfolio['MonthsOnBook'],'M')
Portfolio['HalfAdjustedDate'] = Portfolio['BookDate'] + Portfolio['months_added']
Portfolio['AdjustedDate'] = Portfolio['HalfAdjustedDate'].dt.strftime('%m/%Y')
Portfolio = Portfolio.dropna(subset = ['BookDate'])
Portfolio = Portfolio[Portfolio.ProductType != 'MH']
Portfolio = Portfolio[Portfolio.ProcessStatus == 'Closed']
Portfolio = Portfolio[Portfolio.MonthsOnBook < 49] #not considering loans that have been on book for more than 4 years
Portfolio['Dummy'] = 1
Portfolio.head(5)


#Credit Score by Loan Type

sum_Portfolio2 = Portfolio.groupby(['AdjustedDate','ProductType'])['Dummy'].sum()
sum_Portfolio2 = pd.DataFrame(sum_Portfolio2)
sum_Portfolio2 = sum_Portfolio2.sort_index()
sum_Portfolio2 = sum_Portfolio2.reset_index(drop=False)
sum_Portfolio2['AdjustedDate'] = pd.to_datetime(sum_Portfolio2['AdjustedDate'])
sum_Portfolio2 = sum_Portfolio2.sort_values('AdjustedDate')
sum_Portfolio2 = sum_Portfolio2.drop(sum_Portfolio2.index[0])
sum_Portfolio2 = sum_Portfolio2.drop(sum_Portfolio2.index[0])
sum_Portfolio2 = sum_Portfolio2.drop(sum_Portfolio2.index[0])
sum_Portfolio2 = sum_Portfolio2.drop(sum_Portfolio2.index[0])
sum_Portfolio2 = sum_Portfolio2.drop(sum_Portfolio2.index[-1])
sum_Portfolio2 = sum_Portfolio2.drop(sum_Portfolio2.index[-1])
sum_Portfolio2 = sum_Portfolio2.drop(sum_Portfolio2.index[-1])

sum_Portfolio2.head(50)

fig, ax = plt.subplots(figsize=(15,7))
sum_Portfolio2.groupby(['AdjustedDate','ProductType']).sum()['Dummy'].unstack().plot(ax=ax)
plt.title('Closings Per Month')
plt.grid()
plt.legend()
plt.xlabel('Date')
plt.savefig('Closed_Per_Month.png')
