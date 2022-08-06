import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
import seaborn as sns
import datetime as dt

datafolder = pathlib.Path.cwd().parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'conversion_waterfalls'
orginfile = datafolder / 'VT_Originations_2.txt'
orgin = pd.open_csv(orginfile, sep=',')
booked = orgin[['BookDate', 'RiskTier','CashToCustomer']]
#booked['counter'] = np.ones(len(booked))
booked['BookDate'] = pd.to_datetime(booked['BookDate'])
booked['BookDate'] = booked['BookDate'].dt.normalize()
booked['BookDate'] = booked['BookDate'].loc[booked['BookDate']<dt.datetime(2019,7,1)]
booked['BookDate'] = booked['BookDate'].loc[booked['BookDate']>dt.datetime(2017,1,1)]
booked_by_month = booked.groupby([booked.BookDate.dt.to_period('M'),booked['RiskTier']])['CashToCustomer'].sum().to_frame()
#booked_by_month['tiercounts'] = booked_by_month['BookDate']
#booked_by_month = booked_by_month.drop(columns=['BookDate'])


percents = {}
print(booked_by_month.head(15))
for idx, row in booked_by_month.groupby(level=0):
    x = row['CashToCustomer'].sum()
    percents.update({idx:[]})
    #print(x)
    for index, row1 in row.iterrows():
        percents[idx].append(row1['CashToCustomer']/x)
        #print(row1['tiercounts'])

percents = pd.DataFrame(percents)
percents.to_csv('loan_orgin_by_tier2.csv')
#booked_by_month.to_csv('loan_orgin_by_tier2.csv')
#print(percents['2018-01'].sum())'''






#print(booked_by_month.head(15))
