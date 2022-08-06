import pandas as pd
import pathlib, calendar, os
import datetime as dt, numpy as np
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Kaitlyn' / 'Outputs'
origfile = datafolder / 'VT_Originations_11012019.txt'
perffile1 = datafolder / 'VT_Performance_1.txt'
perffile2 = datafolder / 'VT_Performance_2.txt'
perffile3 = datafolder / 'VT_Performance_3.txt'
perffile4 = datafolder / 'VT_Performance_4.txt'

orig = pd.read_csv(origfile, sep=',', low_memory=False)
perf = pd.read_csv(perffile1, sep=',').append(pd.read_csv(perffile2, sep=','))\
 .append(pd.read_csv(perffile3, sep=',')).append(pd.read_csv(perffile4, sep=','))

perf.reset_index(drop=True, inplace=True)

#To make the waterfall, I'm starting with all live checks from 2018
conversions = orig[['Unique_ContractID','Unique_BranchID','BookDate','Rescored_Tier_2018Model','ProductType','State']].loc[(orig.ProductType == 'LC')&(orig.BookDate.str.contains('2018'))]
conversions = conversions.merge(perf[['Unique_ContractID','MonthsOnBook','Solicitation_Memos','Contacted_Memos','Declined_Apps','Approved_Apps','ProcessStatus']], how='left', on='Unique_ContractID')

conversioncount = 0
df = 

for month in sorted(conversions.MonthsOnBook.unique()):
    conversioncount = conversioncount + conversions.loc[(conversions.MonthsOnBook==month)&(conversions.ProcessStatus == 'Renewed')]['Unique_ContractID'].count()
    conversions.

conversioncount

#Renaming just to make the merge a bit easier by getting rid of the _x or _y nonsense
#conversions = conversions.rename(columns={'Unique_ContractID':'Prev_ContractID','BookDate':'Prev_Book','ProductType':'Prev_Product'})
#Doing this merge just so I can see which live checks have been converted
#conversions = conversions.merge(orig[['Unique_ContractID','BookDate','ProductType','IP_Unique_ContractID']],how='left',left_on='Prev_ContractID',right_on='IP_Unique_ContractID').drop(columns='IP_Unique_ContractID')
#Boolean helper column just indicates if it was converted or not
#conversions.loc[conversions.Unique_ContractID > 0, 'Converted'] = 1
#conversions = conversions.rename(columns={'Unique_ContractID':'New_ContractID'})
#Forcing these object string columns into datetime objects, just in case
#conversions[['Prev_Book','BookDate']] = conversions[['Prev_Book','BookDate']].apply(pd.to_datetime)
#Merging my data from originations with my performance data
#conversions = conversions.merge(conversions[['Prev_ContractID','MonthsOnBook']].groupby('Prev_ContractID').max().reset_index().rename(columns={'MonthsOnBook':'MaxMonthsOnBook'}), on='Prev_ContractID',how='left')
#conversions = conversions.merge(perf[['Unique_ContractID','MonthsOnBook','Solicitation_Memos','Contacted_Memos','Declined_Apps','Approved_Apps','ProcessStatus']], how='left', left_on='Prev_ContractID',right_on='Unique_ContractID').drop(columns='Unique_ContractID')
###I noticed there is a decent handful of rows in which ProcessStatus and Converted dont match up, but I'm going to be leaving process status as the authority
###conversions = conversions.loc[~conversions.Prev_ContractID.isin(conversions.loc[(conversions.ProcessStatus=='Closed')&(conversions.Converted==1)]['Prev_ContractID'])]
#conversions.loc[(conversions.ProcessStatus=='Renewed')&(np.isnan(conversions.Converted))]
###

#Helper columns for boolean indicators that I'm interested in for the waterfall
#conversions['NeverSolicited'] = 0
#conversions.loc[(np.isnan(conversions.Solicitation_Memos)|(conversions.Solicitation_Memos==0))&(conversions.ProcessStatus!='Renewed')&np.isnan(conversions.Declined_Apps)&np.isnan(conversions.Approved_Apps)&np.isnan(conversions.Contacted_Memos),'NeverSolicited'] = 1
#conversions['SolicitNoContact'] = 0
#conversions.loc[np.isnan(conversions.Contacted_Memos)&(conversions.Solicitation_Memos>0)&(conversions.ProcessStatus!='Renewed')&np.isnan(conversions.Declined_Apps)&np.isnan(conversions.Approved_Apps),'SolicitNoContact'] = 1
#conversions['ContactNoApp'] = 0
#conversions.loc[np.isnan(conversions.Declined_Apps)&np.isnan(conversions.Approved_Apps)&(conversions.Contacted_Memos>0)&(conversions.ProcessStatus!='Renewed'),'ContactNoApp'] = 1
#conversions['AppButDeclined'] = 0
#conversions.loc[np.isnan(conversions.Approved_Apps)&(conversions.ProcessStatus!='Renewed')&(conversions.Declined_Apps>0),'AppButDeclined'] = 1
#conversions['ApprovedNoRenewal'] = 0
#conversions.loc[(conversions.Approved_Apps>0)&(conversions.ProcessStatus!='Renewed'),'ApprovedNoRenewal'] = 1
#conversions['Converted'] = 0
#conversions.loc[conversions.ProcessStatus=='Renewed','Converted']=1

conversions = conversions.groupby(['Prev_ContractID','NeverSolicited','SolicitNoContact','ContactNoApp','AppButDeclined','ApprovedNoRenewal','Converted'])['MonthsOnBook'].first().to_frame().reset_index()

waterfalldata = conversions.groupby('MonthsOnBook').agg({'NeverSolicited':sum,'SolicitNoContact':sum,'ContactNoApp':sum,'AppButDeclined':sum,'ApprovedNoRenewal':sum,'Converted':sum,'Prev_ContractID':'count'}).cumsum().reset_index()

waterfalldata
#waterfalldata= waterfalldata.merge(aggcounter,on='MonthsOnBook',how='left')
#waterfalldata.NeverSolicited = waterfalldata.NeverSolicited / waterfalldata.Prev_ContractID
#waterfalldata.SolicitNoContact = waterfalldata.SolicitNoContact / waterfalldata.Prev_ContractID
#waterfalldata.ContactNoApp = waterfalldata.ContactNoApp / waterfalldata.Prev_ContractID
#waterfalldata.AppButDeclined = waterfalldata.AppButDeclined / waterfalldata.Prev_ContractID
#waterfalldata.ApprovedNoRenewal = waterfalldata.ApprovedNoRenewal/ waterfalldata.Prev_ContractID
#waterfalldata.Converted = waterfalldata.Converted / waterfalldata.Prev_ContractID
#waterfalldata.drop(columns=['Prev_ContractID'],inplace=True)
#waterfalldata = waterfalldata.unstack().reset_index().rename(columns={'level_0':'Status','level_1':'MonthsOnBook',0:'Value'})
#waterfalldata = waterfalldata.loc[waterfalldata.Status != 'MonthsOnBook']

fig = px.area(waterfalldata,x='MonthsOnBook',y='Value',color='Status')
fig.show()

fig = go.Figure(go.Waterfall(
    orientation = "v",
    x = waterfalldata.loc[waterfalldata.MonthsOnBook == 8,'Status'],
    y = waterfalldata.loc[waterfalldata.MonthsOnBook == 8,'Value']
))

fig.show()
