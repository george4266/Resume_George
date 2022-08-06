# %% import
import pandas as pd
import pathlib, calendar, os
import datetime as dt, numpy as np
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% dataimports and setup
datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Kaitlyn' / 'Outputs'
origfile = datafolder / 'VT_Originations_11012019.txt'
branchfile = datafolder / 'VT_Branches.txt'

orig = pd.read_csv(origfile, sep=',', low_memory=False)
branch= pd.read_csv(branchfile, sep=',')

### DETERMINING CONVERSIONS ###
livechecks = orig[['Unique_ContractID','Unique_BranchID','BookDate','RiskTier','ProductType','State','Segment']].loc[orig.ProductType == 'LC']
livechecks = livechecks.rename(columns={'Unique_ContractID':'LC_ID','Unique_BranchID':'LC_Branch','BookDate':'LC_Book','RiskTier':'LC_Risk','State':'LC_State','Segment':'LC_Segment'}).drop(columns=['ProductType'])

conversions = livechecks.merge(orig[['Unique_ContractID','Unique_BranchID','State','BookDate','RiskTier','ProductType','IP_Unique_ContractID']],how='left',left_on='LC_ID',right_on='IP_Unique_ContractID')
conversions = conversions.drop(columns=['IP_Unique_ContractID'])
conversions[['LC_Book','BookDate']] = conversions[['LC_Book','BookDate']].apply(pd.to_datetime)
conversions['TimeBetween'] = (conversions.BookDate - conversions.LC_Book).dt.days / 30 #Finding difference between old loan and new loan book dates
conversions['BookYear'] = pd.DatetimeIndex(conversions.LC_Book).year #Helper calculated column
conversions['Converted?'] = 0 #Helper calculated column
conversions.loc[conversions.Unique_ContractID > 0, 'Converted?'] = 1

conversionsbyyear = conversions.groupby(['BookYear','TimeBetween'])['Converted?'].sum().groupby('BookYear').cumsum().reset_index() #Aggregating by group for the graph
conversionsbyyear = conversions.groupby('BookYear')['LC_ID'].count().reset_index().merge(conversionsbyyear, how='inner',on='BookYear')
conversionsbyyear['Conversion Cum %'] = conversionsbyyear['Converted?'] / conversionsbyyear.LC_ID

conversionsbyyearandstate = conversions.groupby(['LC_State','BookYear','TimeBetween'])['Converted?'].sum().groupby(['LC_State','BookYear']).cumsum().reset_index()
conversionsbyyearandstate = conversions.groupby(['LC_State','BookYear'])['LC_ID'].count().reset_index().merge(conversionsbyyearandstate, how='inner',on=['BookYear','LC_State'])
conversionsbyyearandstate['Conversion Cum %'] = conversionsbyyearandstate['Converted?'] / conversionsbyyearandstate.LC_ID

conversionsbyyearandbranch = conversions.groupby(['LC_Branch','BookYear','TimeBetween'])['Converted?'].sum().groupby(['LC_Branch','BookYear']).cumsum().reset_index()
conversionsbyyearandbranch = conversions.groupby(['LC_Branch','BookYear'])['LC_ID'].count().reset_index().merge(conversionsbyyearandbranch, how='inner',on=['BookYear','LC_Branch'])
conversionsbyyearandbranch['Conversion Cum %'] = conversionsbyyearandbranch['Converted?'] / conversionsbyyearandbranch.LC_ID

conversionsbysegment = conversions.loc[(pd.isna(conversions.LC_Segment)==False)|(pd.isna(conversions.LC_Risk)==False)]
conversionsbysegment = conversionsbysegment.groupby(['LC_Risk','LC_Segment','TimeBetween'])['Converted?'].sum().groupby(['LC_Risk','LC_Segment']).cumsum().reset_index()
conversionsbysegment = conversions.loc[(pd.isna(conversions.LC_Segment)==False)|(pd.isna(conversions.LC_Risk)==False)].groupby(['LC_Risk','LC_Segment'])['LC_ID'].count().reset_index().merge(conversionsbysegment, how='inner',on=['LC_Risk','LC_Segment'])
conversionsbysegment['Conversion Cum %'] = conversionsbysegment['Converted?'] / conversionsbysegment.LC_ID

conversionsbyrisk = conversions.loc[np.isnan(conversions.LC_Risk)==False]
conversionsbyrisk = conversions.groupby(['LC_Risk','BookYear','TimeBetween'])['Converted?'].sum().groupby(['LC_Risk','BookYear']).cumsum().reset_index()
conversionsbyrisk = conversions.groupby(['LC_Risk','BookYear'])['LC_ID'].count().reset_index().merge(conversionsbyrisk, how='inner',on=['BookYear','LC_Risk'])
conversionsbyrisk['Conversion Cum %'] = conversionsbyrisk['Converted?'] / conversionsbyrisk.LC_ID

### DETERMINING RENEWALS ### essentially same approach as with conversions
nonLCs = orig[['Unique_ContractID','Unique_BranchID','BookDate','RiskTier','ProductType','State','Segment']].loc[orig.ProductType != 'LC']
nonLCs = nonLCs.rename(columns={'Unique_ContractID':'Prev_ID','Unique_BranchID':'Prev_Branch','BookDate':'Prev_Book','RiskTier':'Prev_Risk','State':'Prev_State','Segment':'Prev_Segment'})
renewals = nonLCs.merge(orig[['Unique_ContractID','Unique_BranchID','BookDate','RiskTier','ProductType','IP_Unique_ContractID']],how='left',left_on='Prev_ID',right_on='IP_Unique_ContractID')
renewals = renewals.drop(columns=['IP_Unique_ContractID'])
renewals[['Prev_Book','BookDate']] = renewals[['Prev_Book','BookDate']].apply(pd.to_datetime)
renewals['TimeBetween'] = (renewals.BookDate - renewals.Prev_Book).dt.days / 30
renewals['BookYear'] = pd.DatetimeIndex(renewals.Prev_Book).year
renewals['Renewed?'] = 0
renewals.loc[renewals.Unique_ContractID > 0, 'Renewed?'] = 1
renewalsbyyear = renewals.groupby(['BookYear','TimeBetween'])['Renewed?'].sum().groupby('BookYear').cumsum().reset_index()
renewalsbyyear = renewals.groupby('BookYear')['Prev_ID'].count().reset_index().merge(renewalsbyyear, how='inner',on='BookYear')
renewalsbyyear['Renewal Cum %'] = renewalsbyyear['Renewed?'] / renewalsbyyear.Prev_ID

renewalsbyyearandstate = renewals.groupby(['Prev_State','BookYear','TimeBetween'])['Renewed?'].sum().groupby(['Prev_State','BookYear']).cumsum().reset_index()
renewalsbyyearandstate = renewals.groupby(['Prev_State','BookYear'])['Prev_ID'].count().reset_index().merge(renewalsbyyearandstate, how='inner',on=['BookYear','Prev_State'])
renewalsbyyearandstate['Renewal Cum %'] = renewalsbyyearandstate['Renewed?'] / renewalsbyyearandstate.Prev_ID

renewalsbyyearandbranch = renewals.groupby(['Prev_Branch','BookYear','TimeBetween'])['Renewed?'].sum().groupby(['Prev_Branch','BookYear']).cumsum().reset_index()
renewalsbyyearandbranch = renewals.groupby(['Prev_Branch','BookYear'])['Prev_ID'].count().reset_index().merge(renewalsbyyearandbranch, how='inner',on=['BookYear','Prev_Branch'])
renewalsbyyearandbranch['Renewal Cum %'] = renewalsbyyearandbranch['Renewed?'] / renewalsbyyearandbranch.Prev_ID

renewalsbyrisk = renewals.loc[np.isnan(renewals.Prev_Risk)==False]
renewalsbyrisk = renewals.groupby(['Prev_Risk','BookYear','TimeBetween'])['Renewed?'].sum().groupby(['Prev_Risk','BookYear']).cumsum().reset_index()
renewalsbyrisk = renewals.groupby(['Prev_Risk','BookYear'])['Prev_ID'].count().reset_index().merge(renewalsbyrisk, how='inner',on=['BookYear','Prev_Risk'])
renewalsbyrisk['Renewal Cum %'] = renewalsbyrisk['Renewed?'] / renewalsbyrisk.Prev_ID

#Creating the interactive figures with plotly
def getconversionbyyear():
    fig = px.line(conversionsbyyear.loc[conversionsbyyear.BookYear >= 2015], x='TimeBetween', y='Conversion Cum %', color='BookYear', range_x=[0,36], range_y=[0,0.55],title='Live Check Conversion Rates Over Time by Year',labels={'TimeBetween':'Months on Book','Conversion Cum %':'Cumulative Conversion Rate'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'conversion_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'conversion_rates_over_time.png').resolve().as_posix()) #Saves figure as png as well
    fig.show()
def getconversionbystate():
    fig = px.line(conversionsbyyearandstate.loc[conversionsbyyearandstate.BookYear >= 2015], x='TimeBetween', y='Conversion Cum %',\
     color='BookYear', range_x=[0,36], range_y=[0,0.55], line_group='LC_State',title='Live Check Conversion Rates Over Time by State',labels={'TimeBetween':'Months on Book','Conversion Cum %':'Cumulative Conversion Rate','LC_State':'State'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'state_conversion_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'state_conversion_rates_over_time.png').resolve().as_posix())
    fig.show()
def getconversionbybranch():
    fig = px.line(conversionsbyyearandbranch.loc[conversionsbyyearandbranch.BookYear >= 2015], x='TimeBetween', y='Conversion Cum %',\
     color='LC_Branch', range_x=[0,36], range_y=[0,0.7], line_group='BookYear',title='Live Check Conversion Rates Over Time by Branch',labels={'TimeBetween':'Months on Book','Conversion Cum %':'Cumulative Conversion Rate','LC_Branch':'Branch'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'branch_conversion_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'branch_conversion_rates_over_time.png').resolve().as_posix())
    fig.show()
def getconversionbysegment():
    fig = px.line(conversionsbysegment.loc[conversionsbysegment.LC_Risk < 4], x='TimeBetween', y='Conversion Cum %', line_dash='LC_Segment',line_group='LC_Risk', color='LC_Risk', range_x=[0,12], range_y=[0,0.4],title='Live Check Conversion Rates Over Time by Segment',\
    labels={'TimeBetween':'Months on Book','Conversion Cum %':'Cumulative Conversion Rate','LC_Segment':'Segment','LC_Risk':'Risk Tier'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'segment_conversion_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'segment_conversion_rates_over_time.png').resolve().as_posix())
    fig.show()
def getconversionbyrisk():
    fig = px.line(conversionsbyrisk.loc[conversionsbyrisk.BookYear >= 2015], x='TimeBetween', y='Conversion Cum %',\
     line_group='LC_Risk',color='BookYear', range_x=[0,36], range_y=[0,0.7], title='Live Check Conversion Rates Over Time by Risk Tier',labels={'TimeBetween':'Months on Book','Conversion Cum %':'Cumulative Conversion Rate','LC_Risk':'RiskTier'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'risk_conversion_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'risk_conversion_rates_over_time.png').resolve().as_posix())
    fig.show()
def getrenewalbyyear():
    fig = px.line(renewalsbyyear.loc[renewalsbyyear.BookYear >= 2015], x='TimeBetween', y='Renewal Cum %', color='BookYear', range_x=[0,36], range_y=[0,0.55],title='Direct Loan Renewal Rates Over Time',labels={'TimeBetween':'Months on Book','Renewal Cum %':'Cumulative Renewal Rate'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'renewal_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'renewal_rates_over_time.png').resolve().as_posix())
    fig.show()
def getrenewalbystate():
    fig = px.line(renewalsbyyearandstate.loc[renewalsbyyearandstate.BookYear >= 2015], x='TimeBetween', y='Renewal Cum %',\
     color='BookYear', range_x=[0,36], range_y=[0,0.55], line_group='Prev_State',title='Direct Loan Renewal Rates Over Time by State',labels={'TimeBetween':'Months on Book','Renewal Cum %':'Cumulative Renewal Rate','Prev_State':'State'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'state_renewal_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'state_renewal_rates_over_time.png').resolve().as_posix())
    fig.show()
def getrenewalbybranch():
    fig = px.line(renewalsbyyearandbranch.loc[renewalsbyyearandbranch.BookYear >= 2015], x='TimeBetween', y='Renewal Cum %',\
     color='Prev_Branch', range_x=[0,36], range_y=[0,0.7], line_group='BookYear',title='Direct Loan Renewal Rates Over Time',labels={'TimeBetween':'Months on Book','Renewal Cum %':'Cumulative Renewal Rate','Prev_Branch':'Branch'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'branch_renewal_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'branch_renewal_rates_over_time.png').resolve().as_posix())
    fig.show()
def getrenewalbyrisk():
    fig = px.line(renewalsbyrisk.loc[renewalsbyrisk.BookYear >= 2015], x='TimeBetween', y='Renewal Cum %',\
     line_group='Prev_Risk',color='BookYear', range_x=[0,36], range_y=[0,0.7], title='Direct Loan Renewal Rates Over Time by Risk Tier',labels={'TimeBetween':'Months on Book','Renewal Cum %':'Cumulative Renewal Rate','Prev_Risk':'RiskTier'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'risk_renewal_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'risk_renewal_rates_over_time.png').resolve().as_posix())
    fig.show()
def getcombinedbyyear():
    fig = px.line(combinedconvrenewbyyear.loc[combinedconvrenewbyyear.BookYear >= 2015], x='TimeBetween', y='Conv/Renew Cum %', color='BookYear', range_x=[0,36], range_y=[0,0.6],title='Combined Conversion and Renewal Rates Over Time by Year',labels={'TimeBetween':'Months on Book','Conv/Renew Cum %':'Cumulative Combined Conversion and Renewal Rate'},height=576, width=1024)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'combined_rates_over_time.html').resolve().as_posix(),auto_open=False)
    #fig.write_image(pathlib.Path(outputfolder / 'combined_rates_over_time.png').resolve().as_posix()) #Saves figure as png as well
    fig.show()

### Total combined renewal and conversion rate
combinedconvrenew = orig[['Unique_ContractID','Unique_BranchID','BookDate','RiskTier','ProductType','State','Segment']].rename(columns={'Unique_ContractID':'Prev_ID','Unique_BranchID':'Prev_Branch','BookDate':'Prev_Book','RiskTier':'Prev_Risk','State':'Prev_State','Segment':'Prev_Segment','ProductType':'Prev_Product'})
combinedconvrenew = combinedconvrenew.merge(orig[['Unique_ContractID','Unique_BranchID','BookDate','RiskTier','ProductType','IP_Unique_ContractID']],how='left',left_on='Prev_ID',right_on='IP_Unique_ContractID').drop(columns=['IP_Unique_ContractID'])
combinedconvrenew[['Prev_Book','BookDate']] = combinedconvrenew[['Prev_Book','BookDate']].apply(pd.to_datetime)
combinedconvrenew['TimeBetween'] = (combinedconvrenew.BookDate - combinedconvrenew.Prev_Book).dt.days / 30
combinedconvrenew['BookYear'] = pd.DatetimeIndex(combinedconvrenew.Prev_Book).year
combinedconvrenew['Conv/Renew?'] = 0
combinedconvrenew.loc[combinedconvrenew.Unique_ContractID > 0, 'Conv/Renew?'] = 1
combinedconvrenewbyyear = combinedconvrenew.groupby(['BookYear','TimeBetween'])['Conv/Renew?'].sum().groupby('BookYear').cumsum().reset_index()
combinedconvrenewbyyear = combinedconvrenew.groupby('BookYear')['Prev_ID'].count().reset_index().merge(combinedconvrenewbyyear, how='inner',on='BookYear')
combinedconvrenewbyyear['Conv/Renew Cum %'] = combinedconvrenewbyyear['Conv/Renew?'] / combinedconvrenewbyyear.Prev_ID

### OVERALL BRANCH CONVERSION AND RENEWAL STATS ###
branch[['Month','BranchOpenDate']] = branch[['Month','BranchOpenDate']].apply(pd.to_datetime)
branch['Year'] = pd.DatetimeIndex(branch.Month).year
branch = branch.loc[branch.Year >= 2015]
YearAvgActiveEmployees = branch.groupby(['Unique_BranchID','Year'])['NumActiveEmployees'].mean().reset_index().rename(columns={'NumActiveEmployees':'YearAvgActiveEmployees'})
branch = branch.merge(YearAvgActiveEmployees, how='left', on=['Unique_BranchID','Year'])

#<!-- saved from url=(0014)about:internet -->
#orig.loc[(np.isnan(orig.RiskTier))&(np.isnan(orig.Rescored_Tier_2018Model))&(np.isnan(orig.Rescored_Tier_2018Model))]

### Don't run this yet it uses way too much memory
#branchconversionsandrenewals = branch[['Unique_BranchID','BranchOpenDate','State','Year','YearAvgActiveEmployees']].merge(conversionsbyyearandbranch.rename(columns={'LC_ID':'TotalLCforBookYear','TimeBetween':'ConvertedTimeBetween','Converted?':'ConvertedCount'}),how='left',right_on=['LC_Branch','BookYear'],left_on=['Unique_BranchID','Year'])
#branchconversionsandrenewals = branchconversionsandrenewals.drop(columns=['LC_Branch','BookYear'])
#branchconversionsandrenewals = branchconversionsandrenewals.merge(renewalsbyyearandbranch.rename(columns={'Prev_ID':'TotalNonLCforBookYear','TimeBetween':'RenewalTimeBetween','Renewed?':'ConvertedCount'}),how='left',right_on=['Prev_Branch','BookYear'],left_on=['Unique_BranchID','Year'])
#branchconversionsandrenewals = branchconversionsandrenewals.drop(columns=['Prev_Branch','BookYear'])

orig
