import pandas as pd
import numpy as np
import plotly.express as px
import plotly.offline as py
import datetime as dt
import pathlib

datafolder = pathlib.Path.cwd().parent
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'solicitation_data'
orginfile = datafolder / 'VT_Originations_2.txt'
perffile1 = datafolder / 'VT_Performance_1.txt'
perffile2 = datafolder / 'VT_Performance_2.txt'
perffile3 = datafolder / 'VT_Performance_3.txt'
perffile4 = datafolder / 'VT_Performance_4.txt'

orgin = pd.read_csv(orginfile, sep=',')
perf = pd.read_csv(perffile1, sep=',').append(pd.read_csv(perffile2, sep=','))\
 .append(pd.read_csv(perffile3, sep=',')).append(pd.read_csv(perffile4, sep=','))

###  Steps to get approved app data ###

solicitation_data = pd.merge(perf[['Unique_ContractID','Contacted_Memos','Approved_Apps']],orgin[['Unique_ContractID','BookDate']],
                        on='Unique_ContractID',how='left')
declined_apps_data = pd.merge(perf[['Unique_ContractID','Contacted_Memos','Declined_Apps']],orgin[['Unique_ContractID','BookDate']],
                        on='Unique_ContractID',how='left')
def by_month(solicitation_data, declined_apps_data):
    solicitation_data['BookDate'] = pd.to_datetime(solicitation_data['BookDate'])
    solicitation_data['BookDate'] = solicitation_data['BookDate'].dt.normalize()
    solicitation_data = solicitation_data.loc[solicitation_data['BookDate']>dt.datetime(2016,12,31)]
    solicitation_data['BookMonth'] = pd.DatetimeIndex(solicitation_data.BookDate).month
    solicitation_data['BookYear'] = pd.DatetimeIndex(solicitation_data.BookDate).year
    number_contacted = solicitation_data.dropna(subset=['Contacted_Memos'])
    number_approved = solicitation_data.dropna(subset=['Approved_Apps'])
    number_contacted_approved = solicitation_data.dropna(subset=['Contacted_Memos', 'Approved_Apps'])
    number_contacted = number_contacted.groupby(['BookMonth', 'BookYear'])['Contacted_Memos'].sum().to_frame().reset_index()
    number_approved = number_approved.groupby(['BookMonth', 'BookYear'])['Approved_Apps'].sum().to_frame().reset_index()
    number_contacted_approved = number_contacted_approved.groupby(['BookMonth', 'BookYear'])['Approved_Apps'].sum().to_frame().reset_index()

    fig = px.bar(number_contacted, x='BookMonth', y='Contacted_Memos', color='BookYear', barmode='group', title='Total Contacted Memos by Year',height=576,width=512)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'contacted_memos_by_year_by_month.html').resolve().as_posix(),auto_open=False)
    fig = px.bar(number_contacted_approved, x='BookMonth', y='Approved_Apps', color='BookYear', barmode='group', title='Total Contacted and Approved by Year by Month',height=576,width=512)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'contacted_approved_by_year_by_month.html').resolve().as_posix(),auto_open=False)
    fig = px.bar(number_approved, x='BookMonth', y='Approved_Apps', color='BookYear', barmode='group', title='Total Approved Apps by Year by Month',height=576,width=512)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'approved_by_year_by_month.html').resolve().as_posix(),auto_open=False)

    declined_apps_data['BookDate'] = pd.to_datetime(declined_apps_data['BookDate'])
    declined_apps_data['BookDate'] = declined_apps_data['BookDate'].dt.normalize()
    declined_apps_data = declined_apps_data.loc[solicitation_data['BookDate']>dt.datetime(2016,12,31)]
    declined_apps_data['BookMonth'] = pd.DatetimeIndex(declined_apps_data.BookDate).month
    declined_apps_data['BookYear'] = pd.DatetimeIndex(declined_apps_data.BookDate).year
    declined_apps_data = declined_apps_data.dropna(subset=['Declined_Apps'])
    declined_contacted = declined_apps_data.dropna(subset=['Contacted_Memos'])
    declined_contacted = declined_contacted.groupby(['BookMonth', 'BookYear'])['Contacted_Memos'].sum().to_frame().reset_index()
    declined_apps_total = declined_apps_data.groupby(['BookMonth', 'BookYear'])['Declined_Apps'].sum().to_frame().reset_index()

    fig = px.bar(declined_apps_total, x='BookMonth', y='Declined_Apps', color='BookYear', barmode='group', title='Total Declined Apps by Year',height=576,width=512)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'declined_apps_by_year_by_month.html').resolve().as_posix(),auto_open=False)
    declined_apps_total.dtypes
    fig = px.bar(declined_contacted, x='BookMonth', y='Contacted_Memos', color='BookYear', barmode='group', title='Total Contacted and Declined by Year by Month',height=576,width=512)
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'contacted_declined_by_year_by_month.html').resolve().as_posix(),auto_open=False)
by_month(solicitation_data, declined_apps_data)


print(solicitation_data.isna().sum())
solicitation_data['BookDate'] = pd.to_datetime(solicitation_data['BookDate'])
solicitation_data['BookDate'] = solicitation_data['BookDate'].dt.normalize()
solicitation_data['BookYear'] = pd.DatetimeIndex(solicitation_data.BookDate).year
solicitation_data = solicitation_data.loc[solicitation_data['BookDate']<dt.datetime(2018,12,31)]
number_contacted = solicitation_data.dropna(subset=['Contacted_Memos'])
number_approved = solicitation_data.dropna(subset=['Approved_Apps'])
number_contacted_approved = solicitation_data.dropna(subset=['Contacted_Memos', 'Approved_Apps'])
number_contacted = number_contacted.groupby(number_contacted.BookYear)['Contacted_Memos'].sum().to_frame().reset_index()
number_approved = number_approved.groupby(number_approved.BookYear)['Approved_Apps'].sum().to_frame().reset_index()
number_contacted_approved = number_contacted_approved.groupby(number_contacted_approved.BookYear)['Approved_Apps'].sum().to_frame().reset_index()
#approved_data = number_contacted.merge(number_approved, on='BookYear', how='left').merge(number_contacted_approved, on='BookYear', how='left')
#approved_data


fig = px.bar(number_contacted, x='BookYear', y='Contacted_Memos', title='Total Contacted Memos by Year',labels={'BookYear':'Year','Contacted_Memos':'Contacted Memos'},height=576,width=512)
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'contacted_memos_by_year.html').resolve().as_posix(),auto_open=False)
fig = px.bar(number_approved, x='BookYear', y='Approved_Apps', title='Total Approved Apps by Year',labels={'BookYear':'Year','Approved_Apps':'Approved Apps'},height=576,width=512)
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'approved_apps_by_year.html').resolve().as_posix(),auto_open=False)
fig = px.bar(number_contacted_approved, x='BookYear', y='Approved_Apps', title='Total Contacted Approved Apps by Year',labels={'BookYear':'Year','Approved_Apps':'Approved Apps'},height=576,width=512)
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'contacted_and_approved_by_year.html').resolve().as_posix(),auto_open=False)
#fig.write_image(pathlib.Path(outputfolder / 'contacted_memos_by_year.png').resolve().as_posix())


declined_apps_data['BookDate'] = pd.to_datetime(declined_apps_data['BookDate'])
declined_apps_data['BookDate'] = declined_apps_data['BookDate'].dt.normalize()
declined_apps_data['BookYear'] = pd.DatetimeIndex(declined_apps_data.BookDate).year
declined_apps_data = declined_apps_data.loc[declined_apps_data['BookDate']<dt.datetime(2018,12,30)]
declined_apps_data = declined_apps_data.dropna(subset=['Declined_Apps'])
declined_contacted = declined_apps_data.dropna(subset=['Contacted_Memos'])
declined_contacted = declined_contacted.groupby(declined_contacted.BookYear)['Contacted_Memos'].sum().to_frame().reset_index()
declined_apps_total = declined_apps_data.groupby(declined_apps_data.BookYear)['Declined_Apps'].sum().to_frame().reset_index()
declined_contacted = declined_contacted[declined_contacted['BookYear'] != 2019]
declined_apps_total = declined_apps_total[declined_apps_total['BookYear'] != 2019]

fig = px.bar(declined_contacted, x='BookYear', y='Contacted_Memos', title='Total Declined and Contacted by Year',labels={'BookYear':'Year','Contacted_Memos':'Declined and Contacted'},height=576,width=512)
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'delined_contacted_by_year.html').resolve().as_posix(),auto_open=False)
fig = px.bar(declined_apps_total, x='BookYear', y='Declined_Apps', title='Total Declined Apps by Year',labels={'BookYear':'Year','Declined_Apps':'Declined Apps'},height=576,width=512)
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'total_delined_by_year.html').resolve().as_posix(),auto_open=False)
