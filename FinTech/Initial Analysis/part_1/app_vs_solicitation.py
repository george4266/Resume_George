import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.offline as py

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Kaitlyn' / 'Outputs'

origfile = datafolder / 'VT_Originations_2.txt'
perffile1 = datafolder / 'VT_Performance_1.txt'
perffile2 = datafolder / 'VT_Performance_2.txt'
perffile3 = datafolder / 'VT_Performance_3.txt'
perffile4 = datafolder / 'VT_Performance_4.txt'

orig = pd.read_csv(origfile, sep=',')
perf = pd.read_csv(perffile1, sep=',').append(pd.read_csv(perffile2, sep=','))\
 .append(pd.read_csv(perffile3, sep=',')).append(pd.read_csv(perffile4, sep=','))

perf.reset_index(drop=True, inplace=True)

time = [2015, 2016, 2017, 2018]

###  Steps to get approved app data ###
def solicitation(orig, perf, time):
    solicitation_data = pd.merge(perf[['Unique_ContractID','Contacted_Memos','Approved_Apps']],orig[['Unique_ContractID','BookDate']],
                            on='Unique_ContractID',how='left')
    solicitation_data = solicitation_data.dropna(subset=['Contacted_Memos']) # added this in after to compare total approved vs contacted and approved
    solicitation_data['BookDate'] = pd.to_datetime(solicitation_data['BookDate'])
    solicitation_data['BookDate'] = solicitation_data['BookDate'].dt.normalize()
    solicitation_data['BookDate'] = solicitation_data['BookDate'].loc[solicitation_data['BookDate']<dt.datetime(2018,12,30)]
    solicitation_data = solicitation_data.groupby(solicitation_data.BookDate.dt.to_period('Y'))['Approved_Apps'].sum().to_frame()

    approved_apps = [0, 0, 356535, 130671]
    contacted_approved = [0, 0, 81998, 105941]
    combined_approved_apps = {'time':time, 'approved_apps': approved_apps}
    combined_approved_apps = pd.DataFrame(combined_approved_apps)
    combined_contacted_approved = {'time':time, 'contacted_approved':contacted_approved}
    combined_contacted_approved = pd.DataFrame(combined_contacted_approved)

    fig = px.bar(combined_approved_apps['time'], y=combined_approved_apps['approved_apps']),title='Number of Approved Applications Over Time',labels={'time':'year','approved_apps':'Number of Approved Apps'})
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'approved_apps_by_year.html').resolve().as_posix())
    fig.write_image(pathlib.Path(outputfolder / 'approved_apps_by_year.png').resolve().as_posix()) #Saves figure as png as well
    fig.show()
    #combined_approved_apps_plot = sns.barplot(data=combined_approved_apps, x='time', y='approved_apps').get_figure()
    #combined_approved_apps_plot.savefig('approved_apps_by_year.png')
    #combined_contacted_approved_plot = sns.barplot(data=combined_contacted_approved, x='time', y='contacted_approved').get_figure()
    #combined_contacted_approved_plot.savefig('contacted_approved_by_year.png')

def declined(orig, perf, time):
    declined_apps = pd.merge(perf[['Unique_ContractID','Contacted_Memos','Declined_Apps']],orig[['Unique_ContractID','BookDate']],
                            on='Unique_ContractID',how='left')
    declined_apps = declined_apps.dropna(subset=['Contacted_Memos']) # added this in after to compare total approved vs contacted and approved
    declined_apps['BookDate'] = pd.to_datetime(declined_apps['BookDate'])
    declined_apps['BookDate'] = declined_apps['BookDate'].dt.normalize()
    declined_apps['BookDate'] = declined_apps['BookDate'].loc[declined_apps['BookDate']<dt.datetime(2018,12,30)]
    declined_apps = declined_apps.groupby(declined_apps.BookDate.dt.to_period('Y'))['Contacted_Memos'].sum().to_frame()# was lazy and used this to calculate total contacted last

    #print(declined_apps.head())

    declined_count = [0, 0, 73649, 110279]
    combined_declined_total = {'time':time, 'declined_count':declined_count}
    contacted_declined = [0, 0, 20575, 90605]
    contacted_declined_combined = {'time':time, 'contacted_declined':contacted_declined}
    contacted = [308008, 421959, 536856, 926547]
    contacted_combined = {'time':time, 'contacted':contacted}
    combined_declined_total = pd.DataFrame(combined_declined_total)
    contacted_declined_combined = pd.DataFrame(contacted_declined_combined)
    contacted_combined = pd.DataFrame(contacted_combined)

    #combined_declined_total_plot = sns.barplot(data=combined_declined_total, x='time', y='declined_count').get_figure()
    #combined_declined_total_plot.savefig('declined_by_year.png')
    #contacted_declined_combined_plot = sns.barplot(data=contacted_declined_combined, x='time', y='contacted_declined').get_figure()
    #contacted_declined_combined_plot.savefig('contacted_declined_by_year.png')
    contacted_combined_plot = sns.barplot(data=contacted_combined, x='time', y='contacted').get_figure()
    contacted_combined_plot.savefig('contacted_by_year.png')




declined(orig, perf, time)
