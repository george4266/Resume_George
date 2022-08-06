import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Outputs' / 'conversion_waterfalls'
orginfile = datafolder / 'VT_Originations_11012019.txt'
appsfile = datafolder  / 'VT_Applications.txt'
perffile1 = datafolder / 'VT_Performance_1.txt'
perffile2 = datafolder / 'VT_Performance_2.txt'
perffile3 = datafolder / 'VT_Performance_3.txt'
perffile4 = datafolder / 'VT_Performance_4.txt'

orgin = pd.read_csv(orginfile, sep=',', low_memory=False)
perf = pd.read_csv(perffile1, sep=',').append(pd.read_csv(perffile2, sep=','))\
 .append(pd.read_csv(perffile3, sep=',')).append(pd.read_csv(perffile4, sep=','))

perf = perf[['Unique_ContractID', 'Solicitation_Memos', 'Contacted_Memos', 'Declined_Apps', 'Approved_Apps', 'MonthsOnBook', 'ProcessStatus']]
perf['Approved_Apps'] = perf['Approved_Apps'].fillna(value=0)
perf['Declined_Apps'] = perf['Declined_Apps'].fillna(value=0)
perf['Contacted_Memos'] = perf['Contacted_Memos'].fillna(value=0)
perf['Solicitation_Memos'] = perf['Solicitation_Memos'].fillna(value=0)


orgin = orgin[['Unique_ContractID', 'BookDate', 'ProductType', 'CashToCustomer']]
orgin['BookDate'] = pd.to_datetime(orgin['BookDate'])
orgin = orgin.loc[(orgin['BookDate']<dt.datetime(2019,1,1)) & (orgin['BookDate'] > dt.datetime(2017,12,31)) ]
orgin = orgin[orgin['ProductType'] == 'LC']

result = pd.merge(orgin, perf, on='Unique_ContractID', how='left')
len(result['MonthsOnBook'].unique())

def create_waterfall(month, result):
    result = result[result['MonthsOnBook'] <= month]
    #total = len(result[result['ProcessStatus'] == 'Renewed']) + len(result[(result['MonthsOnBook'] == 8) & (result['ProcessStatus'] == 'Open')]) + len(result[result['ProcessStatus'] == 'Closed'])

    converted = len(result[result['ProcessStatus'] == 'Renewed'])
    converted_table = result[result['ProcessStatus'] == 'Renewed']
    appr_no_renew = len(result[(result['Approved_Apps'] != 0) & (result['ProcessStatus'] == 'Open') & (result['MonthsOnBook'] == month)]) + len(result[(result['ProcessStatus'] == 'Closed') & (result['Approved_Apps'] != 0)])

    app_decline = len(result[(result['Declined_Apps'] != 0) & (result['MonthsOnBook'] == month) & (result['ProcessStatus'] != 'Renewed') & (result['Approved_Apps'] == 0)]) + len(result[(result['ProcessStatus'] == 'Closed') & (result['Declined_Apps'] != 0) & (result['Approved_Apps'] == 0)])

    contacted_no_app = len(result[(result['Declined_Apps'] == 0) & (result['Approved_Apps'] == 0) & (result['MonthsOnBook'] == month) & (result['Contacted_Memos'] != 0) & (result['ProcessStatus'] == 'Open')]) + len(result[(result['Approved_Apps'] == 0) & (result['Declined_Apps'] == 0) & (result['Contacted_Memos'] != 0) & (result['ProcessStatus'] == 'Closed')])

    solicited = len(result[(result['Solicitation_Memos'] != 0) & (result['Contacted_Memos'] == 0) & (result['Approved_Apps'] == 0) & (result['Declined_Apps'] == 0) & (result['MonthsOnBook'] == month) & (result['ProcessStatus'] == 'Open')]) + len(result[(result['ProcessStatus'] == 'Closed') & (result['Solicitation_Memos'] != 0) & (result['Contacted_Memos'] == 0) & (result['Approved_Apps'] == 0) & (result['Declined_Apps'] == 0)])

    never_solicited = len(result[(result['Solicitation_Memos'] == 0) & (result['Contacted_Memos'] == 0) & (result['Approved_Apps'] == 0) & (result['Declined_Apps'] == 0) & (result['MonthsOnBook'] == month) & (result['ProcessStatus'] == 'Open')]) + len(result[(result['ProcessStatus'] == 'Closed') & (result['Solicitation_Memos'] == 0) & (result['Contacted_Memos'] == 0) & (result['Approved_Apps'] == 0) & (result['Declined_Apps'] == 0)])

    total = never_solicited + solicited + app_decline + contacted_no_app + appr_no_renew + converted
    waterfall = {'Aggregates':[converted/total, app_decline/total, appr_no_renew/total, contacted_no_app/total, solicited/total, never_solicited/total], 'types': ['Converted','approved no renewal','app but declined', 'contacted no app', 'solicited', 'never_solicited']}
    waterfall = pd.DataFrame(waterfall)
    month = str(month)
    outputfolder = 'C:\\Users\\999Na\\Documents\\Senior design\\SeniorDesign2020\\Initial Analysis\\Nate\\Outputs\\conversion_waterfalls'
    waterfall.to_csv(outputfolder+'\\live_check_conversion_waterfall_month_'+month+'.csv')
    fig = go.Figure(go.Waterfall(
    name = "20", orientation = "v",
    x = waterfall['types'],
    textposition = "outside",
    y = waterfall['Aggregates']
    ))

    '''fig.update_layout(
            title = "Conversion waterfall @ month " + month,
            showlegend = True)
    string = 'conversion_waterfall_'+month+'.html'
    #py.offline.plot(fig, filename=pathlib.Path(outputfolder / string).resolve().as_posix(),auto_open=False)
    data = converted_table
    tips = px.data.tips()
    fig = px.histogram(tips, x="CashToCustomer",
                    title='Histogram Cash To Customer in Month '+month,
                    labels={'CashToCustomer':'Cash To Customer'}, # can specify one label per df column
                    opacity=0.8,
                    log_y=True, # represent bars with log scale
                    color_discrete_sequence=['indianred'] # color of histogram bars
                   )
    string2 = 'cash_to_customer_monthonbook_'+month+'_converted.html'
    py.offline.plot(fig, filename=pathlib.Path(outputfolder / string2).resolve().as_posix(),auto_open=False)'''

    return waterfall, total


for month in range(len(result['MonthsOnBook'].unique())):
    create_waterfall(month, result)
create_waterfall(8,result)
