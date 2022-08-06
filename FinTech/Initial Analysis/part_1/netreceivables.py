import pandas as pd
import pathlib
import calendar
import datetime as dt
import numpy as np
import swifter
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

netreceivable = orig[['Unique_ContractID','ProductType','BookDate','RiskTier']].\
 merge(perf[['Unique_ContractID','NetReceivable','MonthsOnBook']],\
 on='Unique_ContractID',how='left').dropna()

netreceivable = netreceivable.loc[netreceivable['ProductType']!='MH']
netreceivable = netreceivable.loc[netreceivable['ProductType']!='RE']
netreceivable['BookDate'] = pd.to_datetime(netreceivable['BookDate'])
netreceivable['CurrentMonth'] = netreceivable[['BookDate','MonthsOnBook']].swifter.apply(lambda x: x['BookDate']+pd.DateOffset(months=x['MonthsOnBook']), axis=1)
netreceivable = netreceivable.loc[(netreceivable['CurrentMonth']<dt.datetime(2019,9,30))&(netreceivable['CurrentMonth']>dt.datetime(2015,1,1))]
aggnetreceivable = netreceivable[['Unique_ContractID','ProductType','NetReceivable','CurrentMonth']].groupby([netreceivable.CurrentMonth.dt.to_period('M'),netreceivable.ProductType])['NetReceivable'].mean().to_frame().reset_index().pivot_table(index='CurrentMonth',columns='ProductType', values='NetReceivable',margins=True,margins_name='Overall').drop(index=['Overall'], columns=['Sales'])
aggnetreceivable.index = aggnetreceivable.index.astype('datetime64[ns]')



fig = px.line(aggnetreceivable.unstack().reset_index(),x='CurrentMonth',y=0,title='Average Net Receivables over Time',color='ProductType',labels={'CurrentMonth':'Month','0':'Average Net Receivable ($)'},height=576, width=512)
py.offline.plot(fig, filename=pathlib.Path(outputfolder / 'new_net_receivables_over_time.html').resolve().as_posix())
fig.write_image(pathlib.Path(outputfolder / 'new_net_receivables_over_time.png').resolve().as_posix())

byrisktier = netreceivable[['Unique_ContractID','ProductType','NetReceivable','CurrentMonth']].groupby([netreceivable.CurrentMonth.dt.to_period('M'),netreceivable.RiskTier])['NetReceivable'].mean().to_frame().reset_index().pivot_table(index='CurrentMonth',columns='RiskTier', values='NetReceivable',margins=True,margins_name='Overall').drop(index=['Overall'])
byrisktier.index = byrisktier.index.astype('datetime64[ns]')

byrisktier
