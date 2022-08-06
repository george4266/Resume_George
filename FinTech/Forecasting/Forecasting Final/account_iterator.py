# %% Takes all accounts in a dataset and advances them to the next month
import pathlib
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import re

datafolder = pathlib.Path.cwd().parent / 'Data'

#############################################################

# RENAME ACCORDINGLY

inputfile = datafolder / 'simulation_set_begin_jan2019_month1.csv'
outputfile = datafolder / 'simulation_set_begin_jan2019_month2.csv'

##############################################################

feature_generation_files = ['feature_generation_sep_2018.csv',
                            'feature_generation_oct_2018.csv',
                            'feature_generation_nov_2018.csv',
                            'feature_generation_dec_2018.csv',
                            'feature_generation_jan_2019.csv',
                            'feature_generation_feb_2019.csv',
                            'feature_generation_mar_2019.csv',
                            'feature_generation_apr_2019.csv',
                            'feature_generation_may_2019.csv',
                            'feature_generation_jun_2019.csv',
                            'feature_generation_jul_2019.csv',
                            'feature_generation_aug_2019.csv',
                            'feature_generation_sep_2019.csv'
                            ]


data = pd.DataFrame()
feat_cols = ['Unique_ContractID','MonthsOnBook','ProductType','30+_Indicator','Tier_MultipleModels','Solicitation_Memos','Contacted_Memos','NetReceivable','GrossBalance','CurrentMonth']
data_dtypes = {'MonthsOnBook':'int16','ProductType':'category','30+_Indicator':'int8','Tier_MultipleModels':'int8','Contacted_Memos':'int16','Solicitation_Memos':'int16'}

for feat_file in feature_generation_files:
    data = data.append(pd.read_csv(pathlib.Path(datafolder / feat_file),usecols=feat_cols,dtype=data_dtypes,parse_dates=['CurrentMonth']))

accounts = pd.read_csv(inputfile)


# %% Advance variables that will randomly change based on a distribution
# Determine month to month changes
data['NextMonth'] = data['CurrentMonth'] + pd.DateOffset(months=1)
data_month_changes = data.merge(data[['Unique_ContractID','30+_Indicator','Solicitation_Memos','Contacted_Memos','GrossBalance','NetReceivable','CurrentMonth']],how='inner',left_on=['Unique_ContractID','NextMonth'],right_on=['Unique_ContractID','CurrentMonth'])
del(data)
data_month_changes['Change_30+_Indicator'] = data_month_changes['30+_Indicator_y'] - data_month_changes['30+_Indicator_x']
data_month_changes['Change_Solicitation_Memos'] = data_month_changes['Solicitation_Memos_y'] - data_month_changes['Solicitation_Memos_x']
data_month_changes['Change_Contacted_Memos'] = data_month_changes['Contacted_Memos_y'] - data_month_changes['Contacted_Memos_x']
data_month_changes['Change_NetReceivable'] = (data_month_changes['NetReceivable_y'] - data_month_changes['NetReceivable_x']) / data_month_changes['NetReceivable_x']
data_month_changes['Change_GrossBalance'] = (data_month_changes['GrossBalance_y'] - data_month_changes['GrossBalance_x'])/data_month_changes['GrossBalance_x']
data_month_changes = data_month_changes.replace([np.inf, -np.inf], 0)

data_month_changes = data_month_changes[['Unique_ContractID','ProductType','Tier_MultipleModels','MonthsOnBook','Change_30+_Indicator','Change_Solicitation_Memos','Change_Contacted_Memos','Change_NetReceivable','Change_GrossBalance']]

# Changes randomly determined by MonthsOnBook, ProductType, RiskTier
varlist = ['Solicitation_Memos','Contacted_Memos','GrossBalance','NetReceivable']

for product in accounts['ProductType'].unique():
    for tier in accounts.loc[accounts.ProductType == product, 'Tier_MultipleModels'].unique():
        for MoB in accounts.loc[(accounts.ProductType == product)&(accounts.Tier_MultipleModels == tier), 'MonthsOnBook'].unique():
            id_list = accounts.loc[(accounts.ProductType == product) & (accounts.Tier_MultipleModels == tier) & (accounts.MonthsOnBook == MoB), 'Unique_ContractID']

            if MoB <= 36:
                dataslice = data_month_changes.loc[(data_month_changes.ProductType == product) & (data_month_changes.Tier_MultipleModels == tier) & (data_month_changes.MonthsOnBook == MoB)]
            else:
                dataslice = data_month_changes.loc[(data_month_changes.ProductType == product) & (data_month_changes.Tier_MultipleModels == tier) & (data_month_changes.MonthsOnBook > 36)]

            for var in varlist:
                change_var = 'Change_' + var
                varfreq = dataslice[change_var].value_counts(normalize=True)
                change_vals = np.random.choice(varfreq.index, size=id_list.count(), p=varfreq.values)

                if (var == 'NetReceivable') or (var == 'GrossBalance'):
                    change_vals = change_vals * accounts.loc[(accounts.ProductType == product) & (accounts.Tier_MultipleModels == tier) & (accounts.MonthsOnBook == MoB), var]

                accounts.loc[(accounts.ProductType == product) & (accounts.Tier_MultipleModels == tier) & (accounts.MonthsOnBook == MoB), var] = accounts.loc[(accounts.ProductType == product) & (accounts.Tier_MultipleModels == tier) & (accounts.MonthsOnBook == MoB), var] + change_vals

                if var == 'Contacted_Memos':
                    greater_than = [1 if x > 0 else 0 for x in change_vals]
                    accounts[(accounts.ProductType == product) & (accounts.Tier_MultipleModels == tier) & (accounts.MonthsOnBook == MoB), 'greater_than?']
                    = greater_than

# %% Advance variables with expected and calculated changes
accounts['MonthsOnBook'] = accounts['MonthsOnBook'] + 1
accounts['Months_left'] = accounts['Months_left'] - 1
accounts['months_til_avg_dl_r'] = accounts['months_til_avg_dl_r'] - 1
accounts['months_til_avg_lc_r'] = accounts['months_til_avg_lc_r'] - 1
accounts['months_til_avg_dl_c'] = accounts['months_til_avg_dl_c'] - 1
accounts['months_til_avg_lc_c'] = accounts['months_til_avg_lc_c'] - 1
accounts['CurrentMonth'] = accounts['CurrentMonth'] + pd.DateOffset(months=1)

accounts['available_offer'] = 0
accounts.loc[accounts['HighCredit'] > accounts['NetReceivable'], 'available_offer'] = accounts['HighCredit'] - accounts['NetReceivable']
accounts['Utilization'] = 1
accounts.loc[accounts.GrossBalance < accounts.HighCredit, 'Utilization'] = accounts['GrossBalance'] / accounts['HighCredit']
accounts['Utilization'] = accounts['Utilization'].replace([-np.inf,np.inf],0)
accounts['PaydownPercent'] = 0
accounts.loc[accounts.NetReceivable <= accounts.AmountFinanced, 'PaydownPercent'] = 1 - (accounts.NetReceivable / accounts.AmountFinanced)
accounts['SC_ratio'] = 0
accounts.loc[(accounts['Contacted_Memos'] !=0) & (accounts['Solicitation_Memos'] != 0), 'SC_ratio'] = accounts['Contacted_Memos']/accounts['Solicitation_Memos']

# Advance dummy account label
lastNum = re.compile(r'(?:[^\d]*(\d+)[^\d]*)+')
def increment(s):
    """ look for the last sequence of number(s) in a string and increment """
    m = lastNum.search(s)
    if m:
        next = str(int(m.group(1))+1)
        start, end = m.span(1)
        s = s[:max(end-len(next), start)] + next + s[end:]
        return s
        accounts.update(accounts.loc[accounts['Unique_ContractID'].str.contains('DUMMY'), 'Unique_ContractID'].apply(lambda x: increment(x)))

# %% EXPORT COMPLETED FILE
accounts.to_csv(outputfile, index=False)
