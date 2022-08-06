# %% Import Data
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import datetime as dt

datafolder = pathlib.Path.cwd().parent / 'Data'

dummy_file = datafolder / 'dummy_test_forecast_Jan2019.csv'
jan_2019 = datafolder / 'feature_gen_dists_jan_2019.csv'
mktg_file = datafolder / 'VT_Marketing_11012019.txt'
apps_file = datafolder / 'VT_Applications_11262019.txt'

dummy = pd.read_csv(dummy_file)
actual = pd.read_csv(jan_2019)
mktg = pd.read_csv(mktg_file)
apps = pd.read_csv(apps_file)

actual = actual.loc[actual['Unique_BranchID'] <= 163]
dummy = dummy.loc[dummy['Unique_BranchID'] <= 163]
apps = apps.loc[apps['Unique_BranchID'] <= 163]
mktg = mktg.loc[mktg['Unique_BranchID'] <= 163]

actual = actual.loc[actual.MonthsOnBook == 1]

# %%

apps.head()
apps = apps.loc[(apps.Application_Source == 'Former Customer')|(apps.Application_Source == 'New Customer')]
apps.AppCreatedDate = pd.to_datetime(apps.AppCreatedDate)

apps = apps.loc[(apps.AppCreatedDate <= dt.datetime(2019, 2, 1))&(apps.AppCreatedDate >= dt.datetime(2019,1,1))&(apps.Booked_Indicator == 1)]
apps.head()
apps.describe()

# %%
mktg.columns
mktg.CashDate = pd.to_datetime(mktg.CashDate)
mktg = mktg.loc[(mktg.CashDate <= dt.datetime(2019, 2, 1))&(mktg.CashDate >= dt.datetime(2019,1,1))]

dummy.groupby('ProductType').describe()
mktg.describe()
