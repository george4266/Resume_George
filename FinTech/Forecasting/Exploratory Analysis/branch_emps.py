# %% import and files
import pandas as pd
import numpy as np
import datetime as dt
import pathlib
import swifter
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
origination_file = datafolder / 'VT_Originations_11262019.txt'
branch_file = datafolder / 'VT_Branches.txt'

origin = pd.read_csv(origination_file)
branch = pd.read_csv(branch_file)
branch
branch['Month'] = pd.to_datetime(branch['Month'])


# %% calculate originations per employee aggregated by branch_file

origin['BookDate'] = pd.to_datetime(origin['BookDate'])
origin1 = origin.loc[(origin['BookDate'] >= dt.datetime(2019,8,1)) & (origin['BookDate'] < dt.datetime(2019,9,1))][['Unique_ContractID', 'Unique_BranchID']]
origin1 = origin1.groupby('Unique_BranchID')['Unique_ContractID'].count().to_frame().reset_index()

branch1 = branch.loc[(branch['Month'] >= dt.datetime(2019,8,1)) & (branch['Month'] < dt.datetime(2019,9,1))]
branch1 = branch1[['Unique_BranchID', 'NumActiveEmployees']]

emp_prod = origin1.merge(branch1, on='Unique_BranchID')

emp_prod['avg_emp_prod'] = emp_prod['Unique_ContractID']/emp_prod['NumActiveEmployees']
emp_prod['avg_emp_prod'].mean()
emp_prod['NumActiveEmployees'].mean()
emp_prod['NumActiveEmployees'].sum()
emp_prod['Unique_ContractID'].sum()
emp_prod['Unique_BranchID'].count()
sns.barplot(x='Unique_BranchID', y='avg_emp_prod', data=emp_prod)
