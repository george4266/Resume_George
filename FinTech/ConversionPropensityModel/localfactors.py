# %% import statements
import pandas as pd
import numpy as np
import pathlib

# %% reading in data
datafolder = pathlib.Path.cwd().parent / 'Data'
branch_file = datafolder/ 'VT_Branches_12102019.txt'
branches = pd.read_csv(branch_file, sep=',')
local_file = datafolder / 'ISESD20-40 ZIP Level Local Factors.csv'
local = pd.read_csv(local_file, sep=',')

# %% merging local info
branches = branches.loc[~(pd.isnull(branches.BrZip))]
branches['BrZip'] = branches['BrZip'].str[0:5].astype('int')

branches = branches.merge(local, how='left', left_on='BrZip', right_on='Name')
branches = branches.drop(columns='Name')
branches

outputloc = datafolder / 'VT_Branches_01072020.csv'
branches.to_csv(outputloc, index=False)
