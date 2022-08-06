# %% import and files
import pandas as pd
import numpy as np
import pathlib

outputfolder = pathlib.Path.cwd() / 'ConversionPropensityModel' / 'Output'



sep_2018 = outputfolder / 'feature_gen_dists_sep_2018.csv'
aug_2019 = outputfolder / 'feature_gen_dists_aug_2019.csv'



files = [sep_2018, aug_2019]


sep_2018_f = pd.read_csv(sep_2018, sep=',', low_memory=False)
sep_2018_f.drop(columns=['indicator', 'OwnRent', 'Approved_Apps_x', 'Approved_Apps_y', 'counter', 'step_one', 'months_added', 'RiskTier'], inplace=True)
sep_2018_f.to_csv(sep_2018)



aug_2019_f = pd.read_csv(aug_2019, sep=',', low_memory=False)
aug_2019_f.drop(columns=['Renewed?', 'Closed?', 'PaidOrCharged?'], inplace=True)
aug_2019_f.columns
aug_2019_f.to_csv(aug_2019)
