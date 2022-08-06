# %% Imports, setup
import h2o, os
import pathlib
import datetime as dt, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from h2o.estimators.random_forest import H2ORandomForestEstimator

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Kaitlyn' / 'Outputs'
convfile = datafolder / 'conversions.csv'

# %% Intitialize h2o
h2o.init()

# %% Loading data
conv = h2o.import_file(pathlib.Path(datafolder / 'conversions.csv').resolve().as_posix())
conv.describe()

# %% Model setup
conv[conv['RiskTier'] == 0.5, 'RiskTier'] = 0
conv['RiskTier'] = conv['RiskTier'].asfactor()
conv['BookQtr'] = conv['BookQtr'].asfactor()
conv['Converted?'] = conv['Converted?'].asfactor()
x_cols = ['RiskTier','CreditScore','OwnRent','State','Term','TotalNote','RegularPayment','BookYear','BookQtr']
y_col = 'Converted?'
train,test,valid = conv.split_frame(ratios=[.6,.2])

conv.describe()

# %% Running model
randf = H2ORandomForestEstimator()

randf.train(x=x_cols,y=y_col, training_frame=train)
performance = randf.model_performance(test_data=test)

print(performance)
