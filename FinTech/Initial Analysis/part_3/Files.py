import pandas as pd
import pathlib
import calendar
import datetime as dt
from datetime import timedelta
import numpy as np
import swifter
import plotly.express as px
import plotly.offline as py
import seaborn as sns
import matplotlib.pyplot as plt




datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Eric' / 'Outputs'

VT_Performance_1_11262019 = datafolder / 'VT_Performance_1_11262019.txt'
VT_Performance_2_11262019 = datafolder / 'VT_Performance_2_11262019.txt'
VT_Performance_3_11262019 = datafolder / 'VT_Performance_3_11262019.txt'
VT_Performance_4_11262019 = datafolder / 'VT_Performance_4_11262019.txt'


Performance = pd.read_csv(VT_Performance_1_11262019, sep=',').append(pd.read_csv(VT_Performance_2_11262019, sep=','))\
 .append(pd.read_csv(VT_Performance_3_11262019, sep=',')).append(pd.read_csv(VT_Performance_4_11262019, sep=','))

Closed = Performance[Performance.ProcessStatus == 'Closed']
Closed

Performance.head(50)


VT_Originations_11262019 = datafolder / 'VT_Originations_11262019.txt'
Originations = pd.read_csv(VT_Originations_11262019, sep=',')
Originations.head(10)

MH = Originations[Originations.ProductType == 'MH']
len(MH)

VT_Branches_12102019 = datafolder / 'VT_Branches.txt'
Branches = pd.read_csv(VT_Branches_12102019, sep=',')

Branches.head()

VT_Marketing_11012019 = datafolder / 'VT_Marketing_11012019.txt'
Marketing_11012019 = pd.read_csv(VT_Marketing_11012019, sep=',')
Marketing_11012019.head()

VT_Applications_11262019 = datafolder / 'VT_Applications_11262019.txt'
Applications = pd.read_csv(VT_Applications_11262019, sep=',')
Applications.head(35)
Applications.Application_Source.unique()
