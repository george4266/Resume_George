import pandas as pd
import pathlib
import calendar
import datetime as dt
import numpy as np
import swifter
import plotly.express as px
import plotly.offline as py
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score


datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Initial Analysis' / 'Eric' / 'Outputs'

VT_Applications_11262019 = datafolder / 'VT_Applications_11262019.txt'

Applications = pd.read_csv(VT_Applications_11262019, sep=',')

Bucketed = []
for row in Applications['CreditScore']:
    if row > 850:    Bucketed.append('7')
    elif row > 696:  Bucketed.append('6')
    elif row > 639:  Bucketed.append('5')
    elif row > 602:  Bucketed.append('4')
    elif row > 566:  Bucketed.append('3')
    elif row > 299:  Bucketed.append('2')
    else:            Bucketed.append('1')
Applications["Bucketed"] = Bucketed

Renewal = []
for row in Applications['Application_Source']:
    if row == "Direct Loan Renewal":    Renewal.append('1')
    else:                              Renewal.append('0')
Applications["Renewal"] = Renewal

Test = Applications[['Bucketed','Renewal']]

x = Test.drop('Renewal', axis=1)
y = Test['Renewal']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

predictions = logmodel.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
