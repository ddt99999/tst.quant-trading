# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:30:40 2016

@author: tongtz
"""

import numpy as np
import pandas as pd
import datetime as dt
#from urllib.request import urlretrieve
#%matplotlib inline

url1 = 'http://hopey.netfonds.no/posdump.php?'
url2 = 'date=%s%s%s&paper=NKE.N&csv_format=csv'
url = url1 + url2

year = '2016'
month = '07'
days = ['21','22']

NKE = pd.DataFrame()
for day in days:
    NKE = NKE.append(pd.read_csv(url % (year, month, day),
                     index_col=0,
                     header=0,
                     parse_dates=True))
NKE.columns = ['bid','bdepth','bdeptht','offer','odepth','odeptht']

NKE.info()

NKE['bid'].plot(figsize=(10,6))

to_plot = NKE[['bid','bdeptht']][(NKE.index > dt.datetime(2016,4,20,0,0)) & (NKE.index < dt.datetime(2016,4,21,2,59))]
to_plot.plot(subplots=True, style='b', figsize=(10,6))

NKE_resam = NKE.resample(rule='5min', how='mean')
np.round(NKE_resam.head(), 2)

NKE_resam['bid'].fillna(method='ffill').plot()

def reversal(x):
    return 2 * 126 - x

NKE_resam['bid'].fillna(method='ffill').apply(reversal).plot()

x = np.linspace(-5,5,500)
e = np.random.standard_normal(len(x)) * 2
data = pd.DataFrame({'x':x, 'y':2*x**2 - 0.5*x + 3 + e})
data.plot(x='x', y='y', style='r.')

model = np.polyfit(x=data['x'], y=data['y'],deg=2)
model

import matplotlib.pyplot as plt
data.plot(x='x', y='y', style='r.')
plt.plot(x, np.polyval(model, x), lw=2.0)