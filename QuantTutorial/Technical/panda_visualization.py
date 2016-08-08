# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:16:28 2016

@author: tongtz
"""

import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns; sns.set()

np.random.seed(1000)
data = np.random.standard_normal((500,2)).cumsum(axis=0)
index = pd.date_range(start='2015-1-1', periods=len(data), freq='B')

df = pd.DataFrame(data, index=index, columns=['A','B'])

df.plot()

df.plot(subplots=True, color='b')
df.plot(legend=False, title='Custom Plot')
df.plot(style=['r.','m^'])
(df**4).plot(logy=True)
df['B'] = df['B'] * 100
df.plot(secondary_y='B', grid=True, figsize=(10,5))


