# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:48:42 2016

@author: tongtz
"""

from sys import version_info
version_info

import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns; sns.set()

rows = 10000
index = pd.date_range(dt.datetime.now().date(), periods=rows, freq='H')
df = pd.DataFrame(data=np.random.standard_normal((rows,5)),
                  columns=['No1','No2','No3','No4','No5'],
                  index=index)

df.info()
df.head()

df['Grl'] = np.random.choice(['A','B','C','D'], rows)
df.tail()

grouped = df.groupby('Grl')
type(grouped)
grouped.head()

grouped.size()
grouped.sum()
grouped.mean()['No2'].head()
grouped.describe()

grouped.get_group('A').head()
grouped.aggregate({'No1': np.mean,
                   'No3': np.std})
                   
#matlablib inline
grouped.mean().plot(kind='barh')

f = lambda x: x.hour % 2 == 0
df['Grl2'] = np.where(f(df.index), 'even', 'odd')
df.tail()

grouped = df.groupby(['Grl','Grl2'])
grouped.size()
grouped.mean()
grouped.aggregate([np.min, np.mean, np.max])[['No1','No2']].boxplot(return_type='dict')
grouped.filter(lambda x: np.mean(x['No2']) > 0.0).head(100) # filter the group No2 mean > 0.0

df1 = pd.DataFrame(data=['100','200','300','400'],
                   columns=['A',],
                   index=['a','b','c','d'])
df1     

df2 = pd.DataFrame(data=['200','150','50'],
                   columns=['B',],
                   index=['f','b','d'])             
df2

df1.append(df2)
df1.append(df2, ignore_index=True)

pd.concat((df1, df2))

df1.join(df2)

df3 = pd.DataFrame({'A': df1['A'], 'B':df2['B']})
df3

c = pd.Series([250,150,50], index=['b','d','c'])
df1['C'] = c
df2['C'] = c

df1
df2

pd.merge(df1, df2)
pd.merge(df1, df2, on='C')
pd.merge(df1, df2, how='outer')
pd.merge(df1, df2, left_index=True, right_index=True)
pd.merge(df1, df2, on='C', left_index=True, right_index=True)



