# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:35:10 2016

@author: tongtz
"""

from sys import version_info
version_info

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import warnings; warnings.simplefilter('ignore')

data = np.random.standard_normal((10000000, 2))
data.nbytes

df = pd.DataFrame(data, columns=['x','y'])

df.head()

res = df['x'] + df['y']
res[:5]

res = df.sum(axis=1)
res[:5]

res = df.values.sum(axis=1)
res[:5]

res = df.eval('x + y')
res[:5]

df[:1000].plot(x='x', y='y', kind='scatter')

res = df[df['x'] > 4.5]
res[:5]
#%time
res = df[(df['x'] > 4.5) | (df['y'] < -4.5)]
res[:5]

# %matplotlib inline
df[(df['x'] > 2.5) | (df['y'] < -2.5)][:1000].plot(x='x',y='y',kind='scatter')

# %matplotlib inline
df[(df['x'] > 2.5) & (df['y'] < -2.5)][:1000].plot(x='x',y='y',kind='scatter')



