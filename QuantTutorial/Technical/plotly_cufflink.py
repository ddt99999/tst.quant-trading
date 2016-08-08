# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:56:34 2016

@author: tongtz
"""

import plotly.plotly as py
import pandas as pd
import cufflinks as cf
import numpy as np
import json

pcreds = json.load(open('plotly_creds'))
py.sign_in('yves', pcreds['api_key'])

df = pd.DataFrame(np.random.randn(100,5),
                  index=pd.date_range('1/1/15', periods=100),
                  columns=['IBM','MSFT','GOOG','VERZ','APPL'])
df = df.cumsum()

help(df.iplot)
df.iplot(filename='Tutorial 1', world_readable=True)