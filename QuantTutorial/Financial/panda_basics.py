# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:27:52 2016

@author: tongtz
"""

import pandas.io.data as web
import pandas as pd
import numpy as np

DAX = web.DataReader(name='^GDAXI', data_source='yahoo', start='2000-1-1')
DAX.info()
DAX.tail()

DAX['Close'].plot(figsize=(8,5))

DAX['42d'] = pd.rolling_mean(DAX['Close'], window=42)
DAX['252d'] = pd.rolling_mean(DAX['Close'], window=252)

DAX[['Close','42d','252d']].tail()

DAX[['Close','42d','252d']].plot(figsize=(8,5))

# Calculating log returns for the index
DAX['Log Return'] = np.log(DAX['Close'] / DAX['Close'].shift(1))

DAX[['Close', 'Log Return']].tail()


