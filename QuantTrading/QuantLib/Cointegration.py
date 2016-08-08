import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as plt

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

to_plot = NKE[['bid','bdeptht']][(NKE.index > dt.datetime(2016,7,21,0,0)) & (NKE.index < dt.datetime(2016,7,22,0,0))]
to_plot.plot(subplots=True, style='b', figsize=(10,6))