# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:30:22 2016

@author: tongtz
"""

import numpy as np
import pandas as pd
import os

data = np.random.standard_normal((1000000,5)).round(5) # sample data set

path = os.path.join(os.path.dirname("__file__"), '/Development/Python3/Quants/data/')
filename = path + 'numbs'

import sqlite3 as sq3
query = 'CREATE TABLE numbers (No1 real, No2 real, No3 real, No4 real, No5 real)'

con = sq3.Connection(filename + '.db')
con.execute(query)

%%time
con.executemany('INSERT INTO numbers VALUES (?,?,?,?,?)', data)
con.commit()

%%time
temp = con.execute('SELECT * FROM numbers').fetchall()
print(temp[:2])
temp = 0.0

%%time
query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'
res = np.array(con.execute(query).fetchall()).round(3)

res = res[::100]

import matplotlib.pyplot as plt
plt.plot(res[:,0], res[:,1],'ro')
plt.grid(True);plt.xlim(-0.5,4.5);plt.ylim(-4.5,0.5)

import pandas.io.sql as pds

%time data = pds.read_sql('SELECT * FROM numbers', con)
con.close()

data.head()

%time data[(data['No1'] > 0) & (data['No2'] < 0)].head()
%time res = data[['No1','No2']][((data['No1'] > 0.5) | (data['No1'] < -0.5)) & ((data['No2'] < -1) | (data['No2'] > 1))]

plt.plot(res.No1, res.No2, 'ro')
plt.grid(True); plt.axis('tight')


# h5s
h5s = pd.HDFStore(filename + '.h5s', 'w')
%time h5s['data'] = data

h5s
h5s.close()

%%time
h5s = pd.HDFStore(filename + '.h5s', 'r')
%time temp = h5s['data']
h5s.close()

np.allclose(np.array(temp),np.array(data))

# CSV
%time data.to_csv(filename + '.csv')
%%time
pd.read_csv(filename + '.csv')[['No1','No2','No3','No4']].hist(bins=20);

# Excel
%time data[:10000].to_excel(filename + '.xlsx')
%time pd.read_excel(filename + '.xlsx', 'Sheet1').cumsum().plot()

%%time
data = pd.DataFrame(np.random.randint(0, 100, (1e6, 5)))
data = pd.merge(data, pd.DataFrame(np.random.standard_normal((1e6, 5))), left_index=True, right_index=True)

data.info()


from time import time

def benchmarking(lib):
    times = []
    sizes = []
    for c in range(10):
        t0 = time()
        name = path + 'data.h5c%s' % c
        h5 = pd.HDFStore(name, complevel=c, complib=lib)
        h5['data'] = data
        h5.close()
        times.append(time() - t0)
        sizes.append(os.path.getsize(name))
    
    return times, sizes
    
# function to plot the results
def plot_results(times, sizes):
    fig, axl = plt.subplots()
    plt.plot(range(10), times, 'r', lw=1.5, label='time')
    plt.xlabel('comp level')
    plt.ylabel('time [sec]')
    plt.legend(loc=0)
    ax2 = ax1.twinx()
    plt.plot(range(10), sizes, 'g', lw=1.5, label='size')
    plt.ylabel('file size [bytes]')
    plt.legend(loc=7)
    plt.grid(True)
    
times, sizes = benchmarking('blosc')
plot_results(times, sizes)
        









