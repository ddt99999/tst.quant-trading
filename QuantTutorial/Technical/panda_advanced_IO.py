# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:19:11 2016

@author: tongtz
"""

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from random import gauss
import os
import pickle

path = os.path.join(os.path.dirname("__file__"), '/Development/Python3/Quants/data/')
filename = path + 'numbs'

a = [gauss(1.5, 2) for i in range(1000000)]

pkl_file = open(path + 'data.pkl', 'wb')

%time pickle.dump(a, pkl_file)

pkl_file

pkl_file.close()

pkl_file = open(path + 'data.pkl', 'rb')
%time b = pickle.load(pkl_file)

b[:5]

np.allclose(np.array(a),np.array(b))

pkl_file = open(path + 'data.pkl', 'wb')
%time pickle.dump(np.array(a), pkl_file)
%time pickle.dump(np.array(a) ** 2, pkl_file)

pkl_file.close()

pkl_file = open(path + 'data.pkl', 'rb')
x = pickle.load(pkl_file)
x
y = pickle.load(pkl_file)
y
pkl_file.close()

# pickle stores object according to the first in, first out (FIFO) principle
pkl_file = open(path + 'data.pkl', 'wb')
pickle.dump({'x':x, 'y':y}, pkl_file)
pkl_file.close()

pkl_file = open(path + 'data.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
for key in data.keys():
    print(key, data[key][:4])
    
rows = 5000
a = np.random.standard_normal((rows, 5))
a.round(4)

t = pd.date_range(start='2014/1/1', periods=rows, freq='H')
t

csv_file = open(path + 'data.csv', 'w')
header = 'date,no1,no2,no3,no4,no5\n'
csv_file.write(header)

for t_, (no1,no2,no3,no4,no5) in zip(t, a):
    s = '%s,%f,%f,%f,%f,%f\n' % (t_, no1, no2, no3, no4, no5)
    csv_file.write(s)
    
csv_file.close()


csv_file = open(path + 'data.csv', 'r')

for i in range(5):
    print(csv_file.readline())
    
content = csv_file.readlines()
for line in content[:5]:
    print(line)
    
#### Numpy IO ####
dtimes = np.arange('2015-01-01 10:00:00', '2021-12-31 22:00:00', dtype='datetime64[m]') # minute intervals
len(dtimes)    

dty = np.dtype([('Date', 'datetime64[m]'), ('No1','f'), ('No2','f')])
data = np.zeros(len(dtimes), dtype=dty)

data['Date'] = dtimes
a = np.random.standard_normal((len(dtimes),2)).round(5)
data['No1'] = a[:,0]
data['No2'] = a[:,1]

%time np.save(path + 'array', data) ## suffix .npy is added

%time np.load(path + 'array.npy')

## compare with pickle, numpy IO on binary seems faster
pkl_file = open(path + 'data_test.pkl', 'wb')

%time pickle.dump(data, pkl_file)

pkl_file.close()

# try with larger dataset
%time data = np.random.standard_normal((10000, 6000))
%time np.save(path + 'array', data)

%time np.load(path + "array.npy")

h5s = pd.HDFStore(filename + '.h5s', 'w')
%time h5s['data'] = data

# Panda Web IO
from pandas_datareader import data as web

symbols = ['AAPL','MSFT','YHOO']
data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, start='2005-1-1', data_source="yahoo")['Adj Close']

data.plot(figsize=(10,6))
h5 = pd.HDFStore(path + 'data_web.h5','w')
h5['data'] = data
h5.close()

data.to_excel(path + 'data.xlsx')

import tables as tb
import datetime as dt
import matplotlib.pyplot as plt

filename = path + 'tab.h5'
h5 = tb.open_file(filename, 'w')
rows = 2000000

row_des = {
    'Date': tb.StringCol(26, pos=1),
    'No1': tb.IntCol(pos=2),
    'No2': tb.IntCol(pos=3),
    'No3': tb.Float64Col(pos=4),
    'No4': tb.Float64Col(pos=5)
}

# when creating the table, we choose no compression for the moment
filters = tb.Filters(complevel=0) # no compression
tab = h5.create_table('/','ints_floats', row_des,title='Integers and Floats', expectedrows=rows, filters=filters)
tab 

random_int = np.random.randint(0, 10000, size=(rows,2))
random_float = np.random.standard_normal((rows,2)).round(5)

pointer = tab.row
for i in range(rows):
    pointer['Date'] = dt.datetime.now()
    pointer['No1'] = random_int[i,0]
    pointer['No2'] = random_int[i,1]
    pointer['No3'] = random_float[i,0]
    pointer['No3'] = random_float[i,1]
    pointer.append() # this appends the data and moves the pointer one row forward
    
tab.flush()

tab

# There is a more performant and Pythonic way to accomlish the same result: by the use of Numpy structured arrays
dty = np.dtype([('Date','S26'),('No1','<i4'),('No2','<i4'),('No3','<f8'),('No4','<f8')])
sarray = np.zeros(len(random_int), dtype=dty)
sarray

%%time
sarray['Date'] = dt.datetime.now()
sarray['No1'] = random_int[:,0]
sarray['No2'] = random_int[:,1]
sarray['No3'] = random_float[:,0]
sarray['No4'] = random_float[:,1]

%time h5.create_table('/','ints_floats_from_array', sarray, title='Integers and Floats', expectedrows=rows, filters=filters)

h5
h5.remove_node('/','ints_floats_from_array')

tab[:3]
tab[:4]['No4']
h5

%time np.sum(tab[:]['No3'])
%time np.sum(np.sqrt(tab[:]['No3']))

%%time
plt.hist(tab[:]['No3'],bins=30)
plt.grid(True)
print(len(tab[:]['No3']))

# PyTables is also able to perform (out-of-memory) analytics of the types seen before

%%time
res = np.array([(row['No3'],row['No4']) for row in tab.where('((No3 < -0.5) | (No3 > 0.5)) & ((No4 < -1) | (No4 > 1))')])[::100]

plt.plot(res.T[0],res.T[1],'ro')
plt.grid(True)

%%time
values = tab.cols.No3[:]
print("Max %18.3f" % values.max())
print("Ave %18.3f" % values.mean())
print("Min %18.3f" % values.min())
print("Std %18.3f" % values.std())

filename = path + 'tab.h5c'
h5c = tb.open_file(filename,'w')

filters = tb.Filters(complevel=4, complib='blosc') # 
tabc = h5c.create_table('/', 'ints_floats', sarray, title='Integers and Floats', expectedrows=rows, filters=filters)
h5c


%%time
res = np.array([(row['No3'],row['No4']) for row in tab.where('((No3 < -0.5) | (No3 > 0.5)) & ((No4 < -1) | (No4 > 1))')])[::100]

%time arr_non = tab.read()
%time arr_com = tabc.read()


# working with array
%%time
array_int = h5.create_array('/','integers', random_int)
array_float = h5.create_array('/','floats', random_float)

h5

filename = path + 'array.h5'
h5 = tb.open_file(filename, 'w')

n = 100
ear = h5.create_earray(h5.root, 'ear', atom=tb.Float64Atom(), shape=(0,n))

%%time
rand = np.random.standard_normal((n,n))
for _ in range(750):
    ear.append(rand)
ear.flush()

ear
ear.size_on_disk
out = h5.create_earray(h5.root, 'out', atom=tb.Float64Atom(), shape=(0,n))

# the numerical expression as a string object
expr = tb.Expr('3 * sin(ear) + sqrt(abs(ear))')
# target to store results is disk-based array
expr.set_output(out, append_mode=True)

%time expr.eval() # evaluation of the numerical expression and storage of results in disk-based array

out[0, :10]

%time imarray = ear.read() # read whole array into memory

import numexpr as ne
expr = '3 * sin(imarray) + sqrt(abs(imarray))'

ne.set_num_threads(1)
%time ne.evaluate(expr)[0,:10]

ne.set_num_threads(4)
%time ne.evaluate(expr)[0,:10]






#h5.close()
#h5c.close()
