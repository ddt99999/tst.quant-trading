# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:08:40 2016

@author: tongtz
"""

from math import *
import numpy as np
import numexpr as ne
import perf_comp as pc

def f(x):
    return abs(cos(x)) ** 0.5 + sin(2 + 3 * x)
    
I = 500000
a_py = range(I)

def f1(a):
    res = []
    for x in a:
        res.append(f(x))
    return res
    
def f2(a):
    return [f(x) for x in a]
    
def f3(a):
    ex = 'abs(cos(x)) ** 0.5 + sin(2 + 3 * x)'
    return [eval(ex) for x in a]


a_np = np.arange(I)

def f4(a):
    return (np.abs(np.cos(a)) ** 0.5 + np.sin(2 + 3 * a))
    


def f5(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(1)
    return ne.evaluate(ex)
    
def f6(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(4)
    return ne.evaluate(ex)

def f7(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(8)
    return ne.evaluate(ex)
    
func_list = ['f1','f2','f3','f4','f5','f6','f7']
data_list = ['a_py','a_py','a_py','a_np','a_np','a_np','a_np']



pc.perf_comp_data(func_list, data_list, rep=1)

np.zeros((3,3), dtype=np.float64, order='C')

# Consider the C-like, i.e. row-wise, storage

c = np.array([[1., 1., 1.],
              [2., 2., 2.],
              [3., 3., 3.]], order='C')
             
# In this case, the 1s, the 2s and the 3s are stored next to each other
  
x = np.random.standard_normal((3,150000))
C = np.array(x, order='C')
F = np.array(x, order='F')
x = 0.0           

%timeit C.sum(axis=0)
%timeit C.sum(axis=1)

%timeit C.std(axis=0)
%timeit C.std(axis=1)

%timeit F.sum(axis=0)
%timeit F.sum(axis=1)

%timeit F.std(axis=0)
%timeit F.std(axis=1)

# Sometimes it is, however, helpful to parallelize code execution locally. 
# Here, the multiprocessing module of Python might prove beneficial

import multiprocessing as mp
import math

def simulate_geometric_brownian_motion(p):
    M, I = p # M - no. discretization, I = no of simulated path
    # time steps, paths
    S0 = 100; r = 0.05; sigma = 0.2; T = 1.0
    # model parameters
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    
    for t in range(1, M + 1):
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * np.random.standard_normal(I))
    return paths
    
paths = simulate_geometric_brownian_motion((5, 2))
paths

I = 10000   # number of paths
M = 10      # number of time steps
t = 32      # number of tasks/simulations

# running on machine with 24 cores/threads
from time import time
times = []
  
for w in range(1, 5):
    t0 = time()
    pool = mp.Pool(processes=w) # the pool of workers
    result = pool.map(simulate_geometric_brownian_motion, t * [(M, I), ]) # the mapping of the function to the list of parameter tuples
    times.append(time() - t0)
    
times  

import matplotlib.pyplot as plt
plt.plot(range(1,5), times)
plt.plot(range(1,5), times, 'ro')  
plt.grid(True)
plt.xlabel('number of processes')
plt.ylabel('time in seconds')
plt.title('%d Monte Carlo simulations' % t)

# Numba is an open source, Numpy-aware optimzing compiler for Python (cf. http://numba.pydata.org). It uses the LLVM compiler infrastructure. http://www.llvm.org

from math import cos, log

def f_py(I, J):
    res = 0
    for i in range(I):
        for j in range(J):
            res += int(cos(log(1)))
    return res
    
I, J = 500, 500
%time f_py(I, J)

def f_np(I, J):
    a = np.ones((I,J), dtype=np.float64)
    return int(np.sum(np.cos(np.log(a)))), a
%time res, a = f_np(I,J)

import numba as nb

f_nb = nb.jit(f_py)

func_list = ['f_py', 'f_np', 'f_nb']
data_list = 3 * ['I, J']

pc.perf_comp_data(func_list, data_list, rep=1)