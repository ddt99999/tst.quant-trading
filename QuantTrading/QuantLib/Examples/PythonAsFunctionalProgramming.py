# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:19:57 2016

@author: tongtz
"""

import math

def numbers():
    for i in range(1024):
        print("=", i)
        yield i 
        
numbers()

def sum_to(n):
    sums=0
    for i in numbers():
        if i == n:
            break
        sums += i
        
    return sums
    
sum_to(5)

# To get prime numbers
# prime(n) = for all x [(2 <= x < 1+ sqrt(n)) and (n(mod x) != 0)]
n = 0
primes = not any(n%p == 0 for p in range(2, 1+int(math.sqrt(n))))


'''
Chapter 3 - Functions,Iterators, and Generators
'''
import collections

class Mersennel(collections.Callable):
    def __init__(self, algorithm):
        self.pow2 = algorithm
        
    def __call__(self, arg):
        return self.pow2(arg) - 1
        
def shifty(b):
    return 1 << b
    
def multy(b):
    if b == 0: return 1
    return 2*multy(b-1)
    
def faster(b):
    if b == 0: 
        return 1
    if b%2==1:
        return 2*faster(b-1)
    t = faster(b//2)
    return t*t
    
mls = Mersennel(shifty)
mlm = Mersennel(multy)
mlf = Mersennel(faster)

ans = mls.pow2(4)

def remove(str, chars):
    if chars:
        return remove(str.replace(chars[0], ""), chars[1:])
    return str
    
test = remove("$14,555,222.00", "$,")

# using a generator
def pfactorsr(x):
    def factor_n(x, n):
        if n*n > x:
            yield x
            return 
        if x % n == 0:
            yield n
            if x//n > 1:
                yield from factor_n(x//n,n)
        else:
            yield from factor_n(x, n+2)
    
    if x % 2 == 0:
        yield 2
        if x//2 > 1:
            yield from pfactorsr(x//2)
        return
        
    yield from factor_n(x,3)
    
for elem in pfactorsr(1000):
    print(elem)
    
# Cleaning raw data with generator function
import csv

def row_iter(source):
    return csv.reader(source, delimiter="\t")
    
def head_split_fixed(row_iter):
    title = next(row_iter)
    assert len(title) == 1 and title[0] == "Anscombe's quartet"
    heading= next(row_iter)
    assert len(heading) == 4 and heading == ['I', 'II', 'III', 'IV']
    columns= next(row_iter)
    assert len(columns) == 8 and columns == ['x', 'y', 'x', 'y', 'x',
    'y', 'x', 'y']
    return row_iter
    
with open("Anscombe.txt") as source:
    data = head_split_fixed(row_iter(source))
    print(list(data))
    
from collections import namedtuple

pair = namedtuple("pair", ("x", "y"))
def series(n, row_iter):
    for row in row_iter:
        yield pair(*row[n*2:n*2+2])

with open("Anscombe.txt") as source:
    data = tuple(head_split_fixed(row_iter(source)))
    sample_I= tuple(series(0,data))
    sample_II= tuple(series(1,data))
    sample_III= tuple(series(2,data))
    sample_IV= tuple(series(3,data))
    
mean = sum(float(pair.y) for pair in sample_I)/len(sample_I)

for subset in sample_I, sample_II, sample_III, sample_III:
    print(len(subset))
    mean = sum(float(pair.y) for pair in subset)/len(subset)
    print(mean)