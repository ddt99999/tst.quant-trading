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
    
'''
Chapter 4 - Collections

Summary
=======
One common application of for loop iterable processing is the
unwrap(process(wrap(iterable))) design pattern. A wrap() function will first
transform each item of an iterable into a two tuples with a derived sort key or other
value and then the original immutable item. We can then process these two tuples
based on the wrapped value. Finally, we'll use an unwrap() function to discard the
value used to wrap, which recovers the original item. 
This happens so often in a functional context that we have two functions that are
used heavily for this; they are as follows:

fst = lambda x: x[0]
snd = lambda x: x[1]

These two functions pick the first and second values from a tuple, and both are
handy for the process() and unwrap() functions.

Another common pattern is wrap(wrap(wrap())). In this case, we're starting
with simple tuples and then wrapping them with additional results to build
up larger and more complex tuples. A common variation on this theme is
extend(extend(extend())) where the additional values build new, more
complex namedtuple instances without actually wrapping the original tuples.
We can summarize both of these as the Accretion design pattern.

We'll apply the Accretion design to work with a simple sequence of latitude and
longitude values. The first step will convert the simple points (lat, lon) on a path
into pairs of legs (begin, end). Each pair in the result will be ((lat, lon), (lat, lon)).
In the next sections, we'll show how to create a generator function that will iterate over
the content of a file. This iterable will contain the raw input data that we will process.
Once we have the data, later sections will show how to decorate each leg with the
haversine distance along the leg. The final result of the wrap(wrap(iterable())))
processing will be a sequence of three tuples: ((lat, lon), (lat, lon), distance). We
can then analyze the results for the longest, shortest distance, bounding rectangle,
and other summaries of the data.
'''

import xml.etree.ElementTree as XML
import urllib

# Applying generator expressions to scalar functions
from math import radians, sin, cos, sqrt, asin

MI = 3959
NM = 3440
KM = 6371

def haversine(point1, point2, R=NM):
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(delta_lat/2)**2 + cos(lat1)*cos(lat2)*sin(delta_lon/2)**2
    c = 2*asin(sqrt(a))
    
    return R*c

def row_iter_kml(file_obj):
    ns_map = {
        "ns0": "http://www.opengis.net/kml/2.2",    
        "ns1": "http://www.google.com/kml/ext/2.2"    
    }
    doc = XML.parse(file_obj)
    return (comma_split(coordinates.text) 
            for coordinates in 
            doc.findall("./ns0:Document/ns0:Folder/ns0:Placemark/ns0:Point/ns0:coordinates", ns_map))
            
def comma_split(text):
    return text.split(",")
    
def pick_lat_lon(lon, lat, alt):
    return lat, lon
    
def lat_lon_kml(row_iter):
    return (pick_lat_lon(*row) for row in row_iter)
    
# parse the sequence recursively
def pairs(iterable):
    def pair_from(curr_item, iterable_tail):
        next_item = next(iterable_tail)
        yield curr_item, next_item
        yield from pair_from(next_item, iterable_tail)
    try:
        return pair_from(next(iterable), iterable)
    except StopIteration:
        return
        
def float_from_pair(lat_lon_iter):
    return ((float(lat), float(lon)) for lat, lon in lat_lon_iter)
        
# tail-call optimization
def legs(lat_lon_iter):
    begin = next(lat_lon_iter)
    for end in lat_lon_iter:
        yield begin, end
        begin = end
 
  
with urllib.request.urlopen("file:./Winter%202012-2013.kml") as source:
    flts = tuple(legs(float_from_pair(lat_lon_kml(row_iter_kml(source))))) 
    print("FLOATS")
    print(flts)
    source.seek(0)
    print("STRINGS")
    v1 = tuple(lat_lon_kml(row_iter_kml(source)))
    print(v1)
    source.seek(0)
    trip = ((start, end, round(haversine(start, end), 4)) 
            for start, end in legs(float_from_pair(lat_lon_kml(row_iter_kml(source)))))


for start, end, dist in trip:
    print(start, end, dist)    

n = 10
flat = ['2', '3', '5', '7', '11', '13', '17', '19', '23', '29', '31', '37', '41', '43', '47', '53', '59', '61', '67', '71' ]
for elem in range(n):
    print(flat[elem::n])
test = list(zip(*(flat[i::n] for i in range(n))))

# Using reversed() to change the order
def digits(x, b):
    if x == 0:
        return
    yield x % b
    yield from digits(x//b,b)
#    for d in to_base(x//b, b):
#        yield d
        
def to_base(x, b):
    return digits(x, b)
    
test = reversed(list((digits(12,2))))
    
for elem in test:
    print(elem)