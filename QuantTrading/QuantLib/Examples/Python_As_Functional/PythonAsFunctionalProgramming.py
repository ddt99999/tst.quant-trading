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
n = 22
is_primes = lambda n: not any(n%p == 0 for p in range(2, 1+int(math.sqrt(n))))


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
    #flts = tuple(legs(float_from_pair(lat_lon_kml(row_iter_kml(source))))) 
    #print("FLOATS")
    #print(flts)
    #source.seek(0)
    #print("STRINGS")
    #v1 = tuple(lat_lon_kml(row_iter_kml(source)))
    #print(v1)
    #source.seek(0)
    trip = list(((start, end, round(haversine(start, end), 4)) 
            for start, end in legs(float_from_pair(lat_lon_kml(row_iter_kml(source))))))

    


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
    
'''
CHAPTER 5 - High Order Functions

As we can see, there are three varieties of higher-order functions, which are
as follows:
• Functions that accept a function as one of its arguments
• Functions that return a function
• Functions that accept a function and return a function
Python offers several higher-order functions of the first variety. We'll look at these built-in higher-order functions in this chapter. 
We'll look at a few of the library modules that offer higher-order functions in later chapters.
'''

'''
We have three ways of getting the maximum and minimum distances from this
sequence of values. They are as follows:
• Extract the distance with a generator function. This will give us only the distances, as we've discarded the other two attributes of each leg. 
  This won't work out well if we have any additional processing requirements.
• Use the unwrap(process(wrap())) pattern. This will give us the legs with the longest and shortest distances. From these, we can extract 
  just the distance, if that's all that's needed. The other two will give us the leg that contains the maximum and minimum distances.
• Use the max() and min() functions as higher-order functions.
'''

# First method - not good as trip has to be called 
long, short = max(dist for start, end, dist in trip), min(dist for start, end, dist in trip)
print(long, short)

# Second method - unwrap(process(wrap())) pattern
def wrap(iterable):
    return ((leg[2], leg) for leg in iterable)
    
def unwrap(dist_leg):
    distance, leg = dist_leg
    return leg
    
long, short = unwrap(max(wrap(trip))), unwrap(min(wrap(trip)))

# Third method
def by_dist(leg):
    lat, lon, dist = leg
    return dist
    
long, short = max(trip, key=by_dist), min(trip, key=by_dist)
print(long, short)

# Using the map() function to apply a function to a collection
text = """\
... 2 3 5 7 11 13 17 19 23 29
... 31 37 41 43 47 53 59 61 67 71
... 73 79 83 89 97 101 103 107 109 113
... 127 131 137 139 149 151 157 163 167 173
... 179 181 191 193 197 199 211 223 227 229
... """

chars = list(v for line in text.splitlines() for v in line.split())
data_int = list(map(int, chars))

#map(lambda x: (start(x),end(x),dist(x)*6076.12/5280), trip)

with urllib.request.urlopen("file:./Winter%202012-2013.kml") as source:
    path = tuple(float_from_pair(lat_lon_kml(row_iter_kml(source))))
    trip = map(lambda start_end: (start_end[0], start_end[1], haversine(*start_end)), zip(path, path[1:]))
    distances2= map(lambda s, e: (s, e, haversine(s, e)), path, path[1:])
    
long = list(filter(lambda leg: by_dist(leg) >= 50, trip))

sum_ans = sum(filter(lambda x: x%3==0 or x%5==0, range(10)))

primes = list(filter(is_primes, range(2,100)))

# Using filter() to identify outliers
dist_data = list(map(by_dist, trip))

tail = iter([1,2,3,None,4,5,6].pop, None)
list(tail)

# Building higher-irder functions with Callables

def group_by_iter(n, iterable):
    n_rows = tuple(next(iterable) for i in range(n))
    while n_rows:
        yield n_rows
        n_rows = tuple(next(iterable) for i in range(n))
        
group_vals = list(group_by_iter(7, filter(lambda x: x%3==0 or x%5==0, range(100))))

from collections.abc import Callable

class NullAware(Callable):
    def __init__(self, func):
        self.func = func
        
    def __call__(self, arg):
        return None if arg is None else self.func(arg)
        
null_log_scale = NullAware(math.log)       

data = [10,100,None,50,60]
scaled = list(map(null_log_scale, data))

class Sum_Filter(Callable):
    __slots__ = ["filter", "function"]
    def __init__(self, filter, function):
        self.filter = filter
        self.function = function
    
    def __call__(self, iterable):
        return sum(self.function(x) for x in iterable if self.filter(x))
        
count_not_none = Sum_Filter(lambda x: x is not None, lambda x: 1)
data = [1, 2, 4, None, 3, 4, 1, None]
count_data = count_not_none(data)

'''
CHAPTER 6 - Recursive and Reduction

'''

def fibonacci(n):
    f = [0]*(n+1)
    
    f[0] = 0
    f[1] = 1
    f[2] = 1
    
    for i in range(3,n+1):
        f[i] = f[i-1] + f[i-2]
    
    return f, f[n]
    
f = []    
f, fib = fibonacci(15)

# Processing collections via recursion

'''
We've defined the mapping of a function to an empty collection as an empty sequence. We've also
specified recursively with a three step expression. First, apply the function to all of the 
collection except the last element, creating a sequence object. Then apply the function
to the last element. Finally, append the last calculation to the previously built sequence
'''
def mapr(f, collection):
    if len(collection) == 0:
        return []
    return mapr(f, collection[:-1]) + [f(collection[:-1])]
    
# Tail-call optimization
def mapf(f, collection):
    return (f(x) for x in collection)

def mapg(f, collection):
    for x in collection:
        yield f(x)


# Group-by reductions - from many to fewer
quantized = (5*(dist//5) for start, stop, dist in trip)

# Building a mapping with Counter
from collections import Counter
Counter(quantized)

Counter(quantized).most_common()

# Building a mapping by sorting
def group_sort(trip):
    def group(data):
        sorted_data= iter(sorted(data))
        previous, count = next(sorted_data), 1
        for d in sorted_data:
            if d == previous:
                count += 1
            elif previous is not None: # and d != previous
                yield previous, count
                previous, count = d, 1 # reset the count for new d
            else:
                raise Exception("Bad design problem")
        yield previous, count
    quantized = (5*(dist//5) for start, stop, dist in trip)
    return dict(group(quantized))

data = group_sort(trip)  

# Grouping or partitioning data by key values
'''
There are no limits to the kinds of reductions we might want to apply to grouped
data. We might have data with a number of independent and dependent variables.
We can consider partitioning the data by an independent variable and computing
summaries like maximum, minimum, average, and standard deviation of the values
in each partition.

The essential trick to doing more sophisticated reductions is to collect all of the data
values into each group. The Counter() function merely collects counts of identical
items. We want to create sequences of the original items based on a key value.
Looked at in a more general way, each 5-mile bin will contain the entire collection of
legs of that distance, not merely a count of the legs. We can consider the partitioning
as a recursion or as a stateful application of defaultdict(list) object. We'll look at
the recursive definition of a groupby() function, since it's easy to design.

Clearly, the groupby(C, key) method for an empty collection, C, is the empty
dictionary, dict(). Or, more usefully, the empty defaultdict(list) object.
For a non-empty collection, we need to work with item C[0], the head, and
recursively process sequence C[1:], the tail. We can use head, *tail = C
command to do this parsing of the collection, as follows:
'''
import collections

C = [1,2,3,4,5]
head, *tail = C

def group_by(key, data):
    def group_into(key, collection, dictionary):
        if len(list(collection)) == 0:
            return dictionary
        print(list(collection))
        head, *tail = collection
        print(head)
        dictionary[key(head)].append(head)
        return group_into(key, tail, dictionary)
        
    return group_into(key, data, collections.defaultdict(list))

binned_distance = lambda leg: 5*(leg[2]//5)
by_distance = group_by(binned_distance, trip)

import pprint
for distance in sorted(by_distance):
    print(distance)
    pprint.pprint(by_distance[distance])

def partition(key, data):
    dictionary = collections.defaultdict(list)
    for head in data:
        dictionary[key(head)].append(head)
    return dictionary
    
# Writing more general group-by reductions

start   = lambda s, e, d: s
end     = lambda s, e, d: e
dist    = lambda s, e, d: d
latitude = lambda lat, lon: lat
longitude = lambda lat, lon: lon

point = ((35.505665, -76.653664), (35.508335, -76.654999), 0.1731)

start(*point)
latitude(*start(*point))

for distance in sorted(by_distance):
    print(distance, max(by_distance[distance], key=lambda pt: latitude(*start(*point))))
    
# Writing high-order reductions

def s0(data):
    return sum(1 for x in data) # or len(data)

def s1(data):
    return sum(x for x in data) # sum(data)

def s2(data):
    return sum(x*x for x in data)
    
def sum_f(function, data):
    return sum(function(x) for x in data)
    
N= sum_f(lambda x: 1, data) # x**0
S= sum_f(lambda x: x, data) # x**1
S2= sum_f( lambda x: x*x, data ) # x**2

def sum_filter_f(filter, function, data):
    return sum(function(x) for x in data if filter(x))
    
count_= lambda x: 1
sum_ = lambda x: x
valid = lambda x: x is not None
N = sum_filter_f(valid, count_, data)

# Writing file parsers
import re  # regular expression module

Color = namedtuple("Color", ("r", "g", "b", "name"))

def row_iter_gpl(file_obj):
    header_pattern = re.compile(r"GIMP Palette\nName:\s*(.*?)\nColumns:\s*(.*?)\n#\n", re.M)
    
    def read_head(file_obj):
        match = header_pattern.match("".join(file_obj.readline() for _ in range(4)))
        return (match.group(1), match.group(2)), file_obj
        
    def read_tail(headers, file_obj):
        return headers, (next_line.split() for next_line in file_obj)
    
    return read_tail(*read_head(file_obj))
    
def color_palette(headers, row_iter):
    name, columns = headers
    colors = tuple(Color(int(r), int(g), int(b), " ".join(name)) for r,g,b,*name in row_iter)
    return name, columns, colors
    
with open("crayola.gpl") as source:
    name, columns, colors = color_palette(*row_iter_gpl(source))
    print(name, columns)
    for color in colors:
        print(color.r, color.g, color.b, color.name)
        
        
'''
CHAPTER 7 - Additional Tuple Techniques

Using an immutable namedtuple as a record

We can do any of the following, depending on the circumstances:
• Use lambdas (or functions) to select a named item using the index
• Use lambdas (or functions) with *parameter to select an item by parameter name, which maps to an index
• Use namedtuples to select an item by attribute name or index

'''

# These definitions don't provide much guidance on the data types involved. We can
# use a simple naming convention to make this a bit more clear. The following are
# some examples of selection functions that use a suffix:
start_point = lambda leg: leg[0]
distance_nm= lambda leg: leg[2]
latitude_value= lambda point: point[0]

# When used judiciously, this can be helpful. It can also degenerate into an elaborately
# complex Hungarian notation as a prefix (or suffix) of each variable.
# The second technique uses the *parameter notation to conceal some details of the
# index positions. The following are some selection functions that use the * notation:
    
start= lambda start, end, distance: start
end= lambda start, end, distance: end
distance= lambda start, end, distance: distance
latitude= lambda lat, lon: lat
longitude= lambda lat, lon: lon

# The third technique is the namedtuple function. In this case, we have nested
# namedtuple functions such as the following:
Leg = namedtuple("Leg", ("start", "end", "distance"))
Point = namedtuple("Point", ("latitude", "longitude"))

# Avoiding stateful classes by using families of tuples

def row_iter( source ):
    """Read a CSV file and emit a sequence of rows.

    >>> import io
    >>> data= io.StringIO( "1\\t2\\t3\\n4\\t5\\t6\\n" )
    >>> list(row_iter(data))
    [['1', '2', '3'], ['4', '5', '6']]
    """
    rdr= csv.reader( source, delimiter="\t" )
    return rdr

def float_none( data ):
    """Float conversion: return None instead of ValueError exception.

    >>> float_none('abc')
    >>> float_none('1.23')
    1.23
    """
    try:
        data_f= float(data)
        return data_f
    except ValueError:
        return None

def head_map_filter( row_iter ):
    """Removing headers by applying a filter to get rows with 8 values.

    >>> rows= [ ["Anscombe's quartet"], ['I', 'II', 'III', 'IV'], ['x','y','x','y','x','y','x','y'], ['1','2','3','4','5','6','7','8']]
    >>> list(head_map_filter( rows ))
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    """
    float_row = lambda row: list(map(float_none, row))

    all_numeric = lambda row: all(row) and len(row) == 8

    return filter(all_numeric, map(float_row, row_iter))

with open("Anscombe.txt") as source:
    data = tuple(head_map_filter(row_iter(source)))
    series_I= tuple(series(0,data))
    series_II= tuple(series(1,data))
    series_III= tuple(series(2,data))
    series_IV= tuple(series(3,data))
    
y_rank= tuple(enumerate(sorted(series_I, key=lambda p: p.y)))
xy_rank= tuple(enumerate(sorted(y_rank, key=lambda rank: rank[1].x)))

x_rank = lambda ranked: ranked[0]
y_rank= lambda ranked: ranked[1][0]
raw = lambda ranked: ranked[1][1]