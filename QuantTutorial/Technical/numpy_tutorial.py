# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:15:02 2016

@author: tongtz
"""

import numpy as np

# Note: Python indexes start in 0 and never include the last one.
arr1 = np.arange(2, 20, 2)
print(arr1)

sl1 = arr1[2:6]
print(sl1)

# To select an entry in a multidimensional array, the corresponding index should be written as a comma-separated list:
matr1 = np.arange(9).reshape(3,3)
print(matr1)
print('element in the row 1 and column 2 is: \n', matr1[1, 2])

lnarr1 = np.arange(0, -13, -2)
print(lnarr1)

subarr1 = lnarr1[[2,6]]
print("Elements in positions 2 and 6: ", subarr1)

## Sublists out of multidimensional arrays
matr1 = np.arange(9).reshape(3,3)
print(matr1)

# Elements in positions (1,0), (2, 2) and (1, 2): 
submatr = matr1[ [1, 2, 1], [0, 2, 2] ]
print("Elements in positions (1,0), (2, 2) and (1, 2): \n", submatr)

arr1 = np.array([-2, 0, 41, 100, 11, 13, 18])
print("original array: ", arr1)

# Slice from 1 to 4, if "set" is omitted it's assumed to be 1
print("Slice, start=1, stop=4, step omitted", arr1[1:4])
#Slice, start=1, stop=4, step omitted [  0  41 100]

# Backwards slice, the "step" is -2
print("Slice, start=5, stop=0, step=-4", arr1[5:0:-4])
#Slice, start=5, stop=1, step=-4 [13  0]

# If "stop" is omitted the default is the last index
# If "start" is omitted the default is the first index
print("Slice, start=0, stop omitted, step=2", arr1[0::2])

mtr1 = np.arange(8, 56, 2).reshape(6,4)
print("Array: \n", mtr1)

print("Slice: \n", mtr1[1::2, 0:3]) # mtr1[row range, 0 <= column range < 3]
#Slice: 
# [[16 18]
# [32 34]
# [48 50]]

print("Slice: \n", mtr1[ [1, 5], 1:])
#Slice: 
# [[18 20 22]
# [50 52 54]]

# Array masking in python works using an array of boolean values
arr1 = np.arange(1, 13, 2).reshape(2,3)
print("First array \n", arr1)
 
mask1 = np.array([True, False, False, True, True, False]).reshape(2,3)
print("Mask \n", mask1)
 
arr2 = arr1[mask1]
print("Mask over array \n", arr2)
#First array 
# [[ 1  3  5]
# [ 7  9 11]]
#Mask
# [[ True False False]
# [ True  True False]]
#Mask over array 
# [1 7 9]

arr1 = np.arange(-10, 10).reshape(5,4)
print('Array of numbers', arr1, sep='\n')
 
mask1 = arr1%3 == 0
print('Mask, True for those divisble by 3', mask1, sep='\n')
 
arr2 = arr1[mask1]
print('Putting the mask over the array', arr2, sep='\n')
#Output
#Array of numbers
#[[-10  -9  -8  -7]
# [ -6  -5  -4  -3]
# [ -2  -1   0   1]
# [  2   3   4   5]
# [  6   7   8   9]]
#Mask, True for those divisble by 3
#[[False  True False False]
# [ True False False  True]
# [False False  True False]
# [False  True False False]
# [ True False False  True]]
#Putting the mask over the array
#[-9 -6 -3  0  3  6  9]

mask1 = (arr1%3 == 0) & (arr1 < 0)
print('Mask, True for those divisble by 3 and less than 0', mask1, sep='\n')
 
arr2 = arr1[mask1]
print('Putting the mask over the array', arr2, sep='\n')
#Mask, True for those divisble by 3 and less than 0
#[[False  True False False]
# [ True False False  True]
# [False False False False]
# [False False False False]
# [False False False False]]
#Putting the mask over the array
#[-9 -6 -3]

A = np.arange(-10, 8, 2).reshape(3, 3)
print('array \n', A)
 
mask = A % 3 == 0
print('mask \n', mask)

A[mask] = A[mask]**2
print('square numbers divisible by 3 \n', A)
#mask 
# [[False False  True]
# [False False  True]
# [False False  True]]
#square numbers divisible by 3 
# [[-10  -8  36]
# [ -4  -2   0]
# [  2   4  36]]

#Efficient Masking
#There's a problem with the masking method to change values in a Numpy array presented in the 
#last example of the previous section, for large arrays the system will quickly run out of memory 
#because the mask has to be allocated. This is why Numpy has the putmask() routine.
np.putmask(A, A % 3 == 0, A**2)
print('square numbers divisible by 3 \n', A)

#Retrieving Indexes
A = np.arange(-10, 10, 2)
print('array \n', A)
 
mask1 = A % 3 == 0
print('mask \n', mask1)
 
idxs = np.where(mask1)
print('indexes \n', idxs)

#The function where returns a tuple of arrays containing the indexes, one array for each dimension
A = np.arange(-10, 8, 2).reshape(3, 3)
print('array \n', A)
 
idx1, idx2 = np.where(A % 3 == 0)
print('indexes \n', idx1, idx2)
#array 
# [[-10  -8  -6]
# [ -4  -2   0]
# [  2   4   6]]
#indexes 
# [0 1 2] [2 2 2]

print('Check if any of the elements is divisible by 3 \n', np.any(A % 3 == 0))
print('Check if all the elements are divisible by 2 \n', np.all(A % 2 == 0))

# NUMPY ARITHMETIC
# Url: https://www.getdatajoy.com/learn/Numpy_Arithmetic
# ================
a = np.arange(4)
print('first array: \n', a)

b = np.array([1+1j, 2, 4+2j, -1+12j])
print('second array: \n', b)

c = a*b
print('their multiplication: \n', c)
# first array: 
# [0 1 2 3]
# second array: 
# [ 1. +1.j  2. +0.j  4. +2.j -1.+12.j]
# their multiplication: 
# [ 0. +0.j  2. +0.j  8. +4.j -3.+36.j]

# basic arithmetics
a = np.arange(9).reshape(3,3)
print('first array: \n', a)

b = np.linspace(-4,-20,9,dtype=np.int).reshape(3,3)
print('second array: \n', b)

c = b**a - a*b
print('The result of the operation: \n', c)

#Broadcasting
#Even if two numpy array don't have the same shape they still can be compatible and some arithmetic operations can be performed. 
a = np.arange(9).reshape(3,3)
print('starting array: \n', a)
 
b1 = 3*a
print('array multiplied by 3: \n', b1)
 
b2 = a/2
print('array divided by 2: \n', b2)
 
# even cumulative operations will work
a += 2
print('the originial array plus 2: \n', a)
#starting array: 
# [[0 1 2]
# [3 4 5]
# [6 7 8]]
#array multiplied by 3: 
# [[ 0  3  6]
# [ 9 12 15]
# [18 21 24]]
#array divided by 2: 
# [[ 0.   0.5  1. ]
# [ 1.5  2.   2.5]
# [ 3.   3.5  4. ]]
#the originial array plus 2: 
# [[ 2  3  4]
# [ 5  6  7]
# [ 8  9 10]]

a = np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])
print('array a =\n', a)
b = np.array([3, 5, 1])
print('array b =\n', b)
 
c = a + b
print('a + b = \n', c)
#array a =
# [[ 1.1]
# [ 2.2]
# [ 3.3]
# [ 4.4]
# [ 5.5]]
#array b =
# [3 5 1]
#a + b = 
# [[  4.1   6.1   2.1]
# [  5.2   7.2   3.2]
# [  6.3   8.3   4.3]
# [  7.4   9.4   5.4]
# [  8.5  10.5   6.5]]

#Adding a New Axis
#The worst case scenario is when the two arrays we are working with don't coincide in any dimension 
#but we still want to operate them somehow. To overcome this we can take broadcasting to the limit and add a new axis.
a = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
b = np.array([3, 5, 1])
c = a[:, np.newaxis] + b
print(c)
#[[  4.1   6.1   2.1]
# [  5.2   7.2   3.2]
# [  6.3   8.3   4.3]
# [  7.4   9.4   5.4]
# [  8.5  10.5   6.5]]

# RANDOM NUMBERS
# ==============
# Numpy provides several methods to generate random numbers, floats and integers, from many well known probability distributions.
n1 = np.random.rand()
print(n1)

arr1 = np.random.rand(3, 4)
print(arr1)

arr1 = np.random.random_sample((3, 2))
print(arr1)
 
arr2 = np.random.random((3, 2))
print('\n', arr2)
 
arr3 = np.random.ranf((3, 2))
print('\n', arr3)
 
arr3 = np.random.sample((3, 2))
print('\n', arr3)

# Integar random number
num1 = np.random.randint(5)
print(num1)
 
num2 = np.random.randint(5, 18)
print('\n', num2)
 
arr1 = np.random.randint(5, 18, (3, 4))
print('\n', arr1)

# Permutations
arr0 = np.arange(5)
print('Array 0 \n', arr0)
 
np.random.shuffle(arr0)
print('Array 0 Shuffled \n', arr0)
 
# shuffle does exactly that, shuffles the elements of an array. Notice that in multi-dimensional arrays 
# this method only works along the first dimension, it only shuffles the rows.
arr1 = np.arange(9).reshape(3,3)
print('Array 1: \n', arr1)
 
np.random.shuffle(arr1)
print('Array 1 Shuffled: \n', arr1)
#Array 1: 
# [[0 1 2]
# [3 4 5]
# [6 7 8]]
#Array 1 Shuffled: 
# [[3 4 5]
# [0 1 2]
# [6 7 8]]

# The function permutation does the same, but instead of shuffling the elements in place it returns a copy leaving the original array unchanged.
arr0 = np.arange(5)
print('Array 0: \n', arr0)
 
arr1 = np.random.permutation(arr0)
print('Array 0 Shuffled: \n', arr1)
print('Array 0 Unchanged: \n', arr0)

# Distributions
arr1 = np.random.normal(67, 5,(4,4)) # mean=67, standard deviation=5, matrix ranking=4x4
print(arr1)

# Reference Guide
# Probability Distributions Arguments inside brackets are optional, the argument size can be an integer to generate a 1-dimensional array, or a tuple for multi-dimensional arrays.

#Numpy function	Description
#===========================
#beta(a, b[, size])	The beta distribution over [0, 1)
#binomial(n,p[,size])	Binomial distribution with size n and probability p.
#chisquare(df[, size])	Chi-squared distribution with df degrees of freedom.
#f(dfnum, dfden[,size])	Draw samples from a Fisher's F distribution.
#gamma(shape[, scale, size])	Draw samples from a Gamma distribution.
#geometric(p[, size])	Draw samples from a Geometric distribution.
#normal([mean, sd, size])	Draw samples from a Normal Gaussian distribution.
#poisson([lam, size])	Draw samples from a Poisson distribution

def sieve(n):
    numbers = np.arange(2,n+1, dtype=int)
    primes = []
    while (len(numbers) != 0):
        prime = numbers[0]
        primes.append(prime)
        mask = numbers % prime != 0
        numbers = numbers[mask]
    return np.array(primes)
 
 
if __name__ == "__main__":
    primes = sieve(100)
    print(primes)

