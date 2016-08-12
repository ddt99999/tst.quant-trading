# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:04:02 2016

@author: tongtz
"""

import numpy as np
import matplotlib.pyplot as plt

Z = [[0,0,0,0,0,0],
     [0,0,0,1,0,0],
     [0,1,0,1,0,0],
     [0,0,1,1,0,0],
     [0,0,0,0,0,0],
     [0,0,0,0,0,0]]
     
Z[0:]
        
# In PYTHON WAY
# =============
def compute_neighbours(Z):
    shape = len(Z), len(Z[0])
    N = [[0]*(shape[0]) for i in range(shape[1])]
    for x in range(1,shape[0]-1):
        for y in range(1,shape[1]-1):
            N[x][y] = Z[x-1][y-1] + Z[x][y-1] + Z[x+1][y-1] + \
                      Z[x-1][y]               + Z[x+1][y] + \
                      Z[x-1][y+1] + Z[x][y+1] + Z[x+1][y+1]
                      
    return N
    
def iterate(Z):
    N = compute_neighbours(Z)
    for x in range(1,shape[0]-1):
        for y in range(1, shape[1]-1):
            if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                Z[x][y] = 0
            elif Z[x][y] == 0 and N[x][y] == 3:
                Z[x][y] = 1
                
    return Z
    
def show(Z):
    for l in Z[1:-1]: 
        print(l[1:-1])
    
    
for i in range(4):
    iterate(Z)
    show(Z)

# IN NUMPY WAY
# ============
Z = np.array([[0,0,0,0,0,0],
              [0,0,0,1,0,0],
              [0,1,0,1,0,0],
              [0,0,1,1,0,0],
              [0,0,0,0,0,0],
              [0,0,0,0,0,0]])
              
print(Z.dtype)
print(Z.shape)
print(Z[1,3])
print(Z[1:5,1:5])

N = np.zeros(Z.shape, dtype=int)
print(Z[ :-2, :-2])
print(Z[ :-2,1:-1])
print(Z[ :-2,2: ])
print(Z[1:-1, :-2])
print(Z[1:-1,1:-1])
print(Z[1:-1,2: ])
print(Z[2: , :-2])
print(Z[2: ,1:-1])
print(Z[2: ,2: ])
print(Z[ :-2, :-2] + Z[ :-2,1:-1])

# Iterate the game of life : naive version
# Count neighbours
N = np.zeros(Z.shape, int)
N[1:-1,1:-1] += (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
                 Z[1:-1,0:-2]                + Z[1:-1,2:] +
                 Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
N_ = N.ravel()


print((Z[ :-2, :-2] + Z[ :-2,1:-1] + Z[ :-2,2:] +
       Z[1:-1, :-2]                + Z[1:-1,2:] +
       Z[2:  , :-2] + Z[2:  ,1:-1] + Z[2:  ,2:]))

def iterate(Z):
    # Iterate the game of life : naive version
    # Count neighbours
    N = np.zeros(Z.shape, int)
    N[1:-1,1:-1] += (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
                     Z[1:-1,0:-2]                + Z[1:-1,2:] +
                     Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
    N_ = N.ravel()
    Z_ = Z.ravel()
    
    # Apply rules
    R1 = np.argwhere( (Z_==1) & (N_ < 2) )
    R2 = np.argwhere( (Z_==1) & (N_ > 3) )
    R3 = np.argwhere( (Z_==1) & ((N_==2) | (N_==3)) )
    R4 = np.argwhere( (Z_==0) & (N_==3) )

    # Set new values
    Z_[R1] = 0
    Z_[R2] = 0
    Z_[R3] = Z_[R3]
    Z_[R4] = 1

    # Make sure borders stay null
    Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = 0
    
def iterate_optimize(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
         
    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    
    Z[...] = 0
    Z[1:-1,1:-1][birth|survive] = 1
    return Z
    
def display(Z):  
    size = np.array(Z.shape)
    dpi = 72.0
    figsize= size[1]/float(dpi),size[0]/float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    plt.imshow(Z,interpolation='nearest', cmap=plt.cm.gray_r)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
# Getting bigger
Z = np.random.randint(0,2,(20,20))

for i in range(5):
    iterate_optimize(Z)
    display(Z)
  
