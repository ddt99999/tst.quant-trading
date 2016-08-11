# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:04:02 2016

@author: tongtz
"""

import numpy as np

Z = [[0,0,0,0,0,0],
     [0,0,0,1,0,0],
     [0,1,0,1,0,0],
     [0,0,1,1,0,0],
     [0,0,0,0,0,0],
     [0,0,0,0,0,0]]
        
def compute_neighbours(Z):
    shape = len(Z), len(Z[0])
    N = [[0]*(shape[0]) for i in range(shape[1])]
    for x in range(1,shape[0]-1):
        for y in range(1,shape[1]-1):
            N[x][y] = Z[x-1][y-1] + Z[x][y-1] + Z[x+1][y-1] + \
                      Z[x-1][y]               + Z[x+1][y] + \
                      Z[x-1][y+1] + Z[x][y+1] + Z[x+1][y+1]
                      
    return N

