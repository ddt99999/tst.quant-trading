# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:43:34 2016

Source: http://times.cs.uiuc.edu/course/410/note/mle.pdf

MLE example

@author: tongtz
"""

import numpy as np

n = 100 
                                # no. of independent Bernoulli trials (i.e. sample size)
t = np.array([1,3,6,9,12,18], dtype=int)                     # time intervals as a column vector
t_dash = np.transpose(t) 
y = np.array([.94, .77, .40, .26, .24, .16])    # observed proportion correct as a column vector
y_dash = np.transpose(y)

x = np.multiply(t_dash, y_dash)


    