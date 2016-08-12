# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:11:06 2016

@author: tongtz
References:
----------
Complex Patterns in a Simple System
John E. Pearson, Science 261, 5118, 189-192, 1993.

Encode movie
------------

ffmpeg -r 30 -i "tmp-%03d.png" -c:v libx264 -crf 23 -pix_fmt yuv420p bacteria.mp4
"""

import numpy as np
import ffmpeg
import matplotlib.pyplot as plt

def laplacian(Z):
    return (                 Z[0:-2,1:-1] +
            Z[1:-1,0:-2] - 4*Z[1:-1,1:-1] + Z[1:-1,2:] +
                             Z[2:  ,1:-1] )

# Parameters from http://www.aliensaint.com/uo/java/rd/
# -----------------------------------------------------
n  = 200
Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065 # Bacteria 1
# Du, Dv, F, k = 0.14, 0.06, 0.035, 0.065 # Bacteria 2
# Du, Dv, F, k = 0.16, 0.08, 0.060, 0.062 # Coral
# Du, Dv, F, k = 0.19, 0.05, 0.060, 0.062 # Fingerprint
# Du, Dv, F, k = 0.10, 0.10, 0.018, 0.050 # Spirals
# Du, Dv, F, k = 0.12, 0.08, 0.020, 0.050 # Spirals Dense
# Du, Dv, F, k = 0.10, 0.16, 0.020, 0.050 # Spirals Fast
# Du, Dv, F, k = 0.16, 0.08, 0.020, 0.055 # Unstable
# Du, Dv, F, k = 0.16, 0.08, 0.050, 0.065 # Worms 1
# Du, Dv, F, k = 0.16, 0.08, 0.054, 0.063 # Worms 2
# Du, Dv, F, k = 0.16, 0.08, 0.035, 0.060 # Zebrafish

Z = np.zeros((n+2,n+2), [('U', np.double),
                         ('V', np.double)])
print(Z.dtype)        

U,V = Z['U'], Z['V']     
u,v = U[1:-1,1:-1], V[1:-1,1:-1]   

r = 20
u[...] = 1.0
U[n/2-r:n/2+r,n/2-r:n/2+r] = np.double(0.50)
V[n/2-r:n/2+r,n/2-r:n/2+r] = np.double(0.25)
u += 0.05*np.random.random((n,n))
v += 0.05*np.random.random((n,n))   

plt.ion()

size = np.array(Z.shape)
dpi = 72.0
figsize= size[1]/float(dpi),size[0]/float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
im = plt.imshow(V, interpolation='bicubic', cmap=plt.cm.gray_r)
plt.xticks([]), plt.yticks([])    

for i in range(25000):
    Lu = laplacian(U)
    Lv = laplacian(V)

    uvv = u*v*v
    u += (Du*Lu - uvv +  F   *(1-u))
    v += (Dv*Lv + uvv - (F+k)*v    )

    if i % 10 == 0:
        im.set_data(V)
        im.set_clim(vmin=V.min(), vmax=V.max())
        plt.draw()
        # To make movie
        plt.savefig("./tmp/tmp-%03d.png" % (i/10) ,dpi=dpi)

plt.ioff()
# plt.savefig("../figures/zebra.png",dpi=dpi)
# plt.savefig("../figures/bacteria.png",dpi=dpi)
# plt.savefig("../figures/fingerprint.png",dpi=dpi)
plt.show()

#ffmpeg -r 30 -i "./tmp/tmp-%03d.png" -c:v libx264 -crf 23 -pix_fmt yuv420p bacteria.mp4