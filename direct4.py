#!/usr/bin/env python3

import numpy as np
import sys
# Generate a random 4D buffer with a well-known seed.

def stats(x):
    return "{} {} ({} bytes total)".format(x.dtype, x.shape, np.product(x.shape)*x.dtype.itemsize)

N = 6
K = 2
if len(sys.argv) > 1:
    N = int(sys.argv[1])
if len(sys.argv) > 2:
    K = int(sys.argv[2])
inbuf = np.random.default_rng(seed=42).random((N,N,N,N))
print("inbuf       :", stats(inbuf))

# Calculate a 4D FFT all at once, and store it to a separate buffer.  (Does not modify inbuf.)










bufXYZW = np.fft.rfftn(inbuf, axes=(0,1,2,3))

print("bufXYZW     :", stats(bufXYZW))


# Make modifications in the frequency domain
modXYZW = np.copy(bufXYZW)
modXYZW[  :K,  :K,  :K,  :K] *= -1
modXYZW[  :K,  :K,-K: ,  :K] *= -1
modXYZW[  :K,-K: ,  :K,  :K] *= -1
modXYZW[  :K,-K: ,-K: ,  :K] *= -1
modXYZW[-K: ,  :K,  :K,  :K] *= -1
modXYZW[-K: ,  :K,-K: ,  :K] *= -1
modXYZW[-K: ,-K: ,  :K,  :K] *= -1
modXYZW[-K: ,-K: ,-K: ,  :K] *= -1
print("modXYZW     :", stats(modXYZW))

# invert the FFT all at once


















outbuf = np.fft.irfftn(modXYZW, axes=(0,1,2,3))
print("outbuf      :", stats(outbuf))
print("outbuf[:2,:2,:2,:2]:\n", outbuf[:2,:2,:2,:2])
