#!/usr/bin/env python3

import numpy as np
import sys
# Generate a random 2D buffer with a well-known seed.

def stats(x):
    return "{} {} ({} bytes total)".format(x.dtype, x.shape, np.product(x.shape)*x.dtype.itemsize)

N = 6
K = 2
if len(sys.argv) > 1:
    N = int(sys.argv[1])
if len(sys.argv) > 2:
    K = int(sys.argv[2])
inbuf = np.random.default_rng(seed=42).random((N,N))
print("inbuf   :", stats(inbuf))

# Calculate a 2D FFT all at once, and store it to a separate buffer.  (Does not modify inbuf.)




bufXY = np.fft.fftn(inbuf, axes=(0,1))
print("bufXY   :", stats(bufXY))


# Make modifications in the frequency domain
modXY = np.copy(bufXY)
modXY[  :K,  :K] *= -1
modXY[  :K,-K: ] *= -1
modXY[-K: ,  :K] *= -1
modXY[-K: ,-K: ] *= -1
print("modXY   :", stats(modXY))

# invert the FFT all at once








modXYiYX = np.fft.ifftn(modXY, axes=(1,0))
print("modXYiYX:", stats(modXYiYX))
outbuf = np.real(modXYiYX)
print("outbuf  :", stats(outbuf))
print("outbuf:\n", outbuf)
