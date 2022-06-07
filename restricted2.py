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
print("inbuf      :", stats(inbuf))

# Calculate a 2D FFT one dimension at a time, storing each intermediate result in its own buffer.  (Does not modify inbuf.)

bufXfull = np.fft.fft(inbuf, axis=1)
bufX = np.concatenate((bufXfull[:,:K], bufXfull[:,-K:]), axis=1)
print("bufX       :", stats(bufX))
bufXY = np.fft.fft(bufX , axis=0)
print("bufXY      :", stats(bufXY))


# Make modifications in the frequency domain
modXY = np.copy(bufXY)
modXY[  :K,  :K] *= -1
modXY[  :K,-K: ] *= -1
modXY[-K: ,  :K] *= -1
modXY[-K: ,-K: ] *= -1
print("modXY      :", stats(modXY))

# invert the FFT one dimension at a time

modXYiY  = np.fft.ifft(modXY  , axis=0)
#print("ifft(modXY[0]) =", np.fft.ifft(modXY[:,0]))
print("modXYiY    :", stats(modXYiY))
modXYiYfull = np.concatenate(
    (modXYiY[:,:K], bufXfull[:,K:-K], modXYiY[:,-K:]), axis=1)
print("modXYiYfull:", stats(modXYiYfull))

modXYiYX = np.fft.ifft(modXYiYfull, axis=1)
print("modXYiYX   :", stats(modXYiYX))
outbuf = np.real(modXYiYX)
print("outbuf     :", stats(outbuf))
print("outbuf:\n", outbuf)
