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

# Calculate a 4D FFT one dimension at a time, storing each intermediate result in its own buffer.  (Does not modify inbuf.)

bufX = np.fft.rfft(inbuf, axis=3)

print("bufX        :", stats(bufX))
bufXY = np.fft.fft(bufX, axis=2)

print("bufXY       :", stats(bufXY))
bufXYZ = np.fft.fft(bufXY, axis=1)

print("bufXYZ      :", stats(bufXYZ))
bufXYZW = np.fft.fft(bufXYZ, axis=0)

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

# invert the FFT one dimension at a time


modXYZWiW = np.fft.ifft(modXYZW, axis=0)
# print("ifft(modXYZ[,,0]) =", np.fft.ifft(modXYZ[:,:,0]))
print("modXYZWiW   :", stats(modXYZWiW))



modXYZWiWZ = np.fft.ifft(modXYZWiW, axis=1)
print("modXYZWiWZ  :", stats(modXYZWiWZ))



modXYZWiWZY = np.fft.ifft(modXYZWiWZ, axis=2)
print("modXYZWiWZY :", stats(modXYZWiWZY))



outbuf = np.fft.irfft(modXYZWiWZY, axis=3)
print("outbuf      :", stats(outbuf))
print("outbuf[:2,:2,:2,:2]:\n", outbuf[:2,:2,:2,:2])
