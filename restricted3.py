#!/usr/bin/env python3

import numpy as np
import sys
# Generate a random 3D buffer with a well-known seed.

def stats(x):
    return "{} {} ({} bytes total)".format(x.dtype, x.shape, np.product(x.shape)*x.dtype.itemsize)

N = 6
K = 2
if len(sys.argv) > 1:
    N = int(sys.argv[1])
if len(sys.argv) > 2:
    K = int(sys.argv[2])
inbuf = np.random.default_rng(seed=42).random((N,N,N))
print("inbuf       :", stats(inbuf))

# Calculate a 3D FFT one dimension at a time, storing each intermediate result in its own buffer.  (Does not modify inbuf.)

bufXfull = np.fft.fft(inbuf, axis=2)
bufX = np.concatenate((bufXfull[:,:,:K], bufXfull[:,:,-K:]), axis=2)
print("bufX        :", stats(bufX))
bufXYfull = np.fft.fft(bufX, axis=1)
bufXY = np.concatenate((bufXYfull[:,:K,:], bufXYfull[:,-K:,:]), axis=1)
print("bufXY       :", stats(bufXY))
bufXYZfull = np.fft.fft(bufXY, axis=0)
bufXYZ = np.concatenate((bufXYZfull[:K,:,:], bufXYZfull[-K:,:,:]), axis=0)
print("bufXYZ      :", stats(bufXYZ))


# Make modifications in the frequency domain
modXYZ = np.copy(bufXYZ)
modXYZ[  :K,  :K,  :K] *= -1
modXYZ[  :K,  :K,-K: ] *= -1
modXYZ[  :K,-K: ,  :K] *= -1
modXYZ[  :K,-K: ,-K: ] *= -1
modXYZ[-K: ,  :K,  :K] *= -1
modXYZ[-K: ,  :K,-K: ] *= -1
modXYZ[-K: ,-K: ,  :K] *= -1
modXYZ[-K: ,-K: ,-K: ] *= -1
print("modXYZ      :", stats(modXYZ))

# invert the FFT one dimension at a time
modXYZfull = np.concatenate(
    (modXYZ[:K,:,:], bufXYZfull[K:-K,:,:], modXYZ[-K:,:,:]), axis=0)
modXYZiZ = np.fft.ifft(modXYZfull, axis=0)
# print("ifft(modXYZ[,,0]) =", np.fft.ifft(modXYZ[:,:,0]))
print("modXYZiZ    :", stats(modXYZiZ))
modXYZiZfull = np.concatenate(
    (modXYZiZ[:,:K,:], bufXYfull[:,K:-K,:], modXYZiZ[:,-K:,:]), axis=1)
modXYZiZY = np.fft.ifft(modXYZiZfull, axis=1)
print("modXYZiZY   :", stats(modXYZiZY))
modXYZiZYfull = np.concatenate(
    (modXYZiZY[:,:,:K], bufXfull[:,:,K:-K], modXYZiZY[:,:,-K:]), axis=2)
modXYZiZYX = np.fft.ifft(modXYZiZYfull, axis=2)
print("modXYZiZYX  :", stats(modXYZiZYX))
outbuf = np.real(modXYZiZYX)
print("outbuf      :", stats(outbuf))
print("outbuf[:2,:2,:2]:\n", outbuf[:2,:2,:2])
