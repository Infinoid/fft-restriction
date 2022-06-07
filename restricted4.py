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

bufXfull = np.fft.rfft(inbuf, axis=3)
bufX = bufXfull[:,:,:,:K]
print("bufX        :", stats(bufX))
bufXYfull = np.fft.fft(bufX, axis=2)
bufXY = np.concatenate((bufXYfull[:,:,:K,:], bufXYfull[:,:,-K:,:]), axis=2)
print("bufXY       :", stats(bufXY))
bufXYZfull = np.fft.fft(bufXY, axis=1)
bufXYZ = np.concatenate((bufXYZfull[:,:K,:,:], bufXYZfull[:,-K:,:,:]), axis=1)
print("bufXYZ      :", stats(bufXYZ))
bufXYZWfull = np.fft.fft(bufXYZ, axis=0)
bufXYZW = np.concatenate((bufXYZWfull[:K,:,:,:], bufXYZWfull[-K:,:,:,:]), axis=0)
print("bufXYZW     :", stats(bufXYZW))


# Make modifications in the frequency domain
modXYZW = np.copy(bufXYZW)
modXYZW[:,:,:,:] *= -1







print("modXYZW     :", stats(modXYZW))

# invert the FFT one dimension at a time
modXYZWfull = np.concatenate(
    (modXYZW[:K,:,:,:], bufXYZWfull[K:-K,:,:,:], modXYZW[-K:,:,:,:]), axis=0)
modXYZWiW = np.fft.ifft(modXYZWfull, axis=0)
# print("ifft(modXYZ[,,0]) =", np.fft.ifft(modXYZ[:,:,0]))
print("modXYZWiW   :", stats(modXYZWiW))
modXYZWiWfull = np.concatenate(
    (modXYZWiW[:,:K,:,:], bufXYZfull[:,K:-K,:,:], modXYZWiW[:,-K:,:,:]), axis=1)
# print("modXYZiZfull:\n", stats(modXYZiZfull))
modXYZWiWZ = np.fft.ifft(modXYZWiWfull, axis=1)
print("modXYZWiWZ  :", stats(modXYZWiWZ))
modXYZWiWZfull = np.concatenate(
    (modXYZWiWZ[:,:,:K,:], bufXYfull[:,:,K:-K,:], modXYZWiWZ[:,:,-K:,:]), axis=2)
# print("modXYZiZfull:\n", modXYZiZfull)
modXYZWiWZY = np.fft.ifft(modXYZWiWZfull, axis=2)
print("modXYZWiWZY :", stats(modXYZWiWZY))
modXYZWiWZYfull = np.concatenate(
    (modXYZWiWZY[:,:,:,:K], bufXfull[:,:,:,K:]), axis=3)
# print("modXYZiZYfull:\n", modXYZiZYfull)
outbuf = np.fft.irfft(modXYZWiWZYfull, axis=3)
print("outbuf      :", stats(outbuf))
print("outbuf[:2,:2,:2,:2]:\n", outbuf[:2,:2,:2,:2])
