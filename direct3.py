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

# Calculate a 3D FFT all at once, and store it to a separate buffer.  (Does not modify inbuf.)







bufXYZ = np.fft.fftn(inbuf, axes=(0,1,2))

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

# invert the FFT all at once











modXYZiZYX = np.fft.ifftn(modXYZ, axes=(2,1,0))
print("modXYZiZYX  :", stats(modXYZiZYX))
outbuf = np.real(modXYZiZYX)
print("outbuf      :", stats(outbuf))
print("outbuf[:2,:2,:2]:\n", outbuf[:2,:2,:2])
