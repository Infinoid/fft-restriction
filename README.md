# Restricted FFT

This is a proof of concept for an idea we had for how to speed up a distributed
fourier neural operator app.

DFNO uses a multi-dimensional FFT, applies weights to some elements in the
frequency domain, discards the other elements, and then goes back to the time
domain.

It doesn't need all of the frequency domain, it only needs the low frequencies
in all dimensions.  In other words, it only needs the corners of the data
block.

So why not just skip the parts it doesn't need?  The inner dimensions of FFT
can do less work, and the distributed version can send less data over the
network.

This is a proof of concept to show that the math still works.

## Proof of concept

I have implemented proofs of concepts for 2D, 3D and 4D.

It operates by generating a random buffer with a fixed seed (for comparison),
doing an N-dimensional FFT, modifying the corners of the buffer (just inverting
the sign), doing an N-dimensional iFFT, and printing some of the output buffer.

For each dimensionality, there are 3 scripts, of increasing complexity.  They
are named like:

* directN.py - Simplest version, uses single `fftn` and `ifftn` calls
* individualN.py - series of one-dimensional `fft` and `ifft` calls
* restrictedN.py - removes unused data from intermediate buffers as it goes

The individual/restricted scripts keep a lot of intermediate buffers lying
around.  The restricted script shrinks the buffer as it goes, and during the
inverse FFTs, it adds the unused data back in.  At the end, the result is the
same as the other two scripts, proving that the non-corner data wasn't needed
and the math still works.

The 4D scripts use `rfft` for the outermost dimension, the same way DFNO does,
and it also restricts the innermost buffer so that it can operate on the
corners without needing any slice syntax.

### Distributed operation

All of this is compatible with distributed operation, as long as the output
distribution matches the input distribution.  The intermediate buffers storing
unused data can remain purely local, thus saving network bandwidth.

### Zeroing out the higher frequencies

That said, the real FNO/DFNO code seems to discard the unused data and replace
it with zeroes, which makes things easier.  It means there's no need to keep
the intermediate buffers around, and as the buffer size grows during the iFFTs,
the extra space can just be filled with zeroes.

## Running it

The scripts take size parameters, the syntax is:
`<script> <datasize> <cornersize>`

For 2D, the data size will be datasize*datasize.  For 3D, it will be datasize cubed,
and so forth.

When you run it, the buffer sizes look like this:

```sh
$ python3 restricted4.py 64 4
inbuf       : float64 (64, 64, 64, 64) (134217728 bytes total)
bufX        : complex128 (64, 64, 64, 4) (16777216 bytes total)
bufXY       : complex128 (64, 64, 8, 4) (2097152 bytes total)
bufXYZ      : complex128 (64, 8, 8, 4) (262144 bytes total)
bufXYZW     : complex128 (8, 8, 8, 4) (32768 bytes total)
modXYZW     : complex128 (8, 8, 8, 4) (32768 bytes total)
modXYZWiW   : complex128 (64, 8, 8, 4) (262144 bytes total)
modXYZWiWZ  : complex128 (64, 64, 8, 4) (2097152 bytes total)
modXYZWiWZY : complex128 (64, 64, 64, 4) (16777216 bytes total)
outbuf      : float64 (64, 64, 64, 64) (134217728 bytes total)
outbuf[:2,:2,:2,:2]:
[it dumps some raw data here to compare with the other scripts]
```
