"""
Graham Davis / Homework 3
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import time as time

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    # Initialize N and n0
    N = a+b
    n0 = (b+1)/2.0

    # Calculate IMDCT
    if isInverse:
        # Initialize k and result vector
        result = np.zeros(N)
        k = np.arange(0,N/2)

        for n in range(0, N):
            # Calculate the IMDCT kernel
            kernel = np.cos(np.multiply(((2*np.pi)/N)*(k+0.5),(np.add(n,n0))))
            # Sum and scale the IMDCT result
            result[n] = np.multiply(2.0, np.sum(np.multiply(data, kernel)))

    # Calculate MDCT
    else:
        # Initialize n and result vector
        result = np.zeros(N/2)
        n = np.arange(0, N)
        
        for k in range(0, result.size):
            # Calculate the MDCT kernel
            kernel = np.cos(np.multiply(((2*np.pi)/N)*(k+0.5),(np.add(n,n0))))
            # Sum and scale the MDCT result 
            result[k] = np.multiply(2.0/N, np.sum(np.multiply(data, kernel)))
    
    return result

### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    # Initialize helper variables
    N = a+b
    n = np.arange(0, N)
    k = np.arange(0, N/2)
    n0 = (b+1)/2.0

    # Compute Fast IMDCT
    if isInverse:
        # Initialize result array
        result = np.zeros(N)

        # Compute pre-twiddle
        Y = np.multiply(data, np.exp(np.divide(np.multiply(1j*2*np.pi*n0, k), N)))

        # Compute the N point IFFT
        y = np.fft.ifft(Y, N)

        # Compute post-twiddle
        yPrime = np.real(np.multiply(y, np.exp(np.multiply(1j*((2*np.pi)/(2*N)), np.add(n,n0)))))

        # Scale result
        result = np.multiply(N*2, yPrime)


    # Compute Fast MDCT
    else:
        # Initialize result array
        result = np.zeros(N/2)
        # Compute pre-twiddle
        y = np.multiply(data, np.exp(np.divide(np.multiply(-1j*2*np.pi, n), 2*N)))

        # Compute N point FFT and keep first N/2 values
        Y = np.fft.fft(y, N)
        Y = Y[0:N/2]

        # Compute post-twiddle
        yPrime = np.real(np.multiply(Y, np.exp(np.multiply(-1j*((2*np.pi)/N)*n0, np.add(k, 0.5)))))

        # Scale result
        result = np.multiply(2.0/N, yPrime)

    return result

def IMDCT(data,a,b):

    return MDCT(data, a, b, True)

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### Code for question 2b ###

    # Initialize input/output arrays
    x = np.array([0,1,2,3,4,4,4,4,3,1,-1,-3])

    # Define helper variables
    a = 4
    b = 4
    sPerBlock = 4

    # Zero pad input
    x = np.concatenate((x, np.zeros(sPerBlock)), axis=0)
    nBlocks = x.size/sPerBlock

    # Initialize blocks
    pSamples = np.zeros(sPerBlock)
    samples = np.zeros(a+b)

    # Initialize output arrays
    output = np.zeros_like(x)
    pOutput = np.zeros(sPerBlock)

    # Loop through blocks
    for block in range(0, nBlocks):
        # Calculate current start sample
        cBlock = block*sPerBlock;
        # Concatenate previous samples and next samples 
        samples = np.concatenate((pSamples, x[cBlock:cBlock+sPerBlock]), axis=0)
        # Calculate MDCT/IMDCT
        cOutput = MDCTslow(MDCTslow(samples, a, b, False), a, b, True)
        # Overlap add cOuput[0:sPerBlock] and pOutput
        output[cBlock:cBlock+sPerBlock] = (cOutput[0:sPerBlock] + pOutput) / 2
        # Update previous sample
        pSamples = x[cBlock:cBlock+sPerBlock]
        # Update previous output
        pOutput = cOutput[sPerBlock:]

    print output


