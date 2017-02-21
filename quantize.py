"""
Graham Davis / Homework 2 Problem 2

quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")
"""

import numpy as np

### Problem 1.a.i ###
def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    #Notes:
    #The overload level of the quantizer should be 1.0

    # Initialize sign bit
    s = 1 if (aNum < 0) else 0

    # Bit shift to have nBits bits
    s = s << (nBits - 1)

    # Calculate code
    code = 0
    if abs(aNum) >= 1:
        code = 2**(nBits-1)-1
    else:
        code = int((((2**nBits-1)*abs(aNum))+1)/2)
    
    # Add code to quantized number bits
    aQuantizedNum = s | code

    return aQuantizedNum

### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """

    # Obtain sign bit
    sMask = 1 << nBits - 1
    s = -1 if aQuantizedNum & sMask else 1

    # Obtain code bits
    cMask = 2**(nBits-1)-1
    code = aQuantizedNum & cMask

    # Calculate dequantized value
    aNum = s*(float(2*abs(code))/(2**nBits-1))

    return aNum

### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """

    aQuantizedNumVec = np.zeros_like(aNumVec, dtype = int)

    # Calculate sign bits
    s = np.less(aNumVec, aQuantizedNumVec).astype(int)
    s = np.left_shift(s, nBits - 1)

    # Calculate code
    aQuantizedNumVec = ((((2**nBits-1)*abs(aNumVec))+1)/2).astype(int)
    aQuantizedNumVec = np.clip(aQuantizedNumVec, 0, 2**(nBits-1)-1)

    # Add sign bit to code
    aQuantizedNumVec = np.bitwise_or(s, aQuantizedNumVec)

    return aQuantizedNumVec

### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """

    aNumVec = np.zeros_like(aQuantizedNumVec, dtype = float) 

    # Calculate sign bits
    s = np.right_shift(aQuantizedNumVec, (nBits - 1))*(-2)+1

    # Obtain code bits
    cMask = 2**(nBits-1)-1
    code = np.bitwise_and(aQuantizedNumVec, cMask)

    # Calculate dequantized value
    aNumVec = (np.multiply(s, ((2.0*abs(code)))/(2**nBits-1))).astype(float)

    return aNumVec

### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    #Notes:
    #The scale factor should be the number of leading zeros

    scale = 0 
    r = (2**nScaleBits)-1 + nMantBits

    # Uniformly quantize aNum with r bits
    uQuantized = QuantizeUniform(aNum, r)

    # Count leading zeros in code
    while (scale < r - 1):
        mask = 1 << (r - 2 - scale)
        if not (uQuantized & mask):
            scale += 1
        else:
            break;

    # Return scale (or 2^nScaleBits - 1 if scale is too large)
    return (2**nScaleBits)-1 if scale > (2**nScaleBits)-1 else scale

### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    mantissa = 0 
    rs = (2**nScaleBits)-1
    r =  rs + nMantBits

    # Uniformly quantize aNum with r bits
    uQuantized = QuantizeUniform(aNum, r) 

    # Calculate sign bit
    s = uQuantized >> (r - 1)
    s = s << nMantBits - 1

    if scale is rs:
        # If we have max number of scale bits, set mantissa mask to 1111
        mMask = 2**(nMantBits-1) - 1
        mantissa = uQuantized & mMask
    else:
        # mMask is r-2-scale one's (to ignore sign bit, scale bits and leading one)
        mMask = (2**r - 1) >> 2 + scale
        # shift mantissa so it is of length nMantBits-1 (does not include sign bit)
        mantissa = (uQuantized & mMask) >> (rs - scale - 1)
    # Add sign bit to mantissa and return
    return mantissa | s

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """

    # calculate number of bits
    r = (2**nScaleBits - 1)+nMantBits

    # get sign bit
    q = mantissa >> nMantBits - 1

    # remove sign bit from mantissa
    mantissa = mantissa & (2**(nMantBits-1) - 1)

    # add scaling zeros
    q = q << scale

    if scale is not 2**(nScaleBits) - 1:
        # keep track of remaining bits
        bRemaining = r - (nMantBits + scale + 1)

        # add leading one
        q = (q << 1) + 1

        # add mantissa bits
        q = (q << nMantBits - 1) | mantissa

        if bRemaining is not 0:
            bRemaining = bRemaining - 1
            # add trailing 1
            q = (q << 1) + 1

            # add remaining bits
            q = q << bRemaining

    else:
        # add mantissa bits
        q = (q << nMantBits - 1) | mantissa

    # dequantize 
    aNum = DequantizeUniform(q, r)

    return aNum

### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    if nMantBits <= 0: return 0
    # Calculate code length
    rs = (2**nScaleBits)-1
    r =  rs + nMantBits

    # Uniformly quantize aNum with r bits
    uQuantized = QuantizeUniform(aNum, r) 

    # Calculate sign bit
    s = uQuantized >> (r - 1)
    s = s << (nMantBits - 1)

    if scale is rs:
        # If we have max number of scale bits, set mantissa mask to 1111
        mMask = 2**(nMantBits-1) - 1
        mantissa = uQuantized & mMask
    else:
        # mMask is r-1-scale one's (to ignore sign bit and scale bits)
        mMask = (2**r - 1) >> (1 + scale)
        # shift mantissa so it is of length nMantBits-1 (does not include sign bit)
        mantissa = (uQuantized & mMask) >> (rs - scale)
    # Add sign bit to mantissa and return
    return mantissa | s

### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """
    if nMantBits <= 0: return 0;

    # calculate number of bits
    r = (2**nScaleBits - 1)+nMantBits

    # get sign bit
    q = mantissa >> nMantBits - 1

    # remove sign bit from mantissa
    mantissa = mantissa & (2**(nMantBits-1) - 1)

    # add scaling zeros
    q = q << scale

    # add trailing zeros if needed
    if scale is not 2**(nScaleBits) - 1:
        # remaining bits to be added
        bRemaining = r - (scale + nMantBits)

        # add mantissa bits
        q = (q << nMantBits - 1) | mantissa

        if mantissa is not 0:
            # add trailing 1
            q = (q << 1) + 1
            bRemaining -= 1

        q = q << bRemaining
    else:
        # add mantissa bits
        q = (q << nMantBits - 1) | mantissa

    # dequantize 
    aNum = DequantizeUniform(q, r)

    return aNum

### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    if nMantBits <= 0: return np.zeros_like(aNumVec, dtype=np.uint64)

    # Calculate code length
    rs = (2**nScaleBits)-1
    r =  rs + nMantBits

    # Uniformly quantize aNumVec
    vQuantized = vQuantizeUniform(aNumVec, r)

    # Calculate sign bit
    s = np.right_shift(vQuantized, (r - 1))
    s = np.left_shift(s, (nMantBits - 1))

    if scale is rs:
        # If we have max number of scale bits, set mantissa mask to 1111
        mMask = 2**(nMantBits-1) - 1

        mantissaVec = np.bitwise_and(vQuantized, mMask)

    else:
        # mMask is r-1-scale one's (to ignore sign bit and scale bits)
        mMask = (2**r - 1) >> (1 + scale)
        # shift mantissa so it is of length nMantBits-1 (does not include sign bit)
        mantissaVec = np.right_shift(np.bitwise_and(vQuantized, mMask), (rs - scale))
    # Add sign bit to mantissa and return
    return np.bitwise_or(mantissaVec, s)

### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """
    if nMantBits <= 0: return np.zeros_like(aNumVec, dtype=np.uint64)
    # calculate number of bits
    r = (2**nScaleBits - 1)+nMantBits
    # get sign bit
    q = np.right_shift(mantissaVec, nMantBits - 1)

    # remove sign bit from mantissa
    mantissaVec = np.bitwise_and(mantissaVec, (2**(nMantBits-1) - 1))
    # add scaling zeros
    q = np.left_shift(q, scale)

    # add trailing zeros if needed
    if scale is not 2**(nScaleBits) - 1:
        # add mantissa bits
        q = np.bitwise_or(np.left_shift(q, nMantBits - 1), mantissaVec)
        # add trailing one if mantissa is non-zero
        trailingOnes = np.greater(mantissaVec, 0).astype(mantissaVec.dtype)

        q = np.left_shift(q, trailingOnes) + trailingOnes

        # add remaining zeros
        bRemaining = r - (trailingOnes + scale + nMantBits)
        q = np.left_shift(q, bRemaining)
    else:
        # add mantissa bits
        q = np.bitwise_or(np.left_shift(q, nMantBits - 1), mantissaVec)

    # dequantize 
    aNumVec = vDequantizeUniform(q, r)

    return aNumVec

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    pass

