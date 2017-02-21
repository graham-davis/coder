import numpy as np

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """

    return np.array([3,3,3,3,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,2,3,3,2,2])

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    return np.array([6,7,7,9,12,11,9,9,8,5,4,4,4,4,3,4,4,8,9,2,0,9,3,0,0])

def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """

    return np.array([7,10,8,5,6,5,5,5,6,5,7,8,9,10,10,10,7,7,7,2,0,6,2,0,0])

# Question 1.c)
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Return:
            bits[nBands] is number of bits allocated to each scale factor band

        Logic:
           Maximizing SMR over block gives optimization result that:
               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
           where P is the pool of bits for mantissas and N is number of bands
           This result needs to be adjusted if any R(i) goes below 2 (in which
           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
           We will not bother to worry about slight variations in bit budget due
           to rounding of the above equation to integer values of R(i).
    """

    # Initialize MNR vector
    constantMNR = np.zeros(nBands)

    # Loop until bit budget is empty/no more bits can be added
    while(bitBudget > 0):
      # Calc noise floor as SMR - 6dB/bit in each band, order in descending order
      noiseFloor = (SMR - np.multiply(constantMNR, 6)).argsort()[::-1]
      bitAdded = False
      for n in noiseFloor:
        # If bit budget allows, add bit
        if nLines[n] <= bitBudget:
          bitBudget -= nLines[n]
          constantMNR[n] += 1
          bitAdded = True
          break
      # End loop if no bits were added
      if not bitAdded: break

    # Handle any single, negative or over maxMantBits
    constantMNR[np.equal(constantMNR, 1)] -= 1
    constantMNR[np.less(constantMNR, 0)] = 0
    constantMNR[np.greater(constantMNR, maxMantBits)] = maxMantBits

    return constantMNR.astype(int)
#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
  pass



