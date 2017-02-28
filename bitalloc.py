import numpy as np

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """
    if bitBudget == 2526:
        array = [4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,2,2,2]
    else:
        array = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,3]
        
    return array

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    if bitBudget == 2526:
        array = np.array([0,0,9,15,16,16,16,15,11,2,0,0,0,0,0,0,0,13,14,0,0,14,0,0,0])
    else:
        array = np.array([3,6,13,16,16,16,16,16,16,5,2,2,2,2,2,2,2,16,16,2,2,16,2,2,0])
        
    return array

def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    if bitBudget == 2526:
        array = np.array([0,4,10,11,12,11,11,11,9,2,0,2,3,4,4,4,0,12,13,0,0,13,0,0,0])
    else:
        array = np.array([3,8,13,15,15,14,15,14,13,5,4,5,7,7,7,7,4,15,15,2,2,15,0,2,0])
        
    return array

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
           rounding of the above equation to integer values of R(i).
    """
    ########################### old version of code ##########################################
#    # constant to multiply by
#    C = np.log(10.0)/(20.0*np.log(2.0))
#    # automatically calculate bitAlloc
#    bitAlloc = np.round(bitBudget/(np.sum(nLines)) + C*(SMR-np.mean(SMR)),0)
#    # zero out anything less than 2
#    bitAlloc =  (bitAlloc >= 2)*bitAlloc
#    # make anything greater than 16 equal to 16
#    bitAlloc = bitAlloc-(bitAlloc - 16)*(bitAlloc > 16)
#
#    # add back bits while there are bits to add
#    while True:
#        ind = np.argmin(6*bitAlloc-SMR)
#        if bitAlloc[ind] == 0:
#            bitAdd = 2
#        else:
#            bitAdd = 1
#        bitAlloc[ind] += bitAdd
#        if bitBudget < np.sum(bitAlloc*nLines):
#            bitAlloc[ind] -= bitAdd
#            break
#    # remove bits while there are bits to remove
#    while True:    
#        nonZeroInds = np.squeeze(np.nonzero(np.logical_not(np.in1d(bitAlloc,0))*np.arange(len(bitAlloc))))
#        maxInd = nonZeroInds[np.argmin(nLines[nonZeroInds])]
#        if bitAlloc[maxInd] > 2:
#            bitAlloc[maxInd] -= 1
#        else:
#            bitAlloc[maxInd] == 0
#        if bitBudget > np.sum(bitAlloc*nLines):
#            break

    # constant to multiply by (6 dB rule)
    C = np.log(10.0)/(20.0*np.log(2.0))
    # perform first calculation using optimized formula
    bitAllocTemp = bitBudget/(np.sum(nLines)) + C*(SMR-np.mean(SMR))
    # locate indexes any non-positive values
    zInd = np.squeeze(np.where(bitAllocTemp<=0.0))
    # locate indexes of positive values
    nzInd = np.squeeze(np.where(bitAllocTemp>0.0))

    # counter that will help prevent infinite loop
    # we don't need to loop more than the number of
    # remaining non-zero bands as that would mean that
    # the algorithm got stuff returning negative value(s)

    #print zInd,zInd.size
    i = nBands - zInd.size

    # loop to continue optimized bit allocation
    while True:
        # recompute bit allocation based on number/size of non-zero bands
        bitAllocTemp = bitBudget/(np.sum(nLines[nzInd])) + C*(SMR[nzInd]-np.mean(SMR[nzInd]))
        # check for non-positive bit allocations
        zInd = np.squeeze(np.where(bitAllocTemp<=0.0))
        # it none or we've exceeded loop time, exit
        if zInd.size == 0 or i <= 0:
            break
        else:
            # get the correct indexes of the positive bit allocation bands
            nzInd = nzInd[np.squeeze(np.where(bitAllocTemp>0.0))]
            # decrement the counter
            i -= 1

    # allocate array to hold final bit allocation
    bitAlloc = np.zeros(nBands)
    # set the positive values using the floor of values returned by allocation algorithm
    bitAlloc[nzInd] = np.floor(bitAllocTemp)
    # make sure values are in valid range
    bitAlloc[np.where(np.logical_and(bitAlloc > 0.0, bitAlloc < 2.0))] = 0
    bitAlloc[np.where(bitAlloc > maxMantBits)] = maxMantBits

    # if we're under bit budget, we need to add bits
    availBits = np.arange(nBands) # keep track of bands where bits can be added
        
    bitAllocTemp = np.copy(bitAlloc)

    while np.sum(bitAllocTemp*nLines) < bitBudget  and availBits.size > 0:
        NMRs = 6*bitAllocTemp[availBits]-SMR[availBits]
        # find the smallest one
        minInd = availBits[np.argmin(NMRs)]
        
        if bitAllocTemp[minInd] == 0:
            bitChange = 2
        else:
            bitChange = 1
        
        if np.sum(bitAllocTemp*nLines)+bitChange*nLines[minInd] < bitBudget:
            bitAllocTemp[minInd] += bitChange
        else:
            availBits = np.delete(availBits,np.where(availBits == minInd))

    bitAlloc = np.copy(bitAllocTemp)

    # if we're still over bit budget, we need to remove bits
    while np.sum(bitAlloc*nLines) > bitBudget:
        # get the nonzero indexes
        nzInd = np.squeeze(np.where(bitAlloc > 0))
        # calculate their NMRs
        NMRs = 6*bitAlloc[nzInd]-SMR[nzInd]
        # find the largest one
        maxInd = nzInd[np.argmax(6*bitAlloc[nzInd]-SMR[nzInd])]    
        #print np.sum(bitAlloc*nLines), bitBudget, nzInd, maxInd, bitAlloc[maxInd]
        # reduce it by the correct amount
        if bitAlloc[maxInd] > 2:
            # larger than 2, subtract 1
            bitAlloc[maxInd] -= 1
            # less than or equal to 2, set to 0
        else:
            bitAlloc[maxInd] = 0

    # return the final allocation
    return bitAlloc.astype(int)

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE
