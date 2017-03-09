import numpy as np
from numpy.fft import fft
from window import HanningWindow

def IsTransient(previousBlock, currentBlock):
    """
    Returns true if the next data block contains a transient signal
    """
    N = len(currentBlock)
    W = np.arange(1, N+1)
    # Calculations for spectral energy difference between current and previous block
    pEnergy = (1./N)*np.sum(np.multiply(W, np.power(np.abs(fft(HanningWindow(previousBlock))), 2)))
    cEnergy = (1./N)*np.sum(np.multiply(W, np.power(np.abs(fft(HanningWindow(currentBlock))), 2)))
    eDiff = cEnergy - pEnergy

    # Spectral energy differences within current block (first half vs. second half)
    fHalf = np.sum(np.multiply(W[0:N/2], np.power(np.abs(fft(HanningWindow(currentBlock[0:N/2]))), 2)))
    sHalf = np.sum(np.multiply(W[0:N/2], np.power(np.abs(fft(HanningWindow(currentBlock[N/2:]))), 2)))
    cDiff =  np.abs(fHalf - sHalf)

    # Time domain signal energy difference within block
    efHalf = np.sum(np.power(np.abs(currentBlock[0:N/2]), 2))
    esHalf = np.sum(np.power(np.abs(currentBlock[N/2:]), 2)) 
    teDiff = np.abs(efHalf - esHalf)

    if eDiff > 150 or cDiff > 45000 or teDiff > 2:
        return True

    return False

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    pass
    
