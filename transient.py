import numpy as np
from numpy.fft import fft
from window import HanningWindow
from matplotlib.pyplot import figure, show

def IsTransient(previousBlock, currentBlock):
    """
    Returns true if the next data block contains a transient signal
    """
    N = len(previousBlock)
    W = np.arange(1, N+1)
    pEnergy = (1./N)*np.sum(np.multiply(W, np.power(np.abs(fft(HanningWindow(previousBlock))), 2)))
    cEnergy = (1./N)*np.sum(np.multiply(W, np.power(np.abs(fft(HanningWindow(currentBlock))), 2)))

    if (cEnergy - pEnergy) > 75:
        return True

    return False

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    pass
    
