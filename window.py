import numpy as np
from mdct import *
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """
    n_Num = len(dataSampleArray)
    n = np.arange(n_Num)
    window = np.sin(np.pi * (n + 0.5) / n_Num)
    windowed_data = window * dataSampleArray
    return  windowed_data

def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    n_Num = len(dataSampleArray)
    n = np.arange(n_Num)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * (n + 0.5) / n_Num))
    windowed_data = window * dataSampleArray
    return  windowed_data

### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following pp. 108-109 and pp. 117-118 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """
    n_Num = len(dataSampleArray)
    half_n_Num = n_Num / 2
    w_kb = np.arange(half_n_Num + 1)
    w_kb = np.i0(np.pi * alpha * np.sqrt(1.0 - np.square(w_kb * 2.0 / half_n_Num - 1.0))) / np.i0(np.pi * alpha)
    window = np.zeros(n_Num)
    window[0 : half_n_Num] = np.sqrt(np.cumsum(np.square(w_kb[0 : half_n_Num])) / np.sum(np.square(w_kb)))
    window[half_n_Num : n_Num] = window[0 : half_n_Num][::-1]   # Reverse
    windowed_data = window * dataSampleArray
    return windowed_data

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass ###