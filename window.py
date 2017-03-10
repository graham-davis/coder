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

def TransitionSineWindow(dataSampleArray,a,b):
    """
    Returns a copy of the dataSampleArray sine-windowed with
    a sine window of length a+b where a+b=len(dataSampleArray).
    a = left-half length, b = right-half length
    """
    # verify that a+b = n_Num
    if a + b != len(dataSampleArray):
        raise ValueError("a+b must equal length of dataSampleArray")

    # build window and then window data
    n_Num = 2*a
    n_Num2 = 2*b
    n = np.arange(a)
    n2 = np.arange(b)+b
    window1 = np.sin(np.pi * (n + 0.5) / n_Num)
    window2 = np.sin(np.pi * (n2 + 0.5) / n_Num2)
    window = np.append(window1,window2)
    windowed_data = window * dataSampleArray
    return windowed_data
        
    
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
    window[half_n_Num : n_Num] = window[0 : half_n_Num][::-1] # Reverse
    windowed_data = window * dataSampleArray
    return windowed_data

def TransitionKBDWindow(dataSampleArray,a,b,alpha=4):
    """
    Returns a copy of the dataSampleArray KBD-windowed with
    a KBD window of length a+b where a+b=len(dataSampleArray).
    a = left-half length, b = right-half length
    """
    # verify that a+b = n_Num
    if a + b != len(dataSampleArray):
        raise ValueError("a+b must equal length of dataSampleArray")
    
    # build window and then window data
    n_Num = 2*a
    n_Num2 = 2*b
    w_kb = np.arange(a + 1)
    w_kb = np.i0(np.pi * alpha * np.sqrt(1.0 - np.square(w_kb * 2.0 / a - 1.0))) / np.i0(np.pi * alpha)
    w_kb2 = np.arange(b + 1)
    w_kb2 = np.i0(np.pi * alpha * np.sqrt(1.0 - np.square(w_kb2 * 2.0 / b - 1.0))) / np.i0(np.pi * alpha)
    window1 = np.sqrt(np.cumsum(np.square(w_kb[0 : a])) / np.sum(np.square(w_kb)))
    window2 = np.sqrt(np.cumsum(np.square(w_kb2[0 : b])) / np.sum(np.square(w_kb2)))
    window = np.append(window1,window2[::-1])
    windowed_data = window * dataSampleArray
    return windowed_data
    
#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass ###