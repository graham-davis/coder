"""
Graham Davis / Homework 3
window.py -- Defines functions to window an array of data samples
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import mdct as mdct
from matplotlib.pyplot import figure, show

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    # Initialize helper variables
    N = dataSampleArray.size
    n = np.arange(0, N)

    # Calculate sine windowed signal
    result = np.multiply(dataSampleArray, np.sin(np.divide(np.multiply(np.pi, np.add(n, 0.5)), N)))

    return result


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    # Initialize helper variables
    N = dataSampleArray.size
    n = np.arange(0, N)

    # Calculate hanning windowed signal
    result = np.multiply(dataSampleArray, np.multiply(0.5,np.subtract(1, np.cos(np.divide(np.multiply(2*np.pi, np.add(n, 0.5)),N)))))
    
    return result


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following pp. 108-109 and pp. 117-118 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """
    n= len(dataSampleArray)
    kb = np.arange(n/2+1) # from 0 to n/2 (n/2+1 values)
    kb = np.i0(np.pi*alpha*np.sqrt(1.0 - (4.0*kb/n - 1.0)**2))/np.i0(np.pi*alpha)
    d = np.zeros(n) # allocate memory
    denom = sum(kb**2) # denominator to normalize running sum
    d[:n/2] = np.cumsum(kb[:-1]**2)/denom # 1st half is normalized running sum d[n/2:] = d[:n/2][::-1] # 2nd half is just the reverse of 1st half
    d = np.sqrt(d) # take square root of elements # window samples and return
    d *= dataSampleArray 
    return np.array(d)

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### Code for question 2f ###

    # Initialize test signal
    nSamples = 1024
    n = np.arange(0, nSamples)
    fs = 44100
    T = 1.0/fs
    freq = 2000
    x = np.cos(np.divide(np.multiply(2*np.pi*freq, n),fs))

    # Window the signal
    sineX = SineWindow(x)
    hannX = HanningWindow(x)

    # Recreate window to calculate power
    sineWindow = np.divide(sineX,x)
    hannWindow = np.divide(hannX,x)

    # Calculate window power
    sinePower = np.divide(np.sum(np.abs(sineWindow)**2),nSamples)
    hannPower = np.divide(np.sum(np.abs(sineWindow)**2),nSamples)

    # Compute FFT and MDCT
    sineFFT = np.fft.fftshift(np.fft.fft(sineX))
    hannFFT = np.fft.fftshift(np.fft.fft(hannX))
    sineMDCT = MDCT(sineX, nSamples/2, nSamples/2, False)

    # Calculate dB SPL for each spectrum
    sfSPL = 96 + 10*np.log10((4/((nSamples**2)*sinePower))*np.abs(sineFFT)**2)
    hfSPL = 96 + 10*np.log10((4/((nSamples**2)*hannPower))*np.abs(hannFFT)**2)
    smSPL = 96 + 10*np.log10((4/(nSamples*sinePower))*np.abs(sineMDCT)**2)
    
    #  Initialize frequency axis for fft and mdct seperately
    fftFreqs = np.linspace(0.0, 1.0/(2*T), nSamples/2)
    mdctFreqs = np.linspace(0.0, 1.0/(2*T), nSamples/2)

    # Plot results
    fig = figure(1)
    p = fig.add_subplot(1,1,1)
    p.plot(fftFreqs, sfSPL[nSamples/2:], label="FFT of Sine Windowed x[n]")
    p.plot(fftFreqs, hfSPL[nSamples/2:], label="FFT of Hann Windowed x[n]")
    p.plot(mdctFreqs, smSPL, label="MDCT of Sine Windowed x[n]")
    p.legend(loc=2)
    p.set_ylabel("dB SPL")
    p.set_xlabel("Frequency (Hz)")
    p.set_title("FFT/MDCT Frequency Analysis of Sine and Hann Windowed Signals (Zoomed)")
    p.set_ylim([-50, 100])
    p.set_xlim([1, 24000])
    p.set_xscale('log')
    show()





