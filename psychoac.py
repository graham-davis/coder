import numpy as np
from window import *
from mdct import *

MAX_SPL = 96.0
MIN_SPL = -30.0

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """

    min_intensity = Intensity(MIN_SPL)

    # set intensity floor
    if type(intensity) is np.ndarray: 
        # Handle array input 
        intensity[np.less(intensity, min_intensity)] = min_intensity
    else:
        # Handle single value input
        if intensity < min_intensity: intensity = min_intensity

    # Calculate SPL from intensity
    spl = MAX_SPL + np.multiply(10, np.log10(intensity))

    return spl

def Intensity(spl):
    """
    Returns the intensity (in units of the reference intensity level) for SPL spl
    """
    # Calculate intensity from SPL
    intensity = np.power(10., np.divide(np.subtract(spl, MAX_SPL), 10.))

    return intensity

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    # Convert to kHz
    freqs = np.divide(f, 1000.0)

    # Clip frequency vector at 10 Hz
    freqs[np.less(freqs, .01)] = .01

    # Calculate threshold
    threshold = np.multiply(3.64, np.power(freqs, -0.8))
    threshold = np.subtract(threshold, np.multiply(6.5, np.exp(np.multiply(-0.6, np.power(np.subtract(freqs, 3.3), 2)))))
    threshold = np.add(threshold, np.multiply((10**-3), np.power(freqs, 4)))

    return threshold


def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    # Convert to kHz
    freqs = np.divide(f, 1000.0)

    # Calculate bark
    bark = np.add(np.multiply(13, np.arctan(np.multiply(0.76, freqs))), \
        np.multiply(3.5, np.arctan(np.power(np.divide(freqs, 7.5), 2))))

    return bark

class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self,f,SPL,isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        # Initialize instance variables (frequency, SPL and delta)
        self.f = f
        self.SPL = SPL
        self.delta = 15 if isTonal else 5.5

    def IntensityAtFreq(self,freq):
        """The intensity of this masker at frequency freq"""
        # Convert freq to barks and return result of IntensityAtBark
        return self.IntensityAtBark(Bark(freq))

    def IntensityAtBark(self,z):
        """The intensity of this masker at Bark location z"""
        # Initialize masker dB at z
        mdB = 0

        # Calculate difference between maskee and masker frequency
        dz = z - Bark(self.f)

        # Calculate masker spreading function
        if abs(dz) <= 0.5:
            mdB = 0
        elif dz < -0.5:
            mdB = -27*(abs(dz)-0.5)
        else:
            mdB = (-27 + (0.367 * np.max([self.SPL-40, 0]))) * (abs(dz)-0.5)

        # Convert from spl to intensity and return
        return Intensity(self.SPL-self.delta+mdB)


    def vIntensityAtBark(self,zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        # Initialize masker dB vector
        mdbVec = np.zeros_like(zVec)

        # Calculate maskee/masker bark difference
        dz = np.subtract(zVec, Bark(self.f))

        # Calculate masker spreading function where |dz| <= 0.5
        mdbVec[np.less_equal(np.absolute(dz), 0.5)] = 0

        # Calculate masker spreading function where dz < -0.5
        i = np.less(dz, -0.5)
        mdbVec[i] = np.multiply(-27, np.subtract(np.absolute(dz[i]), 0.5))

        # Calculate masker spreading function where dz > 0.5
        i = np.greater(dz, 0.5)
        mdbVec[i] = np.multiply((-27 + (0.367 * np.max([self.SPL-40, 0]))), np.subtract(np.absolute(dz[i]), 0.5))

        # Convert from spl to intensity and return
        return Intensity(np.add(self.SPL - self.delta, mdbVec))


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, \
                1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 24000]  

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    # Calculate MDCT line frequencies
    lines = np.arange(0, nMDCTLines)
    lines = np.multiply(lines, (sampleRate / (2. * nMDCTLines)))
    lines = np.add(lines, (sampleRate / (4. * nMDCTLines)))

    # Initialize result list
    result = np.zeros(len(flimit))

    # Loop through remaining MDCT lines
    for line in lines:
        index = 0
        while(True):
            if index is len(flimit): 
                result[index-1] += 1
                break
            if line <= flimit[index]:
                result[index] += 1
                break
            index += 1

    return result.astype(int)

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
        self.nBands = len(nLines)
        self.nLines = nLines

        self.lowerLine = np.zeros_like(nLines)
        self.upperLine = np.zeros_like(nLines)

        for index, line in enumerate(nLines):
            self.lowerLine[index+1:] += line

        self.upperLine = nLines + self.lowerLine - 1


def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    # Helper variables
    N = len(data)

    # Take fft of data and obtain magnitude response
    dFFT = np.absolute(np.fft.fft(HanningWindow(data)))
    dFFT = dFFT[0:N/2]

    # Calculate FFT intensity
    xIntensity = np.multiply(4./((N**2)), np.power(np.abs(dFFT), 2))

    # Calculate FFT frequency vector
    fftFreqs = np.linspace(0.0, sampleRate/2, N/2)
    # Shift vector for MDCT frequencies
    fftFreqs += sampleRate/(2*N)

    # Initialize buffer for peak picking
    buff = 0.03

    r = np.greater(dFFT[1:-1], np.add(dFFT[2:], buff))
    l = np.greater(dFFT[1:-1], np.add(dFFT[0:-2], buff))
    vPeakIndex = np.add(np.where(np.logical_and(r, l)), 1)
    vPeakIndex = vPeakIndex[0]

    # Initialize masking threshold array
    mThreshold = np.zeros_like(fftFreqs)

    # Create peak masker instance
    masker = Masker(0, 0)

    # Find peaks 
    for index in vPeakIndex:
        # Calculate peak SPL
        pSum = np.sum(np.power(dFFT[index-1:index+2], 2))
        masker.SPL = SPL((4./((N**2)))*(pSum))

        # Calculate peak frequency
        masker.f = np.sum(np.multiply(xIntensity[index-1:index+2], fftFreqs[index-1:index+2]))\
             / np.sum(xIntensity[index-1:index+2])

        # Find peak masking threshold and add to total masking threshold
        mThreshold = np.add(mThreshold, masker.vIntensityAtBark(Bark(fftFreqs)))

    # Add threshold in quiet to masking threshold
    mThreshold = np.add(mThreshold, Intensity(Thresh(fftFreqs)))

    # Return masked threshold
    return SPL(mThreshold)


def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency lines for the data
                            in data which have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  is an overall scale factor for the set of MDCT
                            frequency lines
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Sums their masking curves with the hearing threshold at each MDCT
                frequency location to the calculate absolute threshold at those
                points. Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    # Initialize SMR array
    SMRs = np.zeros(sfBands.nBands)

    # Calculate masked threshold
    mThresh = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)

    # Scale down MDCT data
    sMDCTdata = np.multiply(MDCTdata, (2**(-MDCTscale)))

    # Calculate SPL of MDCT data
    mdctIntensity = np.multiply(4., np.power(np.abs(sMDCTdata),2))
    mdctSPL = SPL(mdctIntensity)

    # Calculate SMR for each MDCT line
    lineSMRs = np.subtract(mdctSPL, mThresh)

    # Calculate max SMR for each critical band
    for iBand in range(0, sfBands.nBands):
        # Obtain band range in MDCT lines
        lLine = sfBands.lowerLine[iBand]
        uLine = sfBands.upperLine[iBand]
        # Find max SMR within these MDCT lines
        SMRs[iBand] = np.max(lineSMRs[lLine:uLine+1])

    return SMRs

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    pass

