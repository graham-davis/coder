import numpy as np
import window as w
import scipy.stats as st

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
    return np.power(10.0,(spl-96.0)/10.0) # intensity value from SPL

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    # if f is a single number less than 10, replace with 10
    if isinstance(f,(int,float)) and f < 10.0:
        f = 10.0
    # replace any f values less than 10 with 10
    elif not(isinstance(f,(int,float))):
        f[f < 10.0] = 10.0

    # threshold equation as taken from pg. 155
    return  3.64*np.power(f/1000.0,-0.8)-6.5*np.exp(-0.6*np.power(f/1000.0-3.3,2.0))+\
np.power(10.0,-3.0)*np.power(f/1000.0,4)

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    
    return 13.0*np.arctan((0.76*f)/1000.0)+3.5*np.arctan(np.power(f/7500.0,2))

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
        # initialize the member variables of Masker
        self.f = f
        self.SPL = SPL
        self.isTonal = isTonal

    def IntensityAtFreq(self,freq):
        """The intensity of this masker at frequency freq"""
        return 0 # TO REPLACE WITH YOUR CODE

    def IntensityAtBark(self,z):
        """The intensity of this masker at Bark location z"""
        # tonal mask drop
        deltaT = -16.0
        # noise mask drop
        deltaN = -6.0
        # calculate the Bark difference between masker (Bark(self.f) and maskee (z)
        dz = z - Bark(self.f)
        # compute the SPL of the masking curve at maskee
        spl = self.SPL*(np.absolute(dz) <= 0.5) + (self.SPL-27.0*(np.absolute(dz)-0.5))*(dz < -0.5) +\
        (self.SPL+(-27+0.367*np.max(self.SPL-40,0))*(np.absolute(dz)-0.5))*(dz > 0.5) +\
        self.isTonal*(deltaT) + (not self.isTonal)*(deltaN)
        # return the value as an intensity
        return Intensity(spl)

    def vIntensityAtBark(self,zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        # tonal mask drop
        deltaT = -16.0
        # noise mask drop
        deltaN = -6.0
        # calculate the Bark difference between masker (Bark(self.f) and maskee (z)
        dz = zVec - Bark(self.f)
        # compute the SPL of the masking curve at maskee
        spl = self.SPL*(np.absolute(dz) <= 0.5) + (self.SPL-27.0*(np.absolute(dz)-0.5))*(dz < -0.5) +\
        (self.SPL+(-27+0.367*np.max(self.SPL-40,0))*(np.absolute(dz)-0.5))*(dz > 0.5) +\
        self.isTonal*(deltaT) + (not self.isTonal)*(deltaN)
        # return the value as an intensity
        return Intensity(spl)


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = [100.,200.,300.,400.,510.,630.,770.,920.,1080.,1270.,1480.,1720.,2000.,\
                2320.,2700.,3150.,3700.,4400.,5300.,6400.,7700.,9500.,12000.,15500.,24000.]

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    # create flimits including lower bound of 0
    #flimit2 = [0]flimit
    
    # create vector of MDCT frequencies based on nMDCTLines and sampleRate
    f = (sampleRate*(np.arange(nMDCTLines)+0.5))/(2.0*nMDCTLines)
    
    # create list of arrays containing the indexes of the MDCT frequency lines
    # falling within each scale factor band where the first band is the closed
    # interval between 0-flimit[0] and all others are half-open intervals including
    # the upper limit and excluding the lower limit
    indArrays = [np.where((f <= flimit[0]))] + [np.where((f>flimit[i-1]) & (f <= flimit[i])) \
                                                for i in np.arange(len(flimit)-1)+1]

    # return the size of each array in indArrays which is the number of MDCT lines
    # falling within each scale factor band
    return [len(indArrays[i][0]) for i in range(len(indArrays))]

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLine[i in range(nBands)], upperLine[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
        # number of lines per band
        self.nLines = np.array(nLines)
        # number of bands is the size of nLines
        self.nBands = len(nLines)
        
        # initialize lowerLine and upperLine
        self.lowerLine = np.zeros(25,np.int32)
        self.upperLine = np.zeros(25,np.int32)
        # current line variable
        startInd = 0
        # loop through and compute line indexes
        for i in range(self.nBands):
            if self.nLines[i] == 0:
                self.lowerLine[i] = -1
                self.upperLine[i] = -1
            else:
                self.lowerLine[i] = startInd
                self.upperLine[i] = startInd + self.nLines[i]-1
                startInd = self.upperLine[i]+1

def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    # store block length
    N = len(data)    
    # scale factor bands
    sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(N/2,sampleRate,cbFreqLimits))

    # compute the FFT of the windowed signal
    X = np.fft.fft(w.HanningWindow(data),N)
    # take only the positive frequency components
    X = X[0:N/2]

    # get Hanning window to compute its average power
    hWin = w.HanningWindow(np.ones(N))
    hWinPower = (1.0/N)*np.sum(hWin*hWin)

    # compute the intensities
    Xi = (4.0/(N*N*hWinPower))*np.power(np.absolute(X),2.0)
    # compute the SPL
    Xspl = SPL(Xi)

    # frequency arrays for FFT and MDCT
    f = (sampleRate/2.0)*(np.arange(N/2))/(N/2.0)

    # find peaks, where a peak is a point that is strictly larger than
    # the point before it and the point after it
    Plow = Xspl[0:N/2-2]<Xspl[1:N/2-1]
    PlowDiff = np.absolute(Xspl[0:N/2-2]-Xspl[1:N/2-1])
    Phigh = Xspl[1:N/2-1]>Xspl[2:N/2]
    PhighDiff = np.absolute(Xspl[1:N/2-1]-Xspl[2:N/2])
    Pexists = (Plow & Phigh)*(np.arange(N/2-2))
    # index of the found peaks
    Pind = np.squeeze(np.array(np.nonzero(Pexists)))+1
    # intensity-weighted average frequency of the found peaks
    Pfreq = (Xi[Pind-1]*f[Pind-1]+Xi[Pind]*f[Pind]+Xi[Pind+1]*f[Pind+1])/(Xi[Pind-1]+Xi[Pind]+Xi[Pind+1])
    # aggregate SPL of the found peaks
    Pspl = SPL(Xi[Pind-1]+Xi[Pind]+Xi[Pind+1])
    
    # create mask determining which peaks have a drop-off of 7 dB on either side
    mask = np.logical_or(PlowDiff[Pind-1] > 7.0, PhighDiff[Pind-1] > 7.0)
    # remove any peaks that don't meet that criteria
    Pind = Pind[mask]
    Pfreq = Pfreq[mask]
    Pspl = Pspl[mask]

    # use FFT intensity to generate critical band noise maskers
    XiNoise = Xi
    # zero out lines used for the tonal maskers
    XiNoise[Pind] = 0
    XiNoise[Pind-1] = 0
    XiNoise[Pind+1] = 0
    # arrays to hold noise masker info
    cbNoiseSpl = np.zeros(sfBands.nLines.size)
    cbNoiseFreq = np.zeros(sfBands.nLines.size)
    # create the noise maskers by summing the critical band intensities
    # converting to SPL and using the geometric mean frequency as the
    # masker frequency
    for i in range(sfBands.nLines.size):
        # for the 1st critical band, replace 0 so the geometric mean works
        if i == 0:
            f[0] = 0.1
        # compute power from summed intensity across band (avoiding zeros)
        if np.sum(XiNoise[sfBands.lowerLine[i]:sfBands.upperLine[i]+1]) == 0.0:
            cbNoiseSpl[i] = SPL(0.00001)
        else:
            cbNoiseSpl[i] = SPL(np.sum(XiNoise[sfBands.lowerLine[i]:sfBands.upperLine[i]+1]))

        if sfBands.nLines[i]: 
            # compute freq from geometric mean
            cbNoiseFreq[i] = st.gmean(f[sfBands.lowerLine[i]:sfBands.upperLine[i]+1])
        else:
            cbNoiseFreq[i] = 0
        # put zero back afterwards
        if i == 0:
            f[0] = 0

    # remove any tonal maskers that fall below threshold in quiet
    mask = Thresh(Pfreq) < Pspl
    Pind = Pind[mask]
    Pfreq = Pfreq[mask]
    Pspl = Pspl[mask]
    # remove any noise maskers that fall below threshold in quiet
    mask = Thresh(cbNoiseFreq) < cbNoiseSpl
    cbNoiseSpl = cbNoiseSpl[mask]
    cbNoiseFreq = cbNoiseFreq[mask]

    # arrays to hold whether maskers are tonal, frequencies and spl values
    # form by concatenating tonal and noise values
    allMaskersTonal = np.concatenate((np.ones(Pfreq.size),np.zeros(cbNoiseFreq.size)))
    allMaskersFreq = np.concatenate((Pfreq,cbNoiseFreq))
    allMaskersSpl = np.concatenate((Pspl,cbNoiseSpl))
    # sort all by frequency
    allMaskersSpl = allMaskersSpl[np.argsort(allMaskersFreq)]
    allMaskersTonal = allMaskersTonal[np.argsort(allMaskersFreq)]
    allMaskersFreq = np.sort(allMaskersFreq)

    # todo: remove smaller masker of any maskers that are separated by less than 0.5 Barks

    # generate tonal masking intensity curve using the additive intensity curve method
    maskingIntensity = np.zeros(N/2)
    for i in range(len(allMaskersFreq)):
        temp = Masker(allMaskersFreq[i],allMaskersSpl[i],allMaskersTonal[i])
        maskingIntensity = maskingIntensity + temp.vIntensityAtBark(Bark(f + sampleRate/(2.0*N)))

    # add the intensity curve of the threshold in quiet and then convert to SPL
    test1 = SPL(maskingIntensity + Intensity(Thresh(f + sampleRate/(2.0*N))))

    # add the intensity curve of the threshold in quiet and then convert to SPL
    return SPL(maskingIntensity + Intensity(Thresh(f + sampleRate/(2.0*N))))

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
    
    # get length of block
    N = len(data)
    
    # get the KBD window to compute its average power
    kWin = w.KBDWindow(np.ones(N))
    #kWin = w.SineWindow(np.ones(N))
    kWinPower = (1.0/N)*np.sum(kWin*kWin)
    
    # calculate the MDCTdata SPL
    MDCTdataSpl = SPL((2.0/kWinPower)*np.power(np.absolute(np.power(2.0,-MDCTscale)*MDCTdata),2.0))

    # calculate the masking threshold
    threshold = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)
   
    # array to hold SMR values
    SMRs = np.zeros(len(sfBands.nLines))
    # compute the SMR values
    for i in range(len(sfBands.nLines)):
        # Only calculate SMR if there are lines in this band
        if sfBands.nLines[i]:
            SMRs[i] = np.max(MDCTdataSpl[sfBands.lowerLine[i]:sfBands.upperLine[i]+1])-\
                             np.min(threshold[sfBands.lowerLine[i]:sfBands.upperLine[i]+1])
        else:
            SMRs[i] = 0

    # return the calculate SMR values
    return SMRs # TO REPLACE WITH YOUR CODE

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE
