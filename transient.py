import numpy as np
from numpy.fft import fft
from window import HanningWindow
import psychoac as p
import matplotlib.pyplot as plt
import scipy.signal as s

def IsTransient(previousBlock, currentBlock):
    """
    Returns true if the next data block contains a transient signal
    """
    N = len(currentBlock)
    W = np.arange(1, N+1)
    
    b,a = s.butter(4,0.35,'high')
    
    # Calculations for spectral energy difference between current and previous block
    pEnergy = (1./N)*np.sum(np.multiply(W, np.power(np.abs(fft(HanningWindow(s.lfilter(b,a,previousBlock)))), 2)))
    cEnergy = (1./N)*np.sum(np.multiply(W, np.power(np.abs(fft(HanningWindow(s.lfilter(b,a,currentBlock)))), 2)))
    eDiff = cEnergy - pEnergy

    # Spectral energy differences within current block (first half vs. second half)
    fHalf = np.sum(np.multiply(W[0:N/2], np.power(np.abs(fft(HanningWindow(s.lfilter(b,a,currentBlock[0:N/2])))), 2)))
    sHalf = np.sum(np.multiply(W[0:N/2], np.power(np.abs(fft(HanningWindow(s.lfilter(b,a,currentBlock[N/2:])))), 2)))
    cDiff =  np.abs(fHalf - sHalf)

    # Time domain signal energy difference within block
    efHalf = np.sum(np.power(s.lfilter(b,a,currentBlock[0:N/2]),2.0))
    esHalf = np.sum(np.power(s.lfilter(b,a,currentBlock[N/2:]),2.0)) 
    teDiff = np.abs(efHalf - esHalf)

    if (eDiff > -500) and (eDiff > 0.6 or cDiff > 5000) and ((eDiff > 0.5 and cDiff > 450 and teDiff > .006) or eDiff > 20 or teDiff >= 1 or (cDiff > 2000 and teDiff > 0.4)):
        return True

#-----------------------------------------------------------------------------


#Testing code
if __name__ == "__main__":
    from pcmfile import * # to get access to WAV file handling

    #input_filename = "harp40_1.wav"
    # input_filename = "Audio/castanets.wav"
    input_filename = "Audio/harp40_1.wav"

    inFile= PCMFile(input_filename)
    codingParams=inFile.OpenForReading()  # (includes reading header)
    codingParams.nMDCTLines = 1024
    codingParams.nScaleBits = 4
    codingParams.nMantSizeBits = 4
    codingParams.targetBitsPerSample = 2.9
    # tell the PCM file how large the block size is
    codingParams.nSamplesPerBlock = codingParams.nMDCTLines
    # Set block state
    #   0 - long block
    #   1 - short block
    #   2 - start transition block
    #   3 - end transition block
    codingParams.state = 0

    # Read the input file and pass its data to the output file to be written
    firstBlock = True                                   # Set first block
    previousBlock = []

    sample = 0
    transients = []
    transSize = []
    signal = []

    test = 1
    while True:

         # Read next data block
        currentBlock=inFile.ReadDataBlock(codingParams)
        if not currentBlock: break  # we hit the end of the input file

        signal.extend(currentBlock[0][:])

        if previousBlock:
            #transSize.append(IsTransient(previousBlock[0],currentBlock[0]))
            #transSize.append(IsTransientPE(currentBlock[0],codingParams))
            # print test,
            if IsTransient(previousBlock[0],currentBlock[0]):
                transients.append(sample)
                # print '*'

        test += 1

        # Update currentBlock
        previousBlock = currentBlock
        sample += codingParams.nMDCTLines
        #print sample, ", ",

    # end loop over reading/writing the blocks
    # close the files
    inFile.Close(codingParams)

    #transSize = transSize/np.max(transSize)
    #for i in np.arange(len(transSize)-1):
    #    transSize[i] = transSize[i+1]-transSize[i]

    b,a = s.butter(4,0.35,'high')
    sigNorm = s.lfilter(b,a,signal)
    sigNorm = sigNorm/np.max(sigNorm)

    fig = plt.figure(1)
    p = fig.add_subplot(1,1,1)
    p.plot(np.arange(0, len(signal)), sigNorm)
    for n in range(len(transients)):
        p.plot((transients[n], transients[n]),(-1,1), 'r-')

    #print test.shape
    labelInts = np.arange(1,len(signal)/1024+1)
    labelInts += 1
    labels = map(str,labelInts)
    # for x in range(len(signal)/1024):
    #     p.plot((1024*x,1024*x), (-1, 1), 'k--')
    
    # plt.xticks((labelInts-1)*1024,labels)
    plt.show()
    
