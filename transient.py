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
    from pcmfile import * # to get access to WAV file handling

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
    signal = []

    while True:

         # Read next data block
        currentBlock=inFile.ReadDataBlock(codingParams)
        if not currentBlock: break  # we hit the end of the input file

        signal.extend(currentBlock[0][:])

        if previousBlock:
            if IsTransient(previousBlock[0], currentBlock[0]):
                transients.append(sample)

        # Update currentBlock
        previousBlock = currentBlock
        sample += codingParams.nMDCTLines
        print sample
        print "----------------------------------"


    # end loop over reading/writing the blocks
    # close the files
    inFile.Close(codingParams)

    fig = figure(1)
    p = fig.add_subplot(1,1,1)
    p.plot(np.arange(0, len(signal)), signal)
    for n in range(len(transients)):
        p.plot((transients[n], transients[n]), (-1, 1), 'r-')

    for x in range(len(signal)/1024):
        p.plot((1024*x, 1024*x), (-1, 1), 'k--')

    show()
    
