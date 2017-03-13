﻿"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""
import sys
import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow,TransitionSineWindow # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)
import huffman as huff

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import *  #allocates bits to scale factor bands given SMRs


def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1.*(1<<overallScaleFactor)
    if codingParams.state == 0:
        codingParams.a = codingParams.nMDCTLinesLong
        codingParams.b = codingParams.nMDCTLinesLong
        sfBands = codingParams.sfBandsLong
    elif codingParams.state == 1:
        codingParams.a = codingParams.nMDCTLinesLong
        codingParams.b = codingParams.nMDCTLinesShort
        sfBands = codingParams.sfBandsTrans
    elif codingParams.state == 2:
        codingParams.a = codingParams.nMDCTLinesShort
        codingParams.b = codingParams.nMDCTLinesShort
        sfBands = codingParams.sfBandsShort
    else:
        codingParams.a = codingParams.nMDCTLinesShort
        codingParams.b = codingParams.nMDCTLinesLong
        sfBands = codingParams.sfBandsTrans
        
    N = codingParams.a + codingParams.b
    halfN = N/2
    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN,dtype=np.float64)
    iMant = 0
    for iBand in range(sfBands.nBands):
        nLines = sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level


    # IMDCT and window the data for this channel
    if codingParams.state == 0 or codingParams.state == 2:
        data = SineWindow( IMDCT(mdctLine, halfN, halfN) )
    else:
        data = TransitionSineWindow( IMDCT(mdctLine, codingParams.a, codingParams.b),codingParams.a,codingParams.b)

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    hTables = []
    hBits = []

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s,b,m,o,t,h) = EncodeSingleChannel(data[iCh],codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
        hTables.append(t)
        hBits.append(h)
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor,hTables,hBits)


def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""
    # prepare various constants
    N = codingParams.a + codingParams.b
    halfN = N/2
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders

    if codingParams.state == 0:
        sfBands = codingParams.sfBandsLong
    elif codingParams.state == 1 or codingParams.state == 3:
        sfBands = codingParams.sfBandsTrans
    else:
        sfBands = codingParams.sfBandsShort

    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -= 34         # Block type + nBytes bits
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nHuffTableBits # less huff table type bits
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    # bitBudget += codingParams.reservoir # add reservoir bits to bit budget
    bitsFromRes = np.min([codingParams.reservoir, bitBudget])
    codingParams.reservoir -= bitsFromRes

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data

    if codingParams.state == 0 or codingParams.state == 2:
        mdctTimeSamples = SineWindow(data)
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]
    else:
        mdctTimeSamples = TransitionSineWindow(data,codingParams.a,codingParams.b)
        mdctLines = MDCT(mdctTimeSamples, codingParams.a, codingParams.b)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget+bitsFromRes, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    nHuffMaps = len(codingParams.encodingMaps)
    mHuff=[]
    huffBits=[]
    for h in range(nHuffMaps):
        mHuff.append([])
        huffBits.append(0)
    iMant=0
    for iBand in range(sfBands.nBands):
        nLines= sfBands.nLines[iBand]
        if nLines and bitAlloc[iBand]:      # Only encode mantissas if lines exist in current band
            lowLine = sfBands.lowerLine[iBand]
            highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
            scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
            scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
            # store FP coded mantissa
            m = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            mantissa[iMant:iMant+nLines] = m

            for h in range(nHuffMaps):
                # store Huffman coded mantissa
                huffCode = huff.encode(m, codingParams.encodingMaps[h])
                mHuff[h].append(huffCode)
                huffBits[h] += codingParams.nHuffLengthBits + huffCode[0]
            # increment starting index
            iMant += nLines
        else:
            for h in range(nHuffMaps):
                mHuff[h].append([])

    # If building freq table, at mantissas to freq table
    if codingParams.buildTable:
        codingParams.freqTable = huff.buildFrequencyTable(codingParams.freqTable, mantissa)

    # Initialize optimal bits as non-huffman
    optimalBits = np.sum(np.multiply(bitAlloc,sfBands.nLines))
    huffTable = 0

    # check for optimal bit allocation
    for h in range(nHuffMaps):
        if huffBits[h] < optimalBits:
            huffTable = h + 1
            optimalBits = huffBits[h]
            mantissa = mHuff[h]

    # calculate rollover bits for bit reservoir
    codingParams.reservoir += np.max([bitBudget - optimalBits, 0])

    # else return normal fp mantissas
    return (scaleFactor, bitAlloc, mantissa, overallScale, huffTable, optimalBits)




