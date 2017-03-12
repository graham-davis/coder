"""
pacfile.py -- Defines a PACFile class to handle reading and writing audio
data to an audio file holding data compressed using an MDCT-based perceptual audio
coding algorithm.  The MDCT lines of each audio channel are grouped into bands,
each sharing a single scaleFactor and bit allocation that are used to block-
floating point quantize those lines.  This class is a subclass of AudioFile.

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

Notes on reading and decoding PAC files:

    The OpenFileForReading() function returns a CodedParams object containing:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLinesLong = half the long MDCT block size
        nMDCTLinesShort = half the short MDCT block size
        nMDCTLinesTrans = half the transition MDCT block size
        nSamplesPerBlock = nMDCTLinesLong (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBandsLong = a ScaleFactorBands object corresponding to nMDCTLinesLong
        sfBandsShort = a ScaleFactorBands object corresponding to nMDCTLinesShort
        sfBandsTrans = a ScaleFactorBands object corresponding to nMDCTLinesTrans
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands[Long/Short/Trans], contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand


Notes on encoding and writing PAC files:

    When writing to a PACFile the CodingParams object passed to OpenForWriting()
    should have the following attributes set:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLinesLong = half the long MDCT block size
        nMDCTLinesShort = half the short MDCT block size
        nMDCTLinesTrans = half the transition MDCT block size
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        targetBitsPerSample = the target encoding bit rate in units of bits per sample

    The first three attributes (nChannels, sampleRate, and numSamples) are
    typically added by the original data source (e.g. a PCMFile object) but
    numSamples may need to be extended to account for the MDCT coding delay of
    nMDCTLines and any zero-padding done in the final data block

    OpenForWriting() will add the following attributes to be used during the encoding
    process carried out in WriteDataBlock():

        sfBands = a ScaleFactorBands object
        priorBlock = the prior block of audio data (initially all zeros)

    The passed ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand

Description of the PAC File Format:

    Header:

        tag                  4 byte file tag equal to "PAC "
        sampleRate           little-endian unsigned long ("<L" format in struct)
        nChannels            little-endian unsigned short("<H" format in struct)
        numSamples           little-endian unsigned long ("<L" format in struct)
        nMDCTLinesLong       little-endian unsigned long ("<L" format in struct)
        nMDCTLinesShort      little-endian unsigned long ("<L" format in struct)
        nScaleBits           little-endian unsigned short("<H" format in struct)
        nMantSizeBits        little-endian unsigned short("<H" format in struct)
        nSFBands             little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
          nLinesLong[iBand]  little-endian unsigned short("<H" format in struct)
          nLinesShort[iBand] little-endian unsigned short("<H" format in struct)
          nLinesTrans[iBand] little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                blockType[iCh]                          2 bits (0=long,1=start,2=short,3=stop)
                overallScale[iCh]                       nScaleBits bits
                hTable                                  nHuffTableBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if hTable != 0 
                        huffman coding length           nHuffLengthBits bits
                    if bitAlloc[iCh][iBand] (and nLines_blockType_[iBand]):
                        for m in nLines_blockType_[iBand]:
                            mantissa[iCh][iBand][m]     bitAlloc[iCh][iBand]+1 bits
                <extra custom data bits as long as space is included in nBytes>

"""

from audiofile import * # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codec    # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits  # defines the grouping of MDCT lines into scale factor bands
from huffman import HuffmanNode, buildFrequencyTable, buildEncodingTree, buildEncodingMap, decode
from transient import IsTransient
import sys

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np  # to allow conversion of data blocks to numpy's array object
MAX16BITS = 32767

class PACFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag='PAC '

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag=self.fp.read(4)
        if tag!=self.tag: raise "Tried to read a non-PAC file into a PACFile object"
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLinesLong, nMDCTLinesShort, nScaleBits, nMantSizeBits) \
                 = unpack('<LHLLLHH',self.fp.read(calcsize('<LHLLLHH')))
        nBands = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLinesLong =  unpack('<'+str(nBands)+'H',self.fp.read(calcsize('<'+str(nBands)+'H')))
        nLinesShort = unpack('<'+str(nBands)+'H',self.fp.read(calcsize('<'+str(nBands)+'H')))
        nLinesTrans = unpack('<'+str(nBands)+'H',self.fp.read(calcsize('<'+str(nBands)+'H')))
        sfBandsLong=ScaleFactorBands(nLinesLong)
        sfBandsShort=ScaleFactorBands(nLinesShort)
        sfBandsTrans=ScaleFactorBands(nLinesTrans)
        # load up a CodingParams object with the header data
        myParams=CodingParams()
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLinesLong = myParams.nSamplesPerBlock = nMDCTLinesLong
        myParams.nMDCTLinesShort = nMDCTLinesShort
        myParams.nMDCTLinesTrans = (nMDCTLinesLong + nMDCTLinesShort)/2
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # add in scale factor band information
        myParams.sfBandsLong = sfBandsLong
        myParams.sfBandsShort = sfBandsShort
        myParams.sfBandsTrans = sfBandsTrans
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels): overlapAndAdd.append( np.zeros(nMDCTLinesLong, dtype=np.float64) )
        myParams.overlapAndAdd=overlapAndAdd
        return myParams

    def ReadDataBlock(self, codingParams):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        # loop over channels (whose coded data are stored separately) and read in each data block
        data=[]
        for iCh in range(codingParams.nChannels):
            data.append(np.array([],dtype=np.float64))  # add location for this channel's data
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s=self.fp.read(calcsize("<L"))  # will be empty if at end of file
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd=codingParams.overlapAndAdd
                    codingParams.overlapAndAdd=0  # setting it to zero so next pass will just return
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L",s)[0] # read it as a little-endian unsigned long
            # read the nBytes of data into a PackedBits object to unpack
            pb = PackedBits()
            pb.SetPackedData( self.fp.read(nBytes) ) # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            if pb.nBytes < nBytes:  raise "Only read a partial block of coded PACFile data"

            # extract the data from the PackedBits object
            codingParams.state = pb.ReadBits(2) # read in blockType
            overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)  # overall scale factor
            hTable = pb.ReadBits(codingParams.nHuffTableBits)   # huffman table code
            scaleFactor=[]
            bitAlloc=[]

            if codingParams.state == 0:
                mantissa=np.zeros(codingParams.nMDCTLinesLong,np.int32)  # start w/ all mantissas zero
            elif codingParams.state == 1 or codingParams.state == 3:
                mantissa=np.zeros(codingParams.nMDCTLinesTrans,np.int32)  # start w/ all mantissas zero
            else:
                mantissa=np.zeros(codingParams.nMDCTLinesShort,np.int32)  # start w/ all mantissas zero

            for iBand in range(codingParams.sfBandsLong.nBands): # loop over each scale factor band to pack its data
                ba = pb.ReadBits(codingParams.nMantSizeBits)
                if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                bitAlloc.append(ba)  # bit allocation for this band
                scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                if bitAlloc[iBand]:
                    if codingParams.state == 0:
                        nMDCTLines = codingParams.nMDCTLinesLong
                        nLines = codingParams.sfBandsLong.nLines[iBand]
                        lowerLine = codingParams.sfBandsLong.lowerLine[iBand]
                        upperLine = codingParams.sfBandsLong.upperLine[iBand]
                    elif codingParams.state == 1 or codingParams.state == 3:
                        nMDCTLines = codingParams.nMDCTLinesTrans
                        nLines = codingParams.sfBandsTrans.nLines[iBand]
                        lowerLine = codingParams.sfBandsTrans.lowerLine[iBand]
                        upperLine = codingParams.sfBandsTrans.upperLine[iBand]
                    else:
                        nMDCTLines = codingParams.nMDCTLinesShort
                        nLines = codingParams.sfBandsShort.nLines[iBand]
                        lowerLine = codingParams.sfBandsShort.lowerLine[iBand]
                        upperLine = codingParams.sfBandsShort.upperLine[iBand]

                    
                    # read non huffman encoded mantissas
                    if hTable == 0:
                        m=np.empty(nLines,np.int32)
                        for j in range(nLines):
                            m[j]=pb.ReadBits(bitAlloc[iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                        mantissa[lowerLine:upperLine+1] = m
                    # read huffman mantissas
                    else:
                        nHuffBits = pb.ReadBits(codingParams.nHuffLengthBits)
                        nChunks = int(np.ceil(nHuffBits/16.))
                        huffBits = np.empty(nChunks+1).astype(dtype=np.uint16)
                        huffBits[0] = nHuffBits
                        for i in range(nChunks):  
                            bits = pb.ReadBits(np.min([16, nHuffBits]))
                            if (nHuffBits < 16):
                                bits = bits << (16-nHuffBits)
                            huffBits[i+1] = bits
                            nHuffBits = nHuffBits - 16
                        if huffBits.any():
                            decoded = decode(huffBits, codingParams.encodingTrees[hTable - 1])
                            mantissa[lowerLine:upperLine+1] = decoded

            # done unpacking data (end loop over scale factor bands)

            # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
            decodedData = self.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)
            data[iCh] = np.concatenate( (data[iCh],np.add(codingParams.overlapAndAdd[iCh],decodedData[:codingParams.a]) ) )  # data[iCh] is overlap-and-added data
            codingParams.overlapAndAdd[iCh] = decodedData[codingParams.a:]  # save other half for next pass

        # end loop over channels, return signed-fraction samples for this block
        return data


    def WriteFileHeader(self,codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if not codingParams.numSamples%codingParams.nMDCTLinesLong:
            codingParams.numSamples += (codingParams.nMDCTLinesLong
                        - codingParams.numSamples%codingParams.nMDCTLinesLong) # zero padding for partial final PCM block

        # # also add in the delay block for the second pass w/ the last half-block (JH: I don't think we need this, in fact it generates a click at the end)
        # codingParams.numSamples+= codingParams.nMDCTLines  # due to the delay in processing the first samples on both sides of the MDCT block

        # write the coded file attributes
        self.fp.write(pack('<LHLLLHH',
            codingParams.sampleRate, codingParams.nChannels,
            codingParams.numSamples, codingParams.nMDCTLinesLong, 
            codingParams.nMDCTLinesShort, codingParams.nScaleBits, codingParams.nMantSizeBits  ))
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBandsLong=ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLinesLong,
                                                                codingParams.sampleRate)
                                )
        sfBandsShort=ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLinesShort,
                                                                codingParams.sampleRate)
                                )
        sfBandsTrans=ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLinesTrans,
                                                                codingParams.sampleRate)
                                )
        codingParams.sfBandsLong =sfBandsLong
        codingParams.sfBandsShort=sfBandsShort
        codingParams.sfBandsTrans=sfBandsTrans
        self.fp.write(pack('<L',sfBandsLong.nBands))
        self.fp.write(pack('<'+str(sfBandsLong.nBands)+'H',*(sfBandsLong.nLines.tolist()) ))
        self.fp.write(pack('<'+str(sfBandsShort.nBands)+'H',*(sfBandsShort.nLines.tolist()) ))
        self.fp.write(pack('<'+str(sfBandsTrans.nBands)+'H',*(sfBandsTrans.nLines.tolist()) ))

        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLinesLong,dtype=np.float64) )
        codingParams.priorBlock = priorBlock
        return


    def WriteDataBlock(self,data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # if status for data[iCh] is start or short
        if codingParams.state == 1 or codingParams.state == 2:
            # there are this many subblocks
            numSubBlocks = codingParams.nMDCTLinesLong/codingParams.nMDCTLinesShort
        else: # the entire block is the subblock
            numSubBlocks = 1
       
        # store a copy of data in a temp array
        tempData = np.copy(data)
        # loop through the subblocks and process them (assuming same number of subblocks for all channels)
        for iBlk in range(numSubBlocks):
            fullBlockData=[]
            for iCh in range(codingParams.nChannels):
                # get sizes of half blocks to pass to encode process
                codingParams.a = codingParams.priorBlock[iCh].size
                codingParams.b = codingParams.nMDCTLinesLong/numSubBlocks
                test = tempData[iCh][:codingParams.b + iBlk*codingParams.b]
                fullBlockData.append( 
                        np.concatenate( 
                            ( codingParams.priorBlock[iCh], 
                                tempData[iCh][np.arange(codingParams.b) + iBlk*codingParams.b] ) 
                                      ) 
                                    )
                # current pass's data is next pass's prior block data
                codingParams.priorBlock[iCh] = tempData[iCh][np.arange(codingParams.b) + iBlk*codingParams.b]
            # (ENCODE HERE) Encode the full block of multi=channel data
            
            (scaleFactor,bitAlloc,mantissa, overallScaleFactor,hTables,hBits) = self.Encode(fullBlockData,codingParams)  # returns a tuple with all the block-specific info not in the file header
           
            
            if codingParams.state == 0: 
                sfBands = codingParams.sfBandsLong
            elif codingParams.state == 1 or codingParams.state == 3:
                sfBands = codingParams.sfBandsTrans
            else:
                sfBands = codingParams.sfBandsShort

            # for each channel, write the data to the output file
            for iCh in range(codingParams.nChannels):
                # determine the size of this channel's data block and write it to the output file

                nBytes = 2 + codingParams.nScaleBits  # bits for overall scale factor
                for iBand in range(sfBands.nBands): # loop over each scale factor band to get its bits
                    nBytes += codingParams.nScaleBits + codingParams.nHuffTableBits + codingParams.nMantSizeBits # huff table code allocation
                    # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                    if bitAlloc[iCh][iBand]:
                        # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                        nBytes += bitAlloc[iCh][iBand]*sfBands.nLines[iBand]  # no bit alloc = 1 so actuall alloc is one higher
                if hTables[iCh] is not 0:
                    nBytes += hBits[iCh]  # Add huffman coded bit allocation to nBytes
                # end computing bits needed for this channel's data

                # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
                if nBytes%BYTESIZE==0:  nBytes /= BYTESIZE
                else: nBytes = nBytes/BYTESIZE + 1
                self.fp.write(pack("<L",int(nBytes))) # stores size as a little-endian unsigned long

                # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
                pb = PackedBits()
                pb.Size(nBytes)

                # now pack the nBytes of data into the PackedBits object
                pb.WriteBits(codingParams.state,2) # block type
                pb.WriteBits(overallScaleFactor[iCh],codingParams.nScaleBits)  # overall scale factor
                pb.WriteBits(hTables[iCh],codingParams.nHuffTableBits)         # huff table code
                iMant=0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)
                for iBand in range(sfBands.nBands): # loop over each scale factor band to pack its data
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                    pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                    pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                    if bitAlloc[iCh][iBand]:
                        if hTables[iCh] == 0:
                            for j in range(sfBands.nLines[iBand]):
                                pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                            iMant += sfBands.nLines[iBand]  # add to mantissa offset if we passed mantissas for this band
                        else:
                            nHuffBits = mantissa[iCh][iBand][0]
                            pb.WriteBits(nHuffBits, codingParams.nHuffLengthBits)
                            nChunks = len(mantissa[iCh][iBand])
                            if nHuffBits%16:
                                mantissa[iCh][iBand][nChunks-1] = mantissa[iCh][iBand][nChunks-1] >> (16-(nHuffBits%16))
                            for j in range(1, nChunks-1):
                                pb.WriteBits(mantissa[iCh][iBand][j], 16)
                                nHuffBits -= 16
                            pb.WriteBits(mantissa[iCh][iBand][nChunks-1], nHuffBits)

                # finally, write the data in this channel's PackedBits object to the output file
                self.fp.write(pb.GetPackedData())
            # end loop over channels, done writing coded data for all channels
            
            # after 1 subblock of status 1, change to status 2
            if codingParams.state == 1:
                codingParams.state = 2
                codingParams.prevState = 1

        # end loop over subblocks, done writing coded data for all subblocks
        
        return

    def Close(self,codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            if codingParams.state == 0 or codingParams.state == 3:
                nMDCTLines = codingParams.nMDCTLinesLong
            else:
                nMDCTLines = codingParams.nMDCTLinesShort

            data = []

            for iCh in range(codingParams.nChannels): data.append( np.zeros(nMDCTLines, dtype=np.float) )
            
            self.WriteDataBlock(data, codingParams)
        self.fp.close()


    def Encode(self,data,codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data,codingParams)

    def Decode(self,scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)

#-----------------------------------------------------------------------------

# Testing the full PAC coder (needs a file called "input.wav" in the code directory)
if __name__=="__main__":
    import sys
    import time
    from pcmfile import * # to get access to WAV file handling

    input_filename = "Audio/gspi35_1.wav"
    coded_filename = "coded.pac"
    output_filename = "Output/full_german.wav"

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        coded_filename = sys.argv[1][:-4] + ".pac"
        output_filename = sys.argv[1][:-4] + "_decoded.wav"

    buildTable = 0  # flag to build new huffman table or not
    nTables = 6     # how many huffman tables to load

    encodingTrees = []
    encodingMaps = []
    print "\n\tLoading Huffman Tables ...",

    for i in range(1, nTables+1):
        sys.stdout.write(".")  # just to signal how far we've gotten to user
        sys.stdout.flush()
        treePath = './Trees/encodingTree%d' % i
        encodingTrees.append(pickle.load(open(treePath, 'r')))
        sys.stdout.write(".")  # just to signal how far we've gotten to user
        sys.stdout.flush()
        mapPath = './Maps/encodingMap%d' % i
        encodingMaps.append(pickle.load(open(mapPath, 'r')))

    print "\n\nRunning the PAC coder ({} -> {} -> {}):".format(input_filename, coded_filename, output_filename)
    elapsed = time.time()

    for Direction in ("Encode", "Decode"):
#    for Direction in ("Decode"):

        # create the audio file objects
        if Direction == "Encode":
            print "\n\tEncoding PCM file ({}) ...".format(input_filename),
            inFile= PCMFile(input_filename)
            outFile = PACFile(coded_filename)
        else: # "Decode"
            print "\n\tDecoding PAC file ({}) ...".format(coded_filename),
            inFile = PACFile(coded_filename)
            outFile= PCMFile(output_filename)
        # only difference is file names and type of AudioFile object

        # open input file
        codingParams=inFile.OpenForReading()  # (includes reading header)
        codingParams.nHuffTableBits = 3
        codingParams.nHuffLengthBits = 10

        # pass parameters to the output file
        if Direction == "Encode":
            # set additional parameters that are needed for PAC file
            # (beyond those set by the PCM file on open)

            codingParams.nMDCTLinesLong = 1024
            codingParams.nMDCTLinesShort = 128
            codingParams.nMDCTLinesTrans = (codingParams.nMDCTLinesLong+codingParams.nMDCTLinesShort)/2
            codingParams.nScaleBits = 4
            codingParams.nMantSizeBits = 4
            # dataRate = 128000
            # codingParams.targetBitsPerSample = dataRate/(1.0*codingParams.sampleRate)
            codingParams.targetBitsPerSample = 2.9
            # tell the PCM file how large the block size is
            codingParams.nSamplesPerBlock = codingParams.nMDCTLinesLong
            # Set block state
            #   0 - long block
            #   1 - start transition
            #   2 - short block
            #   3 - stop transition
            codingParams.state = 0
            codingParams.prevState = 0
            # Initialize bit reservoir
            codingParams.reservoir = 0
            # Huffman encoding map
            codingParams.buildTable = buildTable

            if buildTable:
                codingParams.freqTable = [1e-16 for _ in range(2**16)]
    
            codingParams.encodingMaps = encodingMaps


        else: # "Decode"
            # set PCM parameters (the rest is same as set by PAC file on open)
            codingParams.bitsPerSample = 16
            # Huffman decoding tree
            codingParams.encodingTrees = encodingTrees
        # only difference is in setting up the output file parameters

        # open the output file
        outFile.OpenForWriting(codingParams) # (includes writing header)

        # Read the input file and pass its data to the output file to be written
        previousBlock = []                                  # Initialize previous block
        firstBlock = True                                   # Set first block
        
        # read in the current block
        currentBlock=inFile.ReadDataBlock(codingParams)

        while True:
            # Read next (future) data block
            nextBlock=inFile.ReadDataBlock(codingParams)
            if not currentBlock: break  # we hit the end of the input file

            # don't write the first PCM block (it corresponds to the half-block delay introduced by the MDCT)
            if firstBlock and Direction == "Decode":
                firstBlock = False
                currentBlock = nextBlock
                continue 
           
            # Only handle state transitions if we are encoding
            if Direction == "Encode" and nextBlock:
                # Check for transient in currentBlock for any channel
                if True: #not previousBlock == []:
                    isTrans = False
                    for iCh in range(codingParams.nChannels):
                        if IsTransient(currentBlock[iCh], nextBlock[iCh]):
                            isTrans = True

                    if isTrans:
                        #print "isTrans",
                        # Start transition window
                        if codingParams.state == 0 or codingParams.state == 3:
                            codingParams.prevState = codingParams.state
                            codingParams.state = 1
                        # Continue short block
                        else:
                            codingParams.prevState = codingParams.state
                            codingParams.state = 2
                    # No transient in current block
                    else:
                        #print "not Trans",
                        # Begin end transition if current state is short block
                        if codingParams.state == 1 or (codingParams.state == 2 and not codingParams.prevState == 1):
                            codingParams.prevState = codingParams.state
                            codingParams.state = 3
                        # test to see if additional block of short blocks needed after transient
                        elif codingParams.prevState == 1 and codingParams.state == 2:
                            codingParams.prevState = codingParams.state
                            codingParams.state = 2
                        # Stay at long window
                        else: 
                            codingParams.prevState = codingParams.state
                            codingParams.state = 0

            # Update previousBlock, currentBlock
            previousBlock = currentBlock
            currentBlock = nextBlock

            if codingParams.state == 0:
                sys.stdout.write("_ ")  # just to signal how far we've gotten to user
            elif codingParams.state == 1:
                sys.stdout.write("/ ")
            elif codingParams.state == 2:
                sys.stdout.write("^ ")
            else:
                sys.stdout.write("\\ ")
            outFile.WriteDataBlock(previousBlock,codingParams)

            sys.stdout.flush()

        # end loop over reading/writing the blocks

        # close the files
        inFile.Close(codingParams)
        outFile.Close(codingParams)

        if Direction == "Encode" and buildTable:
            freqTable = codingParams.freqTable
    # end of loop over Encode/Decode

    if buildTable:
        print "\n\n\tBuilding Huffman Tables..."
        encodingTree = buildEncodingTree(freqTable)
        encodingMap = buildEncodingMap(encodingTree)

        pickle.dump(encodingTree, open("./Trees/encodingTree6", "w"), 0)
        pickle.dump(encodingMap, open("./Maps/encodingMap6", "w"), 0)

    elapsed = time.time()-elapsed
    print "\nDone with Encode/Decode test\n"
    print elapsed ," seconds elapsed"
