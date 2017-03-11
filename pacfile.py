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
        nMDCTLines = half the MDCT block size (block switching not supported)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBands = a ScaleFactorBands object
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands, contains an allocation of
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
        nMDCTLines = half the MDCT block size (format does not support block switching)
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

        tag                 4 byte file tag equal to "PAC "
        sampleRate          little-endian unsigned long ("<L" format in struct)
        nChannels           little-endian unsigned short("<H" format in struct)
        numSamples          little-endian unsigned long ("<L" format in struct)
        nMDCTLines          little-endian unsigned long ("<L" format in struct)
        nScaleBits          little-endian unsigned short("<H" format in struct)
        nMantSizeBits       little-endian unsigned short("<H" format in struct)
        nSFBands            little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
            nLines[iBand]   little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                overallScale[iCh]                       nScaleBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if bitAlloc[iCh][iBand]:
                        for m in nLines[iBand]:
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
        (sampleRate, nChannels, numSamples, nMDCTLines, nScaleBits, nMantSizeBits) \
                 = unpack('<LHLLHH',self.fp.read(calcsize('<LHLLHH')))
        nBands = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLines=  unpack('<'+str(nBands)+'H',self.fp.read(calcsize('<'+str(nBands)+'H')))
        sfBands=ScaleFactorBands(nLines)
        # load up a CodingParams object with the header data
        myParams=CodingParams()
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines = myParams.nSamplesPerBlock = nMDCTLines
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # add in scale factor band information
        myParams.sfBands =sfBands
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels): overlapAndAdd.append( np.zeros(nMDCTLines, dtype=np.float64) )
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
            overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)  # overall scale factor
            hTable = pb.ReadBits(codingParams.nHuffTableBits)   # huffman table code
            scaleFactor=[]
            bitAlloc=[]
            mantissa=np.zeros(codingParams.nMDCTLines,np.int32)  # start w/ all mantissas zero

            for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                ba = pb.ReadBits(codingParams.nMantSizeBits)
                if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                bitAlloc.append(ba)  # bit allocation for this band
                scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                if bitAlloc[iBand]:
                    # read non huffman encoded mantissas
                    if hTable == 0:
                        # if bits allocated, extract those mantissas and put in correct location in matnissa array
                        m=np.empty(codingParams.sfBands.nLines[iBand],np.int32)
                        for j in range(codingParams.sfBands.nLines[iBand]):
                            m[j]=pb.ReadBits(bitAlloc[iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                        mantissa[codingParams.sfBands.lowerLine[iBand]:(codingParams.sfBands.upperLine[iBand]+1)] = m
                    # read huffman mantissas
                    else:
                        nHuffBits = pb.ReadBits(16)

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
                            mantissa[codingParams.sfBands.lowerLine[iBand]:(codingParams.sfBands.upperLine[iBand]+1)] = decoded
            # done unpacking data (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can unpack any custom data passed in the nBytes of data >

            # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
            decodedData = self.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)
            data[iCh] = np.concatenate( (data[iCh],np.add(codingParams.overlapAndAdd[iCh],decodedData[:codingParams.nMDCTLines]) ) )  # data[iCh] is overlap-and-added data
            codingParams.overlapAndAdd[iCh] = decodedData[codingParams.nMDCTLines:]  # save other half for next pass

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
        if not codingParams.numSamples%codingParams.nMDCTLines:
            codingParams.numSamples += (codingParams.nMDCTLines
                        - codingParams.numSamples%codingParams.nMDCTLines) # zero padding for partial final PCM block

        # # also add in the delay block for the second pass w/ the last half-block (JH: I don't think we need this, in fact it generates a click at the end)
        # codingParams.numSamples+= codingParams.nMDCTLines  # due to the delay in processing the first samples on both sides of the MDCT block

        # write the coded file attributes
        self.fp.write(pack('<LHLLHH',
            codingParams.sampleRate, codingParams.nChannels,
            codingParams.numSamples, codingParams.nMDCTLines,
            codingParams.nScaleBits, codingParams.nMantSizeBits  ))
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBands=ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines,
                                                                codingParams.sampleRate)
                                )
        codingParams.sfBands=sfBands
        self.fp.write(pack('<L',sfBands.nBands))
        self.fp.write(pack('<'+str(sfBands.nBands)+'H',*(sfBands.nLines.tolist()) ))
        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLines,dtype=np.float64) )
        codingParams.priorBlock = priorBlock
        return


    def WriteDataBlock(self,data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # combine this block of multi-channel data w/ the prior block's to prepare for MDCTs twice as long
        fullBlockData=[]
        for iCh in range(codingParams.nChannels):
            fullBlockData.append( np.concatenate( ( codingParams.priorBlock[iCh], data[iCh]) ) )
        codingParams.priorBlock = data  # current pass's data is next pass's prior block data

        # (ENCODE HERE) Encode the full block of multi=channel data
        (scaleFactor,bitAlloc,mantissa, overallScaleFactor,hTables,hBits) = self.Encode(fullBlockData,codingParams)  # returns a tuple with all the block-specific info not in the file header

        # for each channel, write the data to the output file
        for iCh in range(codingParams.nChannels):

            # determine the size of this channel's data block and write it to the output file
            nBytes = codingParams.nScaleBits  # bits for overall scale factor
            for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
                nBytes += codingParams.nScaleBits    # mantissa bit allocation and scale factor for that sf band
                nBytes += codingParams.nHuffTableBits + codingParams.nMantSizeBits # huff table code allocation
                if bitAlloc[iCh][iBand]:
                    # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                    if hTables[iCh] == 0:
                        nBytes += bitAlloc[iCh][iBand]*codingParams.sfBands.nLines[iBand]  # no bit alloc = 1 so actuall alloc is one higher
            if hTables[iCh] is not 0:
                nBytes += hBits[iCh]  # Add huffman coded bit allocation to nBytes
            # end computing bits needed for this channel's data

            # CUSTOM DATA:
            # < now can add space for custom data, if desired>

            # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
            if nBytes%BYTESIZE==0:  nBytes /= BYTESIZE
            else: nBytes = nBytes/BYTESIZE + 1
            nBytes = int(np.ceil(nBytes))

            self.fp.write(pack("<L",int(nBytes))) # stores size as a little-endian unsigned long

            # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
            pb = PackedBits()
            pb.Size(nBytes)

            # now pack the nBytes of data into the PackedBits object
            pb.WriteBits(overallScaleFactor[iCh],codingParams.nScaleBits)  # overall scale factor
            pb.WriteBits(hTables[iCh],codingParams.nHuffTableBits)         # huff table code
            iMant=0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)
            for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                ba = bitAlloc[iCh][iBand]
                if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                if bitAlloc[iCh][iBand]:
                    if hTables[iCh] == 0:
                        for j in range(codingParams.sfBands.nLines[iBand]):
                            pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                        iMant += codingParams.sfBands.nLines[iBand]  # add to mantissa offset if we passed mantissas for this band
                    else:
                        nHuffBits = mantissa[iCh][iBand][0]
                        pb.WriteBits(nHuffBits, 16)
                        nChunks = len(mantissa[iCh][iBand])
                        if nHuffBits%16:
                            mantissa[iCh][iBand][nChunks-1] = mantissa[iCh][iBand][nChunks-1] >> (16-(nHuffBits%16))
                        for j in range(1, nChunks-1):
                            pb.WriteBits(mantissa[iCh][iBand][j], 16)
                            nHuffBits -= 16
                        pb.WriteBits(mantissa[iCh][iBand][nChunks-1], nHuffBits)

            # done packing (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can add in custom data if space allocated in nBytes above>

            # finally, write the data in this channel's PackedBits object to the output file
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels
        return

    def Close(self,codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            data = [ np.zeros(codingParams.nMDCTLines,dtype=np.float),
                     np.zeros(codingParams.nMDCTLines,dtype=np.float) ]
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


    input_filename = "Audio/spmg54_1.wav"
    coded_filename = "coded.pac"
    output_filename = "Output/output.wav"

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        coded_filename = sys.argv[1][:-4] + ".pac"
        output_filename = sys.argv[1][:-4] + "_decoded.wav"

    buildTable = 0  # flag to build new huffman table or not
    nTables = 5     # how many huffman tables to load

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

        # pass parameters to the output file
        if Direction == "Encode":
            # set additional parameters that are needed for PAC file
            # (beyond those set by the PCM file on open)
            codingParams.nMDCTLines = 1024
            codingParams.nScaleBits = 4
            codingParams.nMantSizeBits = 4
            codingParams.targetBitsPerSample = 2.9
            # tell the PCM file how large the block size is
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
            # Set block state
            #   0 - long block
            #   1 - start transition
            #   2 - short block
            #   3 - stop transition
            codingParams.state = 0
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
        
        while True:
            # Read next data block
            currentBlock=inFile.ReadDataBlock(codingParams)
            if not currentBlock: break  # we hit the end of the input file

            # don't write the first PCM block (it corresponds to the half-block delay introduced by the MDCT)
            if firstBlock and Direction == "Decode":
                firstBlock = False
                continue 

            # Only handle state transitions if we are encoding
            if Direction == "Encode":
                if previousBlock:
                    # Check for transient in currentBlock
                    if IsTransient(previousBlock[0], currentBlock[0]):
                        # Start transition window
                        if codingParams.state == 0 or codingParams.state == 3:
                            codingParams.state = 1
                        # Continue short block
                        else:
                            codingParams.state = 2
                    # No transient in current block
                    else:
                        # Begin end transition if current state is short block
                        if codingParams.state == 2 or codingParams.state == 1:
                            codingParams.state = 3
                        # Stay at long window
                        else: 
                            codingParams.state = 0

                # Update previousBlock
                previousBlock = currentBlock
                    
            outFile.WriteDataBlock(currentBlock,codingParams)
            sys.stdout.write(".")  # just to signal how far we've gotten to user
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

        pickle.dump(encodingTree, open("./Trees/encodingTree5", "w"), 0)
        pickle.dump(encodingMap, open("./Maps/encodingMap5", "w"), 0)

    elapsed = time.time()-elapsed
    print "\nDone with Encode/Decode test\n"
    print elapsed ," seconds elapsed"
