from pcmfile import *
from quantize import *
from window import *
from mdct import *

# create the audio file objects of the appropriate audioFile type
inFile= PCMFile("../input/spmg54_1.wav")
outFile = PCMFile("../output/spmg54_1_fb1s3m5.wav")

# Block floating point flag
bfp = True
fDomain = True

# open input file and get its coding parameters
codingParams= inFile.OpenForReading()

# set additional coding parameters that are needed for encoding/decoding
codingParams.nSamplesPerBlock = 1024

# open the output file for writing, passing needed format/data parameters
outFile.OpenForWriting(codingParams)

# initialize FP quantization params
nScaleBits = 3
nMantBits = 5

# frequency domain quantization
if fDomain:
	# initialize overlap-add arrays
	nSamples = codingParams.nSamplesPerBlock
	priorBlock = np.zeros((codingParams.nChannels, nSamples))
	overlapAndAdd = np.zeros((codingParams.nChannels, nSamples))

	# Flag to signify last block
	lBlock = False
	fBlock = True

	# Read the input file and pass its data to the output file to be written
	while True:
		data = inFile.ReadDataBlock(codingParams)

		if not data: 
			# Set last data block to be all zeros
			data = np.zeros((codingParams.nChannels, nSamples))
			lBlock = True

		for iCh in range(codingParams.nChannels):
			# Concatenate current and prior block
		    nBlock = np.concatenate((priorBlock[iCh], data[iCh]))
		    # Update prior block
		    priorBlock[iCh] = data[iCh]
		    # Window new block
		    wBlock = SineWindow(nBlock)
		    # Compute MDCT
		    mdctBlock = MDCT(wBlock, nSamples,nSamples)

		    # Quantize frequency domain data
		    for index, sample in enumerate(mdctBlock):
		        s = ScaleFactor(sample, nScaleBits, nMantBits)
		        if bfp:
		        	m = Mantissa(sample, s, nScaleBits, nMantBits)
			        mdctBlock[index] = Dequantize(s, m, nScaleBits, nMantBits)
		        else:
			        m = MantissaFP(sample, s, nScaleBits, nMantBits)
			        mdctBlock[index] = DequantizeFP(s, m, nScaleBits, nMantBits)

		    # Compute IMDCT
		    wBlock = IMDCT(mdctBlock, nSamples, nSamples)
		    # Unwindow result
		    wBlock = SineWindow(wBlock)
		    # Overlap and add the left side
		    oaaBlock = np.add(overlapAndAdd[iCh], wBlock[0:codingParams.nSamplesPerBlock])
		    overlapAndAdd[iCh] = wBlock[codingParams.nSamplesPerBlock:]
		    # Set Result
		    data[iCh] = oaaBlock

		# Break loop if last block
		if lBlock: break

		# Discard first block
		if fBlock:
			fBlock = False
		else:
			outFile.WriteDataBlock(data, codingParams)

# Time domain quantization
else:
	# Read the input file and pass its data to the output file to be written
	while True:
		# read input file into buffer
		data=inFile.ReadDataBlock(codingParams)

		if not data: break  # we hit the end of the input file

		for iCh in range(codingParams.nChannels):
			for index, sample in enumerate(data[iCh]):
				scale = ScaleFactor(sample, nScaleBits, nMantBits)
				if bfp:
					data[iCh][index] = Dequantize(scale, Mantissa(sample, scale, nScaleBits, nMantBits), nScaleBits, nMantBits)
				else:
					data[iCh][index] = DequantizeFP(scale, MantissaFP(sample, scale, nScaleBits, nMantBits), nScaleBits, nMantBits)

		outFile.WriteDataBlock(data,codingParams)
	# end loop over reading/writing the blocks

# close the files
inFile.Close(codingParams)
outFile.Close(codingParams)