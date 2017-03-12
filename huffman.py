import numpy as np
import heapq
import cPickle as pickle
import numpy as np

BITS = 16


class HuffmanNode:
    def __init__(self, zero=None, one=None):
        self.zero = zero
        self.one = one

def isLeaf(node):
    return not isinstance(node, HuffmanNode)

def buildFrequencyTable(freqTable, data):
    for i in range(0, len(data)):
        thisKey = data[i]
        freqTable[thisKey] += 1
    return freqTable

def buildEncodingTree(freqTable):
    sorted_freqTable = list([(item[1], item[0]) for item in enumerate(freqTable)])
    heapq.heapify(sorted_freqTable)
    while (len(sorted_freqTable) > 1):
        # if (len(sorted_freqTable) % 256 == 0):
        #     print len(sorted_freqTable)
        zero = heapq.heappop(sorted_freqTable)
        one = heapq.heappop(sorted_freqTable)
        newNode = HuffmanNode(zero, one)
        heapq.heappush(sorted_freqTable, (zero[0] + one[0], newNode))
    return heapq.heappop(sorted_freqTable)[1]

def buildEncodingMap(root):
    encodingMap = dict()
    path = ''
    buildEncodingMapHelper(encodingMap, root, path)
    return encodingMap

def buildEncodingMapHelper(encodingMap, node, path):
    if node.zero is not None:
        if isinstance(node.zero[1], HuffmanNode):
            buildEncodingMapHelper(encodingMap, node.zero[1], path + '0')
        else:
            encodingMap[node.zero[1]] = path + '0'
    if node.one is not None:
        if isinstance(node.one[1], HuffmanNode):
            buildEncodingMapHelper(encodingMap, node.one[1], path + '1')
        else:
            encodingMap[node.one[1]] = path + '1'

def decode(codedData, encodingTree):
    codedData = np.asarray(codedData).astype(dtype=np.uint16)
    if len(codedData) <= 1:
        return []
    length = codedData[0]
    codedDataString = "".join(['{0:016b}'.format(data) for data in codedData[1:]])[:length]
    return decodeHelper(codedDataString, encodingTree)

def decodeHelper(codedDataString, encodingTree):
    decodedData = []
    currentEncodeTree = encodingTree
    for i in range(len(codedDataString)):
        if codedDataString[i] == '0':
            currentEncodeTree = currentEncodeTree.zero[1]
        elif codedDataString[i] == '1':
            currentEncodeTree = currentEncodeTree.one[1]
        if isLeaf(currentEncodeTree):
            decodedData.append(currentEncodeTree)
            currentEncodeTree = encodingTree
    decodedData = np.asarray(decodedData).astype(dtype=np.int16)
    return decodedData

def encode(data, encodingMap):
    data = np.asarray(data).astype(dtype=np.uint16)
    encodedDataString = "".join([encodingMap[i] for i in data])
    length = len(encodedDataString)
    if length == 0:
        return []
    toFill = BITS - ((length - 1) % BITS + 1)
    encodedDataString = encodedDataString + "0" * toFill
    encodedDataStringChunks = [encodedDataString[start: start + BITS]
                               for start in range(0, len(encodedDataString), BITS)]
    encodedData = [int(chunk, 2) for chunk in encodedDataStringChunks]
    return [length] + encodedData



# Testing code
if __name__ == "__main__":

    # ### YOUR TESTING CODE STARTS HERE ###
    # print '\n-------------GETTING FREQ COUNT-------------'
    x = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                  [3, 4, 5, 6, 6, 6, 6, 6, -7, 7, 8, 8, 8, 8, 8, 8, 10]])

    # Create/Update Table
    # nBands = 2
    # freqTable = [1e-16 for _ in range(2**16)]
    # for i in range(0, nBands):
    #     freqTable = buildFrequencyTable(freqTable, x[i])

    # encodingTree = buildEncodingTree(freqTable)
    # encodingMap = buildEncodingMap(encodingTree)

    # pickle.dump(encodingTree, open("encodingTreeTest", "w"), 0)
    # pickle.dump(encodingMap, open("encodingMapTest", "w"), 0)

    # Load table
    encodingTree = pickle.load(open("encodingTreeTest", "r"))
    encodingMap = pickle.load(open("encodingMapTest", "r"))


    # print list(encodingMap.items())
    # print [encodingMap[i] for i in x[1]]

    # Encode
    temp = encode([0,0,0,0,0], encodingMap)
    print temp
    print ['{0:016b}'.format(i) for i in temp]
    # Decode
    print decode([10, 43648], encodingTree)



