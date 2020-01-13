import numpy as np

def pad2DArrayWithZeros(numpyArray):
    shape = numpyArray.shape
    diff = shape[0] - shape[1]
    if diff == 0:
        return numpyArray
    dimToPad = 1 if diff > 0 else 0
    padding = [[0, 0], [0, 0]]
    padding[dimToPad][1] = abs(diff)
    return np.pad(numpyArray, padding, mode='constant')
