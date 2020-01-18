'''Helper methods for parsing images to feed into ML models'''
import cv2
import numpy as np

def convert_to_square(image, size):
    '''Resize an image array, and convert it to a square, by padding it with zeros'''
    shape = image.shape
    shorter_dim = get_shorter_dim(shape)
    longer_dim = int(not shorter_dim)
    if shorter_dim == -1:
        return cv2.resize(image, (size, size))
    dsize = [0, 0]
    dsize[shorter_dim] = size
    dsize[longer_dim] = int(size * (shape[shorter_dim] / shape[longer_dim]))
    result = cv2.resize(image, tuple(dsize))
    return pad_with_zeros(result)

def pad_with_zeros(image):
    '''Pad an an image array with zeros to give it a square shape'''
    shape = image.shape
    shorter_dim = get_shorter_dim(shape)
    if shorter_dim == -1:
        return image
    diff = shape[0] - shape[1]
    padding = [[0, 0], [0, 0]]
    padding[shorter_dim][1] = abs(diff)
    return np.pad(image, padding, mode='constant')

def get_shorter_dim(shape):
    '''Find the shortest dimension from a given 2D shape'''
    if shape[0] == shape[1]:
        return -1
    return int(shape[1] < shape[0])
