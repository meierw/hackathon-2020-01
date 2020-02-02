'''Training the page orientation model'''

import os
import cv2
import numpy as np
import helper

# TODO:
#   - Convert image to a square (done)
#       - Add whitespace (zeros) instead of squishing, to give the model hints about which is the long side
#   - Resize image to some fixed square value (done)
#   - Train to identify pieces of paper from open books
#       - Don't do anything with open books, yet. Those will need additional processing
#   - Train to identify rotation of found pieces of paper

# TODO (maybe):
#   - Concatenate all four rotations of image into one square image

DATA_DIR = os.path.dirname(__file__) + '/data'
TEST_IMAGES, TEST_LABELS = helper.load_samples(DATA_DIR + '/test')

print(TEST_LABELS[3])
cv2.imshow('hello', TEST_IMAGES[3])
cv2.waitKey()
