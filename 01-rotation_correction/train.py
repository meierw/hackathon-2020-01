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

train_dir = os.path.dirname(__file__) + '/data/train'
# img = cv2.imread(f'{train_dir}/d0dcbcd6cd9243a38882210c6355078f.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread(f'{train_dir}/d3a855c97948435b9f3d2d140a941e66.jpg', cv2.IMREAD_GRAYSCALE)
img = helper.convert_to_square(img, 1000)
img = cv2.bitwise_not(img)
img = img / 255

cv2.imshow('hello', img)
cv2.waitKey()
