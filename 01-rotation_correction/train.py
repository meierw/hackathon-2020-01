'''Training the page orientation model'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras

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
IMAGE_SIZE = 300

print('Loading data')
TRAIN_IMAGES, TRAIN_LABELS = helper.load_samples(DATA_DIR + '/train', IMAGE_SIZE)
TEST_IMAGES, TEST_LABELS = helper.load_samples(DATA_DIR + '/test', IMAGE_SIZE)
print('Data loaded')

LAYER_SIZE = 20
MODEL = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
    keras.layers.Dense(LAYER_SIZE, activation='relu'),
    # keras.layers.Dropout(0.1),
    keras.layers.Dense(LAYER_SIZE, activation='relu'),
    # keras.layers.Dropout(0.1),
    keras.layers.Dense(LAYER_SIZE, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
MODEL.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

MODEL.fit(TRAIN_IMAGES, TRAIN_LABELS[:, 0], epochs=13)

MODEL.evaluate(TEST_IMAGES, TEST_LABELS[:, 0], verbose=2)

PREDICTIONS = MODEL.predict(TEST_IMAGES)

print('Done')
