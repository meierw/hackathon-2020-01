'''Training the page orientation model'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from tensorflow import keras
from datetime import datetime

import cv2
import numpy as np
import helper

# TODO:
#   - Train to identify rotation of found pieces of paper

DATA_DIR = os.path.dirname(__file__) + '/data'
IMAGE_SIZE = 300

print('Loading data')
TRAIN_IMAGES, TRAIN_LABELS = helper.load_samples(DATA_DIR + '/train', IMAGE_SIZE, paper_type='paper')
TEST_IMAGES, TEST_LABELS = helper.load_samples(DATA_DIR + '/test', IMAGE_SIZE, paper_type='paper')
print('Data loaded')

FIRST_LAYER_SIZE = 4000
LAYER_SIZE = 500
DROPOUT_SIZE = 0.1
MODEL = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
    keras.layers.Dense(FIRST_LAYER_SIZE, activation='relu'),
    keras.layers.Dense(LAYER_SIZE, activation='relu'),
    keras.layers.Dense(LAYER_SIZE, activation='relu'),
    keras.layers.Dense(LAYER_SIZE, activation='relu'),
    keras.layers.Dense(LAYER_SIZE, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
MODEL.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

MODEL.fit(TRAIN_IMAGES, TRAIN_LABELS[:, 1], epochs=15)

TEST_LOSS, TEST_ACCURACY = MODEL.evaluate(TEST_IMAGES, TEST_LABELS[:, 1], verbose=2)

PREDICTIONS = MODEL.predict(TEST_IMAGES)

if TEST_ACCURACY > 0.7:
    ACC = str(TEST_ACCURACY).split('.')[1]
    MODEL.save(f'predict_paper_rotation_{ACC}.h5')

print('Done')
