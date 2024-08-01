import tensorflow as tf
import numpy as np
import pandas as pd
import re
from shutil import copyfile
from glob import glob
from json import load, dump
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D,\
    Activation
from tensorflow.keras import Model, Sequential
import os
from os.path import basename
from time import time

print(tf.__version__)
tiny_class_dict = load(open('./data/class_dict_10.json', 'r'))
print(tiny_class_dict)
WIDTH = 64
HEIGHT = 64
EPOCHS = 1000
PATIENCE = 50
LR = 0.001
NUM_CLASS = 10
BATCH_SIZE = 32

# "Compile" the model with loss function and optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy()

tiny_vgg = tf.keras.models.load_model('trained_vgg_best.h5')

tiny_val_class_dict = load(open('./data/val_class_dict_10.json', 'r'))

test_images = './data/class_10_val/map_images/*.JPEG'

import glob
files = glob.glob(test_images)
print(files)

for test_image in files:
    print(test_image)
    # test_image = './data/class_10_val/map_images/Forest_1465.JPEG'

    # Read image and convert the image to [0, 1] range 3d tensor
    img = tf.io.read_file(test_image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [WIDTH, HEIGHT])

    img = tf.expand_dims(img, 0)  # Create batch axis

    img_predictions = tiny_vgg.predict(img)

    class_names = ["highway", "forest", "river", "permanent crop", "industrial", "annual crop", "sea or lake", "herbaceous", "residential", "pasture"]
    pred_label = class_names[np.argmax(np.round(img_predictions,2))]
    print(" Predicted label is :: "+ pred_label)


    Cartographer: Rebeca Nunes
Date: 01/08/24
Class: Data to Knowledge
Professor Little
University of Muenster