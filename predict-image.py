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

# Create test dataset
def process_path_test(path):
    """
    Get the (class label, processed image) pair of the given image path. This
    funciton uses python primitives, so you need to use tf.py_funciton wrapper.
    This function uses global variables:

        WIDTH(int): the width of the targeting image
        HEIGHT(int): the height of the targeting iamge
        NUM_CLASS(int): number of classes

    The filepath encoding for test images is different from training images.

    Args:
        path(string): path to an image file
    """

    # Get the class
    print(path)
    path = path.numpy()
    print(path)
    image_name = basename(path.decode('ascii'))
    # print(image_name)
    label_index = tiny_val_class_dict[image_name]['index']

    # Convert label to one-hot encoding
    label = tf.one_hot(indices=[label_index], depth=NUM_CLASS)
    label = tf.reshape(label, [NUM_CLASS])
    print(label.numpy())

    # Read image and convert the image to [0, 1] range 3d tensor
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [WIDTH, HEIGHT])

    return(img, label)

def prepare_for_training(dataset, batch_size=32, cache=True,
                         shuffle_buffer_size=1000):

    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    # Only shuffle elements in the buffer size
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Pre featch batches in the background
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

@tf.function
def test_step(image_batch, label_batch):
    predictions = tiny_vgg(image_batch)
    test_loss = loss_object(label_batch, predictions)

    test_mean_loss(test_loss)
    test_accuracy(label_batch, predictions)


tiny_val_class_dict = load(open('./data/val_class_dict_10.json', 'r'))

test_images = './data/class_10_val/test_images/val_10.JPEG'

test_path_dataset = tf.data.Dataset.list_files(test_images)

test_labeld_dataset = test_path_dataset.map(
    lambda path: tf.py_function(
        process_path_test,
        [path],
        [tf.float32, tf.float32]
    )
)
# print(test_labeld_dataset)

test_dataset = prepare_for_training(test_labeld_dataset,
                                    batch_size=BATCH_SIZE)

# Test on hold-out test images
test_mean_loss = tf.keras.metrics.Mean(name='test_mean_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

for image_batch, label_batch in test_dataset:
    # print(image_batch)
    # print(image_batch.numpy())
    print(image_batch.numpy().shape)
    print(label_batch.numpy())
    my_prediction = tiny_vgg.predict(image_batch)
    print(my_prediction)
    index = np.argmax(my_prediction)
    print(index)

    test_step(image_batch, label_batch)
    print(label_batch.numpy())


template = '\ntest loss: {:.4f}, test accuracy: {:.4f}'
print(template.format(test_mean_loss.result(),
                      test_accuracy.result() * 100))

