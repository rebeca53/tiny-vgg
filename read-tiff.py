import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tiny_vgg = tf.keras.models.load_model('trained_vgg_best.h5')
WIDTH = 64
HEIGHT = 64
test_image = './ms-data/class_10_val/map_images/Industrial_473.tif'

# attempt with tensorflow.io
# img loaded has all pixels equal to 4
# LIMITATION: RGB + A, no other channels
# import tensorflow_io as tfio
# img = tf.io.read_file(test_image)
# img = tfio.experimental.image.decode_tiff(img) #RGBA image
# img = img[:,:,:3] # Remove alpha channel

# attempt with tensorflow.io
# has a limit of channels (<=4)
# import cv2
# img = cv2.imread(test_image, cv2.IMREAD_UNCHANGED)
# print(img.shape) 
# channels = cv2.split(img)

# attempt with tifffile -- IN PROGRESS
# it imports all channels, but it display nothing!
import tifffile


training_images = './ms-data/class_10_train/*/images/*.tif'

import glob
from os.path import basename
import re
from json import load, dump
from skimage import exposure, img_as_ubyte
def normalize(image):
        image = (image - image.min()) / (image.max() - image.min())
        return image




# Define the input and output ranges for scaling
input_min = 0
input_max = 2750
output_min = 1
output_max = 255

# Apply scaling to each band
def scale_band(band,  input_min = 0, input_max = 2750, output_min = 1, output_max = 255):
    scaled_band = np.clip(((band - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min,
                          output_min, output_max)
    return scaled_band.astype(np.uint8)


files = glob.glob(training_images)
print(files)
NUM_CLASS = 10
tiny_class_dict = load(open('./ms-data/class_dict_10.json', 'r'))
max_number = 10
i = 0

for path in files:
    if i == max_number:
        exit()
        
# path = './ms-data/class_10_train/n01882714/images/AnnualCrop_1.tif'
    print(path)

    # Read image and convert the image to [0, 1] range 3d tensor
    img = tifffile.imread(path)
    print(img.shape)
    print(img.dtype)
    img = img[:, :, [3, 2, 1]] # change order to have RGB
    
    # linear normalization
    # img = normalize(img) # -> 48%


    img = img/10000*3.5

    # img = scale_band(img, input_min, input_max, output_min, output_max)

    # img = scale_band(img)
    
    # Contrast stretching
#     p2, p98 = np.percentile(img, (70, 30))
#     img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # img =  exposure.equalize_hist(img) -> 30%
    # img = img.astype('uint8')
    print(img.dtype)

    fig = plt.figure()

    # fig.add_subplot(1,3,3)
    plt.imshow(img)
    plt.show()

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [WIDTH, HEIGHT])

    i = i + 1
exit()












img = tifffile.imread(test_image) # use the key=0
print(img.shape)
print(img.dtype)
# img = img[:,:,1:4] # get B02, B03, B04 - BGR
img = img[:, :, [3, 2, 1]] # change order to have RGB
img = img.astype('uint8')

# attempt with rasterio -- IN PROGRESS
# I can't install it!
#import rasterio
# img = rasterio.open(test_image)

fig = plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(img)

img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [WIDTH, HEIGHT])

img = tf.expand_dims(img, 0)  # Create batch axis

img_predictions = tiny_vgg.predict(img)
class_names = ["highway", "forest", "river", "permanent crop", "industrial", "annual crop", "sea or lake", "herbaceous", "residential", "pasture"]
pred_label = class_names[np.argmax(np.round(img_predictions,2))]
print(" Predicted label is :: "+ pred_label)

print(img.shape)
print(img.dtype)


img = tf.io.read_file('./data/class_10_val/map_images/Industrial_473.JPEG')
# print(img)
img = tf.image.decode_jpeg(img, channels=3)

fig.add_subplot(1,3,2)
plt.imshow(img)
plt.show()


img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [WIDTH, HEIGHT])


img = tf.expand_dims(img, 0)  # Create batch axis

# img_predictions = tiny_vgg.predict(img)
print(img.shape)
print(img.dtype)


img_predictions = tiny_vgg.predict(img)
class_names = ["highway", "forest", "river", "permanent crop", "industrial", "annual crop", "sea or lake", "herbaceous", "residential", "pasture"]
pred_label = class_names[np.argmax(np.round(img_predictions,2))]
print(" Predicted label is :: "+ pred_label)


