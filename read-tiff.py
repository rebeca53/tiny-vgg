import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tiny_vgg = tf.keras.models.load_model('trained_vgg_best.h5')
WIDTH = 64
HEIGHT = 64
test_image = './Forest_1465.tif'

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
img = tifffile.imread(test_image, key=0) # use the key=0
print(img.shape)
print(img.dtype)
# img = img[:,:,1:4] # get B02, B03, B04 - BGR
img = img[:, :, [3, 2, 1]] # change order to have RGB

# attempt with rasterio -- IN PROGRESS
# I can't install it!
#import rasterio
# img = rasterio.open(test_image)

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img.astype('uint8'))

img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [WIDTH, HEIGHT])

img = tf.expand_dims(img, 0)  # Create batch axis

img_predictions = tiny_vgg.predict(img)
class_names = ["highway", "forest", "river", "permanent crop", "industrial", "annual crop", "sea or lake", "herbaceous", "residential", "pasture"]
pred_label = class_names[np.argmax(np.round(img_predictions,2))]
print(" Predicted label is :: "+ pred_label)

print(img.shape)
print(img.dtype)


img = tf.io.read_file('./data/class_10_val/map_images/Forest_1465.JPEG')
# print(img)
img = tf.image.decode_jpeg(img, channels=3)

fig.add_subplot(1,2,2)
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






