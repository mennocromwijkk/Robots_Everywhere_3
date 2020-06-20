# -*- coding: utf-8 -*-

# Code base credit:
# https://towardsdatascience.com/machinex-image-data-augmentation-using-keras-b459ef87cd22

#%% Setup

# Import keras image processing
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import os

# Define range within which the image can be edited
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.2,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

#%% Data Augmentation

image_folder = 'Compressed photos'
for file in os.listdir(image_folder): # performs the actions for each file in the folder

    # Load images
    full_dir = image_folder + '\\' + file
    img = load_img(full_dir)

    x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, 150, 150)

    # Creates 'images_amount'x different images and saves them in the specified place
    images_amount = 1
    i = 0
    for batch in datagen.flow(x,save_to_dir='augmented', save_prefix='img', save_format='jpg'):
        i += 1
        if i >= images_amount:
            break

