# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:27:54 2018

@author: Yang Liu
"""

import cv2, os
import numpy as np
import matplotlib.image as mpimg
import random
import sklearn
import pandas as pd

def get_training_set(csv_files, image_path):

# get Training set in form of a list    
    
    #%% create a list of taining data
    # make train, validation and test dataset
    correction = 0.2 # this is a parameter to tune
    x_filenames = []
    y_train     = []
    
    driving_log     = pd.read_csv(csv_files)
    x_center_filenames  = driving_log.center  # IMG/filename
    x_left_filenames    = driving_log.left
    x_right_filenames   = driving_log.right
    
    y_center_steering   = driving_log.steering
    y_left_steering     = y_center_steering + correction
    y_right_steering    = y_center_steering - correction
    
    y_train.extend(y_center_steering)
    y_train.extend(y_left_steering)
    y_train.extend(y_right_steering)
    
    x_filenames.extend(x_center_filenames)
    x_filenames.extend(x_left_filenames)
    x_filenames.extend(x_right_filenames)
    
    #%% data augumentation
    
    for i in range(len(x_filenames)):
        x_filenames[i] = x_filenames[i].split('/')[-1]
    
    ag_filenames = []
    for name in x_filenames: 
        ag_filenames.append(make_augumented_img(image_path,name))
    
    x_filenames.extend(ag_filenames)
    y_tmp = [y * -1 for y in y_train]
    y_train.extend(y_tmp)
    
    train_sets = list(zip(x_filenames, y_train))
    
    return train_sets
    
    
def load_image(path, image_file):
# Load RGB images from a file

    return mpimg.imread(os.path.join(path, image_file.strip()))


def crop(img):

# remove the sky at the top and the car front 

    return img[50:-20, :, :] # remove the sky and the car front


def resize(img):

# Resize the image

    return cv2.resize(img, (200, 66), cv2.INTER_AREA)


def rgb2yuv(img):
    
# Convert the image from RGB to YUV "
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def gaussian_blur(img, kernel_size):

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def preprocess(img):  
    
# Combine all the preprocess steps 
    
    img = crop(img)
    img = resize(img)
    img = rgb2yuv(img)
 #   img = np.array(img)
 #   img = gaussian_blur(img,7)
    return img


def flip_image(image):
    
# Flip Image
    
    image_flipped = np.fliplr(image)
    return image_flipped


def make_augumented_img(src_path, src_file):
    
# Make augumented image
    
    image = load_image(src_path, src_file)    
    image = flip_image(image)

# rename the image    
    image_name = 'flip_' + src_file
    full_im_name = os.path.join(src_path, image_name.strip())
    
# store the image if it is not created before
    if not os.path.isfile(full_im_name):
        mpimg.imsave(full_im_name,image)
    
    return image_name

def generator(path, samples, batch_size=128):
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name, angle = batch_sample
                image = preprocess(load_image(path, image_name))
                images.append(image)
                angles.append(angle)
# trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)