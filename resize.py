# Introduction to Artificial Intelligence
# MNIST Dataset
# Program to resize images to a target size
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# Modified by William He Yu for project 1

import pickle
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform

# Load the training and test data from the Pickle file
with open("mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

train_len = train_data.shape[0]
test_len = test_data.shape[0]

original_size = 28

# Resize the images to the following sizes and create the pickle file for each.
size_list = [24, 20, 16, 12, 10, 8, 6, 4]

for target_size in size_list:

    target_n_pixels = target_size*target_size

    # Resize all the training images
    train_data_resized = np.zeros( (train_len, target_n_pixels) )
    for img_idx in range(train_len):

        # Get the image
        img = train_data[img_idx].reshape(original_size,original_size)

        # Resize the image
        img_resized = skimage.transform.resize(\
            img, (target_size,target_size), anti_aliasing=True)

        # Put it back in vector form
        train_data_resized[img_idx] = img_resized.reshape(1, target_n_pixels)

    # Resize all the test images
    test_data_resized = np.zeros( (test_len, target_n_pixels) )
    for img_idx in range(test_len):

        # Get the image
        img = test_data[img_idx].reshape(original_size,original_size)

        # Resize the image
        img_resized = skimage.transform.resize(\
            img, (target_size,target_size), anti_aliasing=True)

        # Put it back in vector form
        test_data_resized[img_idx] = img_resized.reshape(1, target_n_pixels)

    # Archive the rescaled data set
    filename = "mnist_dataset_{}.pickle".format(target_size)
    with open(filename, 'wb') as f:
        pickle.dump([train_data_resized, train_labels, \
                    test_data_resized, test_labels], \
                    f, pickle.HIGHEST_PROTOCOL)
