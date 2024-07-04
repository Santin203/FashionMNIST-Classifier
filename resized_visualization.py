# Introduction to Artificial Intelligence
# Visualization of resized images
# By William He Yu for project 1


import pickle
import numpy as np
import matplotlib.pyplot as plt

# determine the size list and label list
size_list = [24, 20, 16, 12, 10, 8, 6, 4]
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# load and visualize the first image of the original dataset
with open("mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

image = train_data[0].reshape(28,28)
plt.figure()
plt.imshow(image, cmap="gray_r")
plt.title("28 x 28, Label: "+str(labels[train_labels[0]]))
plt.show()

#load and visualize the first image of all resized datasets
for size in size_list:
    filename = "mnist_dataset_{}.pickle".format(size)
    with open(filename, "rb") as f:
        train_data, train_labels, test_data, test_labels = pickle.load(f)
    
    image = train_data[0].reshape(size,size)
    plt.figure()
    plt.imshow(image, cmap="gray_r")
    plt.title("{} x {}, Label: {}".format(size, size, labels[train_labels[0]]))
    plt.show()