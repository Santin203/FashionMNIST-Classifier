# Introduction to Artificial Intelligence
# MNIST Dataset
# K-Nearest Neighbors
# By Jose Campos
# Based on codes made by Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# Modified by: Jose Campos

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble

# Load the training and test data from the Pickle file
with open("mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

#
# Train classifiers
#

#Note: The paramters used in this file the model yield the best results

#Train Knn Classifier
neighbors = 4
weight = 'distance'
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbors, weights = weight)
knn.fit(train_data, train_labels)

# Predict the probabilities of each class
pred = knn.predict(test_data)

#
# Metrics
#

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, pred)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.5f}".format(sklearn.metrics.accuracy_score(test_labels, pred)))
print("Precision:  {:.5f}".format(sklearn.metrics.precision_score(test_labels, pred, average='weighted')))
print("Recall:     {:.5f}".format(sklearn.metrics.recall_score(test_labels, pred, average='weighted')))

train_pred = knn.predict(train_data)

print("Against Training Data")
# Accuracy, precision & recall
print("Accuracy:   {:.5f}".format(sklearn.metrics.accuracy_score(train_labels, train_pred)))
print("Precision:  {:.5f}".format(sklearn.metrics.precision_score(train_labels, train_pred, average='weighted')))
print("Recall:     {:.5f}".format(sklearn.metrics.recall_score(train_labels, train_pred, average='weighted')))

print(f"\nResults obtained using n_neighbors = {neighbors}, wheights = {weight}")