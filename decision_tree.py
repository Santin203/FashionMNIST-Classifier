# Introduction to Artificial Intelligence
# MNIST Dataset
# Decision Tree Classifier
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# modified by William He Yu for project 1

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.tree

#
# Load and prepare input data
#

# Load the training and test data from the Pickle file
with open("mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

#
# Train classifier
#

# Train a Decision Tree classifier
# modified parameter msl for testing, 1e-4 is the best
msl = 1e-4
model = sklearn.tree.DecisionTreeClassifier(min_samples_leaf = msl)
print("With min_samples_leaf={}".format(msl))

model.fit(train_data, train_labels)

# Make the class predictions
pred = model.predict(test_data)

#
# Metrics
#

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, pred)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, pred, average='weighted')))

# Per-Class Precision & Recall
precision = sklearn.metrics.precision_score(test_labels, pred, average=None)
recall = sklearn.metrics.recall_score(test_labels, pred, average=None)
num_classes = len(np.unique(train_labels))
for n in range(num_classes):
    print("  Class {}: Precision: {:.3f} Recall: {:.3f}".format(n, precision[n], recall[n]))

# Compute the prediction accuracy against the training data
print("Against training set:")
pred_training = model.predict(train_data)
print("  Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(train_labels, pred_training)))
print("  Precision:  {:.3f}".format(sklearn.metrics.precision_score(train_labels, pred_training, average='weighted')))
print("  Recall:     {:.3f}".format(sklearn.metrics.recall_score(train_labels, pred_training, average='weighted')))


