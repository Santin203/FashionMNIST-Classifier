# Introduction to Artificial Intelligence
# MNIST Dataset
# Random Forest Classifier
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble

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

# Train a Random Forest classifier
n_estimators = 100
msl = 1e-4
n_repetitions = 10

model = sklearn.ensemble.RandomForestClassifier(\
    n_estimators = n_estimators,
    min_samples_leaf = msl,
    n_jobs=-1)

results = []

for rep in range(n_repetitions):
    
    model.fit(train_data, train_labels)
    pred = model.predict(test_data)
    accuracy = sklearn.metrics.accuracy_score(test_labels, pred)
    print("  Repetition {}: Test accuracy: {:.4f}".format(rep, accuracy))
    results.append(accuracy)

results_np = np.array(results)
print("With n_estimators={}, min_samples_leaf={}:".format(n_estimators, msl))
print("Min:  {:.4f}".format(results_np.min()))
print("Max:  {:.4f}".format(results_np.max()))
print("Mean: {:.4f}".format(results_np.mean()))
print("Std:  {:.4f}".format(results_np.std()))

