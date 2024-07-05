# Introduction to Artificial Intelligence
# MNIST Dataset
# Heterogeneous Ensemble Classifier
# By Jose Campos and William He Yu
# Based on codes made by Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.tree

# Load the training and test data from the Pickle file
# Availabe sizes: 24, 20, 16, 12, 10, 8, 6, 4

size = 24
filename = "mnist_dataset_{}.pickle".format(size)

# Use this filename if you want to use the original dataset
#filename = "mnist_dataset.pickle"

with open(filename, "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

#
# Train classifiers
#

#Note: The paramters used in this file for each model yield the best results

# Train a Random Forest classifier
n_estimators = 100
msl_rf = 1
max_fs = 200
n_repetitions = 10

rf = sklearn.ensemble.RandomForestClassifier(\
    n_estimators = n_estimators,
    min_samples_leaf = msl_rf, max_features = max_fs, n_jobs=-1)

# Train a Softmax Regression classifier
# Use stochastic approach to save time

solv_algo = 'saga'
tol = 1e-2
max_iter = 50
sm = sklearn.linear_model.LogisticRegression(\
    solver=solv_algo, tol=tol, max_iter = max_iter, n_jobs=-1) 

#Train Knn Classifier
neighbors = 4
weight = 'distance'
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)

er = sklearn.ensemble.VotingClassifier(
    estimators=[('SoftMax', sm),
                ('RandomForest', rf),
                ('Knn', knn)
                ],voting='soft', n_jobs=-1)

#Declare array for results
results = []

#Repeat test to compute average
for rep in range(n_repetitions):
    
    er.fit(train_data, train_labels)
    pred = er.predict(test_data)
    accuracy = sklearn.metrics.accuracy_score(test_labels, pred)
    print("  Repetition {}: Test accuracy: {:.5f}".format(rep, accuracy))
    results.append(accuracy)

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, pred)
print("Confusion Matrix:")
print(cmatrix)

#Print Results
results_np = np.array(results)
print("With a random forest trained with n_estimators={}, min_samples_leaf={}".format(n_estimators, msl_rf))
print("With a Knn trained with n_neighbors={}, wheights={}".format(neighbors, weight))
print("With a softmax regression trained with a solver algorithm of={}, tol={}, max_iter={}".format(solv_algo,tol,max_iter))
print("Model Results:\n")
print("Min Accuracy:  {:.5f}".format(results_np.min()))
print("Max Accuracy:  {:.5f}".format(results_np.max()))
print("Mean Accuracy: {:.5f}".format(results_np.mean()))
print("Std:  {:.5f}".format(results_np.std()))


