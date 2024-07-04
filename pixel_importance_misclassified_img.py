# Introduction to Artificial Intelligence
# MNIST Dataset
# Random Forest Classifier, analysis of pixel importance
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import sklearn.ensemble

#
# Load and prepare input data
#

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Load the training and test data from the Pickle file
with open("mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

num_classes = len(np.unique(train_labels))

# Change the integer labels to string labels
train_labels = [labels[i] for i in train_labels]
test_labels = [labels[i] for i in test_labels]


#
# Train classifier
#

# Train a Random Forest classifier
n_estimators = 100
msl_rf = 1
max_fs = 200
n_repetitions = 10

rf = sklearn.ensemble.RandomForestClassifier(\
    n_estimators = n_estimators,
    min_samples_leaf = msl_rf, max_features = max_fs,
    n_jobs=-1)

# Train a Softmax Regression classifier
# Use stochastic approach to save time

solv_algo = 'saga'
tol = 1e-2
max_iter = 50
sm = sklearn.linear_model.LogisticRegression(\
    solver=solv_algo, tol=tol, max_iter = max_iter) 

#Train Knn Classifier
neighbors = 4
weight = 'distance'
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbors,n_jobs=-1)

model = sklearn.ensemble.VotingClassifier(n_jobs=-1,
    estimators=[('SoftMax', sm),
                ('RandomForest', rf),
                ('Knn', knn)
                ],voting='soft')

model.fit(train_data, train_labels)

# Make the class predictions
pred = model.predict(test_data)


#
# Metrics
#


# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, pred, average='weighted')))

# Plot some of the incorrect predictions
#"""
num_displayed = 0
x = 0
while (num_displayed < 10):
    x += 1

    print("Predicted: ", pred[x], " Correct: ", test_labels[x])

    print("X: ", x, " Num_displayed: ", num_displayed)

    # Skip correctly predicted 
    if (pred[x] == test_labels[x]):
        continue

    num_displayed += 1

    # Display the images
    image = test_data[x].reshape(28,28)
    plt.figure()
    plt.imshow(image, cmap="gray_r")
    plt.title("Predicted: "+str(pred[x])+" Correct: "+str(test_labels[x]))
    plt.show()
#"""

#
# Analyze feature importances
#

# Display the feature importances as an image for the Random Forest classifier
rf.fit(train_data, train_labels)
coef_img = rf.feature_importances_.reshape(28, 28)
plt.figure()
plt.imshow(coef_img, cmap="gray_r")
plt.show()
