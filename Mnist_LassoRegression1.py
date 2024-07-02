# Introduction to Artificial Intelligence
# MNIST Dataset
# Lasso Linear Regression
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.linear_model

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

# One-hot encode the labels
encoder = sklearn.preprocessing.OneHotEncoder(categories='auto', sparse_output=False)
train_labels_onehot = encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels_onehot = encoder.transform(test_labels.reshape(-1, 1))
num_classes = len(encoder.categories_[0])

#
# Train classifier
#

# Train a linear regression classifier
alph = 0.0005
model = sklearn.linear_model.Lasso(alpha=alph)
model.fit(train_data, train_labels_onehot)

# Predict the probabilities of each class
pred_proba = model.predict(test_data)

# Pick the maximum
pred = np.argmax(pred_proba, axis=1).astype("uint8")

#
# Explore coefficients
#

print("Min coef:", np.min(model.coef_))
print("Max coef:", np.max(model.coef_))
print("Coef mean:", np.mean(model.coef_))
print("Coef stddev: ", np.std(model.coef_))

# Plot a histogram of coefficient values
#"""
hist, bins = np.histogram(model.coef_, 500)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Coefficient values")
plt.show()
#"""

# Display the coefficients as an image
for n in range(num_classes):
    coef_img = model.coef_[n].reshape(28, 28)
    plt.figure()
    plt.imshow(coef_img, cmap="viridis", norm=plt.Normalize(-0.01, 0.01, clip=True))
    plt.title("Coefficients for class "+str(n))
plt.show()


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

# Per-Class Precision & Recall
precision = sklearn.metrics.precision_score(test_labels, pred, average=None)
recall = sklearn.metrics.recall_score(test_labels, pred, average=None)
for n in range(num_classes):
    print("  Class {}: Precision: {:.5f} Recall: {:.5f}".format(n, precision[n], recall[n]))

# Compute the prediction accuracy against the training data
pred_proba_training = model.predict(train_data)
print("Against training set:")
pred_training = np.argmax(pred_proba_training, axis=-1).astype("uint8")
print("  Accuracy:   {:.5f}".format(sklearn.metrics.accuracy_score(train_labels, pred_training)))
print("  Precision:  {:.5f}".format(sklearn.metrics.precision_score(train_labels, pred_training, average='weighted')))
print("  Recall:     {:.5f}".format(sklearn.metrics.recall_score(train_labels, pred_training, average='weighted')))


