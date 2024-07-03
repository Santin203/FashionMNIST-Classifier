import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble
from sklearn.neighbors import KNeighborsRegressor

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

#Train Knn Classifier
neighbors = 300
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=300)
knn.fit(train_data, train_labels)

# Predict the probabilities of each class
pred_proba = knn.predict(test_data)

# Pick the maximum
pred = np.argmax(pred_proba, axis=1).astype("uint8")

#
# Explore coefficients
#

print("Min coef:", np.min(knn.coef_))
print("Max coef:", np.max(knn.coef_))
print("Coef mean:", np.mean(knn.coef_))
print("Coef stddev: ", np.std(knn.coef_))

# Plot a histogram of coefficient values
hist, bins = np.histogram(knn.coef_, 500)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Coefficient values")
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
pred_proba_training = knn.predict(train_data)
print("Against training set:")
pred_training = np.argmax(pred_proba_training, axis=-1).astype("uint8")
print("  Accuracy:   {:.5f}".format(sklearn.metrics.accuracy_score(train_labels, pred_training)))
print("  Precision:  {:.5f}".format(sklearn.metrics.precision_score(train_labels, pred_training, average='weighted')))
print("  Recall:     {:.5f}".format(sklearn.metrics.recall_score(train_labels, pred_training, average='weighted')))

print(f"\nResults obtained using alpha = {alph}")