import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.tree

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

# Train a Random Forest classifier
n_estimators = 100
msl_rf = 1e-6
n_repetitions = 10

rf = sklearn.ensemble.RandomForestClassifier(\
    n_estimators = n_estimators,
    min_samples_leaf = msl_rf,
    n_jobs=-1)

# Train a Decision Tree classifier
msl_dt = 1e-4
dt = sklearn.tree.DecisionTreeClassifier(min_samples_leaf = msl_dt)
print("With min_samples_leaf={}".format(msl_dt))

er = sklearn.ensemble.VotingClassifier(
    estimators=[('DecisionTree', dt),
                ('RandomForest', rf)
                ],voting='soft')

results = []

for rep in range(n_repetitions):
    
    er.fit(train_data, train_labels)
    pred = er.predict(test_data)
    accuracy = sklearn.metrics.accuracy_score(test_labels, pred)
    print("  Repetition {}: Test accuracy: {:.5f}".format(rep, accuracy))
    results.append(accuracy)

results_np = np.array(results)
print("With a random forest trained with n_estimators={}, min_samples_leaf={}".format(n_estimators, msl_rf))
print("With a decision tree trained with min_samples_leaf={}".format(msl_dt))
print("Model Results:\n")
print("Min:  {:.5f}".format(results_np.min()))
print("Max:  {:.5f}".format(results_np.max()))
print("Mean: {:.5f}".format(results_np.mean()))
print("Std:  {:.5f}".format(results_np.std()))


