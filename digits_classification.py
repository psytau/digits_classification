import tensorflow as tf
import os
mnist_data = tf.keras.datasets.mnist.load_data

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

n_samples = 500 #len(x_train)
print("samples size", n_samples)
x_training_data = x_train[:n_samples].reshape((n_samples, -1))
y_training_data = y_train[:n_samples]

clf = svm.SVC(gamma=0.001, cache_size=3000, verbose=True)
clf.fit(x_training_data, y_training_data)
if (not os.path.exists('./data')):
  os.makedirs('./data')
dump(clf, './data/digitsClassification.joblib')
