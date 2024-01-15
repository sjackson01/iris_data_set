import sys
import mglearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier
  
iris_dataset = load_iris()

print("\n")

print("Keysof iris_dataset: \n{}".format(iris_dataset.keys()))

print("\n")

#Parital Description
print(iris_dataset['DESCR'][:193] + "\n...")

#Full Description 
#print("Data Set Description: {}".format(iris_dataset['DESCR']))

print("Target names: {}".format(iris_dataset['target_names']))

print("\n")

print("Feature names: {}".format(iris_dataset['feature_names']))

print("\n")

print("Type of data: {}".format(type(iris_dataset['data'])))

print("\n")

print("Shape of data: {}".format(iris_dataset['data'].shape))

print("\n")

print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))

print("\n")

print("Type of target: {}".format(type(iris_dataset['target'])))

print("\n")

print("Shape of target: {}".format(iris_dataset['target'].shape))

print("\n")

print("Target:\n{}".format(iris_dataset['target']))

print("\n")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

print("\n")

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

#define dataset 
#iris = sns.load_dataset("iris")

#create pair plot for all numeric variables
#sns.pairplot(iris)
#plt.show()

#K-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

#Test Measurments 
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

print("\n")

#Predicts Test Measurment Predicts Class 0 or Setosa 
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))

print("\n")

print("Predicted target name: {}".format(

iris_dataset['target_names'][prediction]))

#Test Set Predicitons 
y_pred = knn.predict(X_test)

print("\n")

print("Test set predictions:\n {}".format(y_pred))

#Get the mean
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#Get the score
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

print("\n")

#Summary : Test accuracy K nearest neighbor iris data set 
X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0) 
knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train, y_train) 
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))




