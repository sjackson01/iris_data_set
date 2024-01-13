from sklearn.datasets import load_iris

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

