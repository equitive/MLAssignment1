import math

import pandas as pd
import numpy as np

import timeit
start = timeit.default_timer()

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

from sklearn import preprocessing



# START AMAZON KICK STARTER DATA PREPROCESSING
# dataset = pd.read_csv(dir_path+"/Datasets/kickstarter-projects.csv", sep=";")
# # The names of all the columns in the data.
# print(dataset.columns.values)
# # Turn labels into numeric values
# stateLabels = preprocessing.LabelEncoder()
# stateLabels.fit(dataset['state']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(stateLabels.classes_))
# # print(stateLabels.transform(list(stateLabels.classes_)))
# dataset['state'] = stateLabels.transform(dataset['state'])
#
# categoryLabels = preprocessing.LabelEncoder()
# categoryLabels.fit(dataset['category']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(categoryLabels.classes_))
# # print(categoryLabels.transform(list(categoryLabels.classes_)))
# dataset['category'] = categoryLabels.transform(dataset['category'])
#
# mainCategoryLabels = preprocessing.LabelEncoder()
# mainCategoryLabels.fit(dataset['main_category']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(mainCategoryLabels.classes_))
# # print(mainCategoryLabels.transform(list(mainCategoryLabels.classes_)))
# dataset['main_category'] = mainCategoryLabels.transform(dataset['main_category'])
#
# currencyLabels = preprocessing.LabelEncoder()
# currencyLabels.fit(dataset['currency']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(currencyLabels.classes_))
# # print(currencyLabels.transform(list(currencyLabels.classes_)))
# dataset['currency'] = currencyLabels.transform(dataset['currency'])
#
# countryLabels = preprocessing.LabelEncoder()
# countryLabels.fit(dataset['country']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(countryLabels.classes_))
# # print(countryLabels.transform(list(countryLabels.classes_)))
# dataset['country'] = countryLabels.transform(dataset['country'])
#
# y = dataset['state']
# dataset = dataset.drop('state', axis=1)
# dataset = dataset.drop('deadline', axis=1)
# dataset = dataset.drop('launched', axis=1)
# dataset = dataset.drop('name', axis=1)
# dataset = dataset.drop('ID', axis=1)
#
# X = dataset


# # START GOOGLE APP STORE DATA PREPROCESSING
dataset = pd.read_csv(dir_path+"/Datasets/googleplaystore-cleaned.csv", sep=";")
# The names of all the columns in the data.
print(dataset.columns.values)
# Turn labels into numeric values

categoryLabels = preprocessing.LabelEncoder()
categoryLabels.fit(dataset['Category']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(categoryLabels.classes_))
# print(categoryLabels.transform(list(categoryLabels.classes_)))
dataset['Category'] = categoryLabels.transform(dataset['Category'])

installsLabels = preprocessing.LabelEncoder()
installsLabels.fit(dataset['Installs']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(mainCategoryLabels.classes_))
# print(mainCategoryLabels.transform(list(mainCategoryLabels.classes_)))
dataset['Installs'] = installsLabels.transform(dataset['Installs'])

typeLabels = preprocessing.LabelEncoder()
typeLabels.fit(dataset['Type']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(currencyLabels.classes_))
# print(currencyLabels.transform(list(currencyLabels.classes_)))
dataset['Type'] = typeLabels.transform(dataset['Type'])

contantRatingLabels = preprocessing.LabelEncoder()
contantRatingLabels.fit(dataset['Content Rating']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(countryLabels.classes_))
# print(countryLabels.transform(list(countryLabels.classes_)))
dataset['Content Rating'] = contantRatingLabels.transform(dataset['Content Rating'])

genreLabels = preprocessing.LabelEncoder()
genreLabels.fit(dataset['Genres']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(countryLabels.classes_))
# print(countryLabels.transform(list(countryLabels.classes_)))
dataset['Genres'] = genreLabels.transform(dataset['Genres'])

priceLabels = preprocessing.LabelEncoder()
priceLabels.fit(dataset['Price']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(countryLabels.classes_))
# print(countryLabels.transform(list(countryLabels.classes_)))
dataset['Price'] = priceLabels.transform(dataset['Price'])

sizeLabels = preprocessing.LabelEncoder()
sizeLabels.fit(dataset['Size']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(countryLabels.classes_))
# print(countryLabels.transform(list(countryLabels.classes_)))
dataset['Size'] = sizeLabels.transform(dataset['Size'])

dataset['Rating'] = [round(x) for x in dataset['Rating']]

y = dataset['Rating']
dataset = dataset.drop('Android Ver', axis=1)
dataset = dataset.drop('Current Ver', axis=1)
dataset = dataset.drop('Last Updated', axis=1)
dataset = dataset.drop('App', axis=1)
dataset = dataset.drop('Rating', axis=1)

X = dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

# StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train)
#
# y_pred = svclassifier.predict(X_test)
#
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


import sys
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

# Plot traning and test data
# plot_data(X_train, y_train, X_test, y_test)

# Create a linear SVM classifier
clf = svm.SVC(kernel='linear')

# Train classifier
clf.fit(X_train, y_train)

# Plot decision function on training and test data
# plot_decision_function(X_train, y_train, X_test, y_test, clf)

# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
print("Linear Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,clf_predictions))
print(classification_report(y_test,clf_predictions))
stop = timeit.default_timer()

print('Time: ', stop - start)
start = timeit.default_timer()


# Create SVM classifier based on RBF kernel.
rbf_clf = svm.SVC(kernel='rbf', C = 10.0, gamma=0.1)

# Train classifier
rbf_clf.fit(X_train, y_train)

# Plot decision function on training and test data
# plot_decision_function(X_train, y_train, X_test, y_test, clf)

# Make predictions on unseen test data
rbf_clf_predictions = rbf_clf.predict(X_test)
print("RBF Accuracy: {}%".format(rbf_clf.score(X_test, y_test) * 100 ))

print(confusion_matrix(y_test,rbf_clf_predictions))
print(classification_report(y_test,rbf_clf_predictions))
stop = timeit.default_timer()

print('Time: ', stop - start)


# Create SVM classifier based on poly kernel.
# poly_clf = svm.SVC(kernel='poly', gamma=2)

# Train classifier
# poly_clf.fit(X_train, y_train)

# Plot decision function on training and test data
# plot_decision_function(X_train, y_train, X_test, y_test, clf)

# Make predictions on unseen test data
# poly_clf_predictions = poly_clf.predict(X_test)
# print("POLY Accuracy: {}%".format(poly_clf.score(X_test, y_test) * 100 ))
# print(confusion_matrix(y_test,poly_clf_predictions))
# print(classification_report(y_test,poly_clf_predictions))

# https://www.learnopencv.com/svm-using-scikit-learn-in-python/
# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
# http://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html