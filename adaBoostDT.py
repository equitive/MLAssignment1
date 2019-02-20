import math

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import timeit
start = timeit.default_timer()

import os
dir_path = os.path.dirname(os.path.realpath(__file__))




#Turn labels into numeric values
from sklearn import preprocessing

# START AMAZON KICK STARTER DATA PREPROCESSING
# dataset = pd.read_csv(dir_path+"/Datasets/kickstarter-projects.csv", sep=";")
#
# # The names of all the columns in the data.
# print(dataset.columns.values)
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

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Create data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Instantiate
# abc = AdaBoostClassifier()

abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,
# algorithm="SAMME"
                         # random_state=0,
                         )

# abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
#     n_estimators=50,
#     learning_rate=1,
#     algorithm="SAMME") # discrete

# abc = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=2),
#     n_estimators=600,
#     learning_rate=1) # real

# Fit
abc.fit(X_train, y_train)

# Predict
y_pred = abc.predict(X_test)

# Accuracy
accuracy_score(y_pred, y_test)

print(accuracy_score(y_pred, y_test))
stop = timeit.default_timer()

print('Time: ', stop - start)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# https://www.ritchieng.com/machine-learning-ensemble-of-learners-adaboost/
# https://chrisalbon.com/machine_learning/trees_and_forests/adaboost_classifier/
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py