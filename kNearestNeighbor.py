import math

import pandas as pd
import numpy as np

import matplotlib.pylab as plt

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import timeit
start = timeit.default_timer()

#Turn labels into numeric values
from sklearn import preprocessing

# START AMAZON KICK STARTER DATA PREPROCESSING
# dataset = pd.read_csv(dir_path+"/Datasets/kickstarter-projects.csv", sep=";")
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

# # # START GOOGLE APP STORE DATA PREPROCESSING
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
## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## Import the Classifier.
from sklearn.neighbors import KNeighborsClassifier
## Instantiate the model with 5 neighbors.
knn = KNeighborsClassifier(n_neighbors=5)#, algorithm='brute', weights="distance")
## Fit the model on the training data.
knn.fit(X_train, y_train)
## See how the model performs on the test data.
knn.score(X_test, y_test)

print(knn.score(X_test, y_test))

stop = timeit.default_timer()

print('Time: ', stop - start)

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
  # Setup a k-NN Classifier with k neighbors: knn
  knn = KNeighborsClassifier(n_neighbors=k,
                             # algorithm='brute',
                             # weights="distance"
                             )

  # Fit the classifier to the training data
  knn.fit(X_train, y_train)

  #Compute accuracy on the training set
  train_accuracy[i] = knn.score(X_train, y_train)

  #Compute accuracy on the testing set
  test_accuracy[i] = knn.score(X_test, y_test)

# # Generate plot Amazon
# plt.title('k-NN: Varying Number of Neighbors (Kickstarter)')
# plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.show()

# # Generate plot Google
# plt.title('k-NN: Varying Number of Neighbors (Playstore)')
# plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.show()
y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))