import math
import pandas as pd
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import timeit
start = timeit.default_timer()
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

from sklearn.model_selection import train_test_split, GridSearchCV

# Create data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

# StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500, verbose=True, early_stopping=True)
mlp.fit(X_train,y_train)

# MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=500, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#        verbose=False, warm_start=False)

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

stop = timeit.default_timer()

print('Time: ', stop - start)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(predictions, y_test))
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
