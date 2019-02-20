import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
# %matplotlib inline
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# import csv
# with open(dir_path+"/Datasets/kickstarter-projects.csv", 'rb') as csvfile:
#         spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#         for row in spamreader:
#             print ', '.join(row)


import timeit
start = timeit.default_timer()


#Turn labels into numeric values
from sklearn import preprocessing

# START AMAZON KICK STARTER DATA PREPROCESSING
dataset = pd.read_csv(dir_path+"/Datasets/kickstarter-projects.csv", sep=";")
print(dataset.shape)
print(dataset.head())

print(dataset.describe())
stateLabels = preprocessing.LabelEncoder()
stateLabels.fit(dataset['state']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(stateLabels.classes_))
# print(stateLabels.transform(list(stateLabels.classes_)))
dataset['state'] = stateLabels.transform(dataset['state'])

categoryLabels = preprocessing.LabelEncoder()
categoryLabels.fit(dataset['category']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(categoryLabels.classes_))
# print(categoryLabels.transform(list(categoryLabels.classes_)))
dataset['category'] = categoryLabels.transform(dataset['category'])

mainCategoryLabels = preprocessing.LabelEncoder()
mainCategoryLabels.fit(dataset['main_category']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(mainCategoryLabels.classes_))
# print(mainCategoryLabels.transform(list(mainCategoryLabels.classes_)))
dataset['main_category'] = mainCategoryLabels.transform(dataset['main_category'])

currencyLabels = preprocessing.LabelEncoder()
currencyLabels.fit(dataset['currency']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(currencyLabels.classes_))
# print(currencyLabels.transform(list(currencyLabels.classes_)))
dataset['currency'] = currencyLabels.transform(dataset['currency'])

countryLabels = preprocessing.LabelEncoder()
countryLabels.fit(dataset['country']) #["paris", "paris", "tokyo", "amsterdam"])
# print(list(countryLabels.classes_))
# print(countryLabels.transform(list(countryLabels.classes_)))
dataset['country'] = countryLabels.transform(dataset['country'])

y = dataset['state']
dataset = dataset.drop('state', axis=1)
dataset = dataset.drop('deadline', axis=1)
dataset = dataset.drop('launched', axis=1)
dataset = dataset.drop('name', axis=1)
dataset = dataset.drop('ID', axis=1)

X = dataset

# # START GOOGLE APP STORE DATA PREPROCESSING
# dataset = pd.read_csv(dir_path+"/Datasets/googleplaystore-cleaned.csv", sep=";")
# # The names of all the columns in the data.
# print(dataset.columns.values)
# # Turn labels into numeric values
#
# categoryLabels = preprocessing.LabelEncoder()
# categoryLabels.fit(dataset['Category']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(categoryLabels.classes_))
# # print(categoryLabels.transform(list(categoryLabels.classes_)))
# dataset['Category'] = categoryLabels.transform(dataset['Category'])
#
# installsLabels = preprocessing.LabelEncoder()
# installsLabels.fit(dataset['Installs']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(mainCategoryLabels.classes_))
# # print(mainCategoryLabels.transform(list(mainCategoryLabels.classes_)))
# dataset['Installs'] = installsLabels.transform(dataset['Installs'])
#
# typeLabels = preprocessing.LabelEncoder()
# typeLabels.fit(dataset['Type']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(currencyLabels.classes_))
# # print(currencyLabels.transform(list(currencyLabels.classes_)))
# dataset['Type'] = typeLabels.transform(dataset['Type'])
#
# contantRatingLabels = preprocessing.LabelEncoder()
# contantRatingLabels.fit(dataset['Content Rating']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(countryLabels.classes_))
# # print(countryLabels.transform(list(countryLabels.classes_)))
# dataset['Content Rating'] = contantRatingLabels.transform(dataset['Content Rating'])
#
# genreLabels = preprocessing.LabelEncoder()
# genreLabels.fit(dataset['Genres']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(countryLabels.classes_))
# # print(countryLabels.transform(list(countryLabels.classes_)))
# dataset['Genres'] = genreLabels.transform(dataset['Genres'])
#
# priceLabels = preprocessing.LabelEncoder()
# priceLabels.fit(dataset['Price']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(countryLabels.classes_))
# # print(countryLabels.transform(list(countryLabels.classes_)))
# dataset['Price'] = priceLabels.transform(dataset['Price'])
#
# sizeLabels = preprocessing.LabelEncoder()
# sizeLabels.fit(dataset['Size']) #["paris", "paris", "tokyo", "amsterdam"])
# # print(list(countryLabels.classes_))
# # print(countryLabels.transform(list(countryLabels.classes_)))
# dataset['Size'] = sizeLabels.transform(dataset['Size'])
#
# dataset['Rating'] = [round(x) for x in dataset['Rating']]
#
# y = dataset['Rating']
# dataset = dataset.drop('Android Ver', axis=1)
# dataset = dataset.drop('Current Ver', axis=1)
# dataset = dataset.drop('Last Updated', axis=1)
# dataset = dataset.drop('App', axis=1)
# dataset = dataset.drop('Rating', axis=1)
#
# X = dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

import graphviz
# dot_data = tree.export_graphviz(classifier, out_file=None)
dot_data = tree.export_graphviz(classifier, out_file=None,
                         # feature_names=X,
                         # class_names=list(stateLabels.classes_),
                         class_names=['1.0', '2.0','3.0', '4.0', '5.0'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Amazon Decision Tree Visual")
# graph.render("Google Decision Tree Visual")


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('[0] canceled | [1] failed | [2] live | [3] successful | [4] suspended')
# print('[1.0] 1 Star | [2.0] 2 Star | [3.0] 3 Star | [4.0] 4 Star | [5.0] 5 Star')
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import accuracy_score
print("Accuracy Score")
print(accuracy_score(y_pred, y_test))

stop = timeit.default_timer()

print('Time: ', stop - start)

# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor()
# regressor.fit(X_train, y_train)
#
# y_pred = regressor.predict(X_test)
#
# # df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
# # print(df)
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print('[0] canceled | [1] failed | [2] live | [3] successful | [4] suspended')

# https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
# http://scikit-learn.org/dev/modules/tree.html
# Decision trees complete, do it for play store ratings now

# Round the play store ratings (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)
# Get Days elapsed on the store
# Get title length

## Amazon Dataset Classes
# ['canceled', 'failed', 'live', 'successful', 'suspended']
# [0 1 2 3 4]

# ['3D Printing', 'Academic', 'Accessories', 'Action', 'Animals', 'Animation', 'Anthologies', 'Apparel', 'Apps', 'Architecture', 'Art', 'Art Books', 'Audio', 'Bacon', 'Blues', 'Calendars', 'Camera Equipment', 'Candles', 'Ceramics', "Children's Books", 'Childrenswear', 'Civic Design', 'Classical Music', 'Comedy', 'Comic Books', 'Comics', 'Community Gardens', 'Conceptual Art', 'Cookbooks', 'Country & Folk', 'Couture', 'Crafts', 'Crochet', 'DIY', 'DIY Electronics', 'Dance', 'Design', 'Digital Art', 'Documentary', 'Drama', 'Drinks', 'Electronic Music', 'Embroidery', 'Events', 'Experimental', 'Fabrication Tools', 'Faith', 'Family', 'Fantasy', "Farmer's Markets", 'Farms', 'Fashion', 'Festivals', 'Fiction', 'Film & Video', 'Fine Art', 'Flight', 'Food', 'Food Trucks', 'Footwear', 'Gadgets', 'Games', 'Gaming Hardware', 'Glass', 'Graphic Design', 'Graphic Novels', 'Hardware', 'Hip-Hop', 'Horror', 'Illustration', 'Immersive', 'Indie Rock', 'Installations', 'Interactive Design', 'Jazz', 'Jewelry', 'Journalism', 'Kids', 'Knitting', 'Latin', 'Letterpress', 'Literary Journals', 'Live Games', 'Makerspaces', 'Metal', 'Mixed Media', 'Mobile Games', 'Movie Theaters', 'Music', 'Music Videos', 'Musical', 'Narrative Film', 'Nature', 'Nonfiction', 'Painting', 'People', 'Performance Art', 'Performances', 'Periodicals', 'Pet Fashion', 'Photo', 'Photobooks', 'Photography', 'Places', 'Playing Cards', 'Plays', 'Poetry', 'Pop', 'Pottery', 'Print', 'Printing', 'Product Design', 'Public Art', 'Publishing', 'Punk', 'Puzzles', 'Quilts', 'R&B', 'Radio & Podcasts', 'Ready-to-wear', 'Restaurants', 'Robots', 'Rock', 'Romance', 'Science Fiction', 'Sculpture', 'Shorts', 'Small Batch', 'Software', 'Sound', 'Space Exploration', 'Spaces', 'Stationery', 'Tabletop Games', 'Taxidermy', 'Technology', 'Television', 'Textiles', 'Theater', 'Thrillers', 'Translations', 'Typography', 'Vegan', 'Video', 'Video Art', 'Video Games', 'Wearables', 'Weaving', 'Web', 'Webcomics', 'Webseries', 'Woodworking', 'Workshops', 'World Music', 'Young Adult', 'Zines']
# [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
#   18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
#   36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
#   54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
#   72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
#   90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
#  108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
#  126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
#  144 145 146 147 148 149 150 151 152 153 154 155]

# ['Art', 'Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film & Video', 'Food', 'Games', 'Journalism', 'Music', 'Photography', 'Publishing', 'Technology', 'Theater']
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

# ['AUD', 'CAD', 'CHF', 'DKK', 'EUR', 'GBP', 'HKD', 'MXN', 'NOK', 'NZD', 'SEK', 'SGD', 'USD']
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12]

# ['AT', 'AU', 'BE', 'CA', 'CH', 'DE', 'DK', 'ES', 'FR', 'GB', 'HK', 'IE', 'IT', 'LU', 'MX', 'NL', 'NO', 'NZ', 'SE', 'SG', 'US']
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
