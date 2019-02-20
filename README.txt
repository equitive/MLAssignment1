ASSIGNMENT ONE READ ME

Student: Ruth 

Presets:

Ensure that the current folder contain this structure:

Amazon Decision Tree Visual	adaBoostDT.py			svm.py
Amazon Decision Tree Visual.pdf	decisionTrees.py		venv
Amazon Decision Tree.pdf	kNearestNeighbor.py
Datasets			neuralNet.py

The Datasets folder should contain this structure:

googleplaystore-cleaned.csv	kickstarter-projects-clean.csv
kickstarter-projects-1.csv	kickstarter-projects.csv

Running the Code:

1. To Run the Decision Tree Code:
	python decisionTrees.py
2. To Run the AdaBoost Code:
	python adaBoostDT.py
3. To Run the SVM Code:
	python svm.py
4. To Run the Neural Net Code
	python neuralNet.py
5. To Run the k-Nearest Neighbor Code:
	python kNearestNeighbor.py   

To see the differences between the Amazon Kickstarter Dataset and the Google Playstore Dataset, simply enter the python code that you wish to view the differences between and uncomment the following lines:

FOR AMAZON KICK STARTER:

Go to the line that states:

	# START AMAZON KICK STARTER DATA PREPROCESSING

Uncomment the lines below it until the:
	# X = dataset

Them comment the lines after:

	# # START GOOGLE APP STORE DATA PREPROCESSING

Until:
	X = dataset

The code should be able to run for the Amazon Kick Starter Dataset.

* If running the Amazon Kickstarter Decsision Tree Python Code, there is a code that creates the Decision Tree as a PDF form. 

Ensure that this is the format:
dot_data = tree.export_graphviz(classifier, out_file=None,
                         # feature_names=X,
                         class_names=list(stateLabels.classes_),
                         # class_names=['1.0', '2.0','3.0', '4.0', '5.0'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Amazon Decision Tree Visual")

FOR GOOGLE PLAYSTORE DATASET:

Go to the line that states:

	# START GOOGLE APP STORE DATA PREPROCESSING

Uncomment the lines below it until the:
	# X = dataset

Them comment the lines after:

	# START AMAZON KICK STARTER DATA PREPROCESSING

Until:
	X = dataset

The code should be able to run for the Google Playstore Dataset.

* If running the Amazon Kickstarter Decsision Tree Python Code, there is a code that creates the Decision Tree as a PDF form. 

Ensure that this is the format:
dot_data = tree.export_graphviz(classifier, out_file=None,
                         # feature_names=X,
                         # class_names=list(stateLabels.classes_),
                         class_names=['1.0', '2.0','3.0', '4.0', '5.0'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Google Decision Tree Visual")


