# Importing the libraries
import sys

import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Read the data
my_data = pd.read_csv('drug.csv')

# Extract the features
X = my_data.drop('Drug', axis=1).values
y = my_data['Drug']

# Preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define the decision tree
tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train the model
tree.fit(X_train, y_train)

# Predict the model
predictions = tree.predict(X_test)

# Print the accuracy
print("Accuracy : " + str(sum(predictions == y_test) / len(y_test)))
