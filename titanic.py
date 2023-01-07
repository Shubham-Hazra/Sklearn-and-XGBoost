# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Read the data
df = pd.read_csv('titanic/train.csv', index_col='PassengerId')

# Preprocessing
modes = df.mode().iloc[0]
df['Fare'] = df.Fare.fillna(0)
df.fillna(modes, inplace=True)
df['LogFare'] = np.log1p(df['Fare'])
df['Embarked'] = pd.Categorical(df.Embarked)
df['Sex'] = pd.Categorical(df.Sex)

# Separate target from predictors
y = df.Survived
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X[['Sex', 'Embarked']] = X[['Sex', 'Embarked']].apply(lambda x: x.cat.codes)

# Break off validation set from training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Imputation
X_train = SimpleImputer().fit_transform(X_train)
X_test = SimpleImputer().fit_transform(X_test)

# Create a decision tree classifier
clf = DecisionTreeClassifier(
    max_leaf_nodes=10, min_samples_split=5)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = clf.predict(X_test)

# Compute the accuracy of the classifier
accuracy = clf.score(X_test, y_test)
print("Accuracy: " + str(accuracy))
