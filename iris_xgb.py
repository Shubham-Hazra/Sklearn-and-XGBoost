import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Read the data
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and validation data, for both features and target
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

# Define model
my_model = XGBClassifier()

# Fit model
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(X)

# Print Accuracy
print("Accuracy : " + str(sum(predictions == y) / len(y)))
