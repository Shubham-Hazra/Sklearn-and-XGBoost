# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Read the data
data = pd.read_csv('house_prices/train.csv', index_col='Id')

# Drop houses where the target is missing
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Separate target from predictors
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Imputation
X_train = SimpleImputer().fit_transform(X_train)
X_test = SimpleImputer().fit_transform(X_test)


# Create a decision tree classifier
clf = DecisionTreeRegressor()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = clf.predict(X_test)

# Compute and print the mean absolute error of the forest model
mae = mean_absolute_error(y_test, preds)
print("Mean Absolute Error: " + str(mae))

# Plot the predictions
plt.scatter(np.linspace(
    0, 10, y_test.shape[0]), y_test, c='b', label='True Values')
plt.scatter(np.linspace(
    0, 10, y_test.shape[0]), preds, c='r', label='Predictions')
plt.legend()
plt.show()
