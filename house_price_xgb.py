import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Read the data
data = pd.read_csv('house_prices/train.csv')

# Drop houses where the target is missing
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Select target and features
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

# Split data into training and validation data, for both features and target
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.25)

# Imputation
train_X = SimpleImputer().fit_transform(train_X)
test_X = SimpleImputer().fit_transform(test_X)

# Define model
my_model = XGBRegressor()

# Fit model
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)

# Print MAE
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# Define new model with a few parameters changed
my_model = XGBRegressor(
    n_estimators=1000, early_stopping_rounds=5, learning_rate=0.05)

# Fit model with early stopping
my_model.fit(train_X, train_y,
             eval_set=[(test_X, test_y)], verbose=False)

# make predictions
predictions = my_model.predict(test_X)

# Print MAE
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# Plot the predictions
plt.scatter(np.linspace(
    0, 10, test_y.shape[0]), test_y, c='b', label='True Values')
plt.scatter(np.linspace(
    0, 10, test_y.shape[0]), predictions, c='r', label='Predictions')
plt.legend()
plt.show()
