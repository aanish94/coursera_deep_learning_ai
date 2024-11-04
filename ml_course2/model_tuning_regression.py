# for array computations and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for building and training neural networks
import tensorflow as tf

# custom functions
# import utils

# Load the dataset from the text file
data = np.loadtxt('utils/data/data_w3_ex1.csv', delimiter=',')

# Split the inputs and outputs into separate arrays
x = data[:,0]
y = data[:,1]

# Convert 1-D arrays into 2-D because the commands later will require it
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

# Feature Scaling
scaler_linear = StandardScaler()
X_train_scaled = scaler_linear.fit_transform(x_train)

# Train the model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Evaluate model
yhat = linear_model.predict(X_train_scaled)

# print(f"Training MSE: {mean_squared_error(yhat, y_train) / 2}")

# Use cross-validation data
X_cv_scaled = scaler_linear.transform(x_cv)
yhat = linear_model.predict(X_cv_scaled)

# print(f"CV MSE: {mean_squared_error(yhat, y_cv) / 2}")

# Try adding polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)

# Scale inputs again
scaler_poly = StandardScaler()
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

# Train model
model = LinearRegression()
model.fit(X_train_mapped_scaled, y_train)

# Evaluate and calculate MSE
yhat = model.predict(X_train_mapped_scaled)
# print(f"Training MSE: {mean_squared_error(yhat, y_train) / 2}")

# Add polynominal features to CV set
X_cv_mapped = poly.transform(x_cv)
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

yhat = model.predict(X_cv_mapped_scaled)
# print(f"CV MSE: {mean_squared_error(yhat, y_cv) / 2}")

"""Generalize"""

# Initialize lists to save errors, models and feature transforms
train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

# Loop from degree 1 to 10
for degree in range(1, 11):

    # Add polynomial features to the dataset
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)

    # Scale training data
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # Create and train model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    # Compute training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(yhat, y_train) / 2
    train_mses.append(train_mse)

    # Add polynomial features to the CV set
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute CV MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(yhat, y_cv) / 2
    cv_mses.append(cv_mse)

# Get the model with the lowest CV MSE (add 1 because list indices start at 0)
# This also corresponds to the degree of the polynomial added
degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")

# Compute generalization error against the test set
# Add polynomial features to the test set
X_test_mapped = polys[degree-1].transform(x_test)

# Scale the test set
X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

# Compute the test MSE
yhat = models[degree-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")
