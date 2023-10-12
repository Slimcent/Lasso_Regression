import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# L1 regularization (Lasso) adds the absolute values of the model's coefficients (weights) as a penalty term to the cost
# function.

# Sample data
data = {
    'X1': [1, 2, 3, 4, 5],
    'X2': [3, 4, 2, 5, 1],
    'X3': [7, 5, 10, 8, 6],
    'y': [10, 20, 25, 30, 40]
}

df = pd.DataFrame(data)

# Independent variables (features)
X = df[['X1', 'X2', 'X3']]

# Dependent variable
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a Lasso regression model
lasso = Lasso(alpha=0.1)  # You can adjust the regularization strength with the alpha parameter
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Get the model coefficients (including regularization)
coefficients = lasso.coef_
intercept = lasso.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)
