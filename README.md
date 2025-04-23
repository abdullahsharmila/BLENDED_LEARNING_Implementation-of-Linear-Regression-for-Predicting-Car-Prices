# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
###  Developed by: Abdullah R
### RegisterNumber: 212223230004
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Required Libraries**: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.  
2. **Dataset Loading**: Load the dataset containing car prices and associated features.  
3. **Data Preparation**: Address missing data and perform feature selection if needed.  
4. **Data Splitting**: Divide the dataset into training and testing subsets.  
5. **Model Training**: Develop a linear regression model and train it using the training data.  
6. **Prediction Generation**: Apply the model to predict outcomes for the test dataset.  
7. **Model Evaluation**: Evaluate the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.  
8. **Assumption Verification**: Analyze residual plots to check for homoscedasticity, normal distribution, and linearity.  
9. **Result Presentation**: Display the predictions and performance evaluation metrics.  

## Program:
```
#  Program to implement linear regression model for predicting car prices and test assumptions.
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv'
df = pd.read_csv(url)

# Select relevant features and target variable
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]  # Features
y = df['price']  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# 1. Assumption: Linearity
plt.scatter(y_test, y_pred)
plt.title("Linearity: Observed vs Predicted Prices")
plt.xlabel("Observed Prices")
plt.ylabel("Predicted Prices")
plt.show()

# 2. Assumption: Independence (Durbin-Watson test)
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_test}")

# 3. Assumption: Homoscedasticity
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Homoscedasticity: Residuals vs Predicted Prices")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.show()

# 4. Assumption: Normality of residuals
sns.histplot(residuals, kde=True)
plt.title("Normality: Histogram of Residuals")
plt.show()

sm.qqplot(residuals, line='45')
plt.title("Normality: Q-Q Plot of Residuals")
plt.show()
```

## Output:
<img width="691" alt="Screenshot 2024-11-17 at 6 57 33 PM" src="https://github.com/user-attachments/assets/e8edf7da-c7d3-4d8c-b2c1-314573edabfd">
<img width="691" alt="Screenshot 2024-11-17 at 6 57 40 PM" src="https://github.com/user-attachments/assets/8ab707aa-08e0-46bc-9001-53dd2fcc03b0">




## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
