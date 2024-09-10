import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
housing_data = r"...\Machine Learning\DataSet\2-HousingData.csv"
df = pd.read_csv(housing_data)

# 1. Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# 2. General information of the dataset (data types, missing values)
print("\nGeneral information of the dataset:")
print(df.info())

# 3. Display summary statistics
print("\nSummary statistics of the dataset:")
print(df.describe())

# 4. Analyze missing data
print("\nMissing data status (per column):")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_data_summary = pd.DataFrame({'Missing Value Count': missing_data, 'Missing Value Percentage (%)': missing_percentage})
print(missing_data_summary)

# 5. Strategy for missing data (fill with mean or median)
df.fillna(df.mean(), inplace=True)  # Fill missing data with mean

# 6. Display the correlation matrix
print("\nCorrelation matrix:")
corr_matrix = df.corr()
print(corr_matrix)

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 7. Visualize data distribution using histograms
df.hist(bins=20, figsize=(20, 15))
plt.show()

# 8. Analyze the distribution of the target variable MEDV
sns.histplot(df['MEDV'], kde=True)
plt.title("Distribution of MEDV (Target Variable)")
plt.show()

# Basic linear regression (trial 1)
print('trial 1')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Separate the features and the target variable (MEDV)
X = df.drop('MEDV', axis=1)  # Features
y = df['MEDV']  # Target variable

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model (Mean Squared Error and R² Score)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')

# Plot real vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()


# Display model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Plot true vs predicted values for the first linear regression model (trial 1)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label="Linear Regression Predictions", color='red', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Line")
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Linear Regression - Trial 1)')
plt.legend()
plt.show()

# Enhanced model with feature engineering (trial 2)
print('trial 2')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Fill missing values (with the mean)
df.fillna(df.mean(), inplace=True)

# Feature engineering: Apply log transformation (to model non-linear relationships)
df['LSTAT_log'] = np.log(df['LSTAT'] + 1)  # Logarithm of LSTAT (+1 to avoid log(0))
df['RM_sq'] = df['RM'] ** 2  # Square term for the number of rooms

# Interaction term: Interaction between NOX and INDUS
df['NOX_INDUS'] = df['NOX'] * df['INDUS']

# Separate features and target variable (MEDV)
X = df[['RM_sq', 'LSTAT_log', 'NOX_INDUS', 'CHAS', 'NOX', 'PTRATIO', 'DIS']]  # Selected and new features
y = df['MEDV']  # Target variable

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model (Mean Squared Error and R² Score)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')

# Display model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
# Plot true vs predicted values for the second linear regression model (trial 2)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label="Enhanced Linear Regression Predictions", color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Line")
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Enhanced Linear Regression - Trial 2)')
plt.legend()
plt.show()

# Ridge and Lasso regression (Trial 3)
print('trial 3')
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Data preparation (filling missing values and transformations)
df.fillna(df.mean(), inplace=True)
df['LSTAT_log'] = np.log(df['LSTAT'] + 1)  # Logarithm of LSTAT
df['RM_sq'] = df['RM'] ** 2  # Square of RM
df['NOX_INDUS'] = df['NOX'] * df['INDUS']  # Interaction between NOX and INDUS

# Define features and target variable
X = df[['RM_sq', 'LSTAT_log', 'NOX_INDUS', 'CHAS', 'NOX', 'PTRATIO', 'DIS']]  # Selected features
y = df['MEDV']  # Target variable (house price)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for Ridge and Lasso regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Ridge regression model
ridge_model = Ridge(alpha=1.0)  # alpha controls the regularization strength
ridge_model.fit(X_train_scaled, y_train)

# Create Lasso regression model
lasso_model = Lasso(alpha=0.1)  # alpha controls the regularization strength
lasso_model.fit(X_train_scaled, y_train)

# Make predictions with Ridge
y_pred_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Make predictions with Lasso
y_pred_lasso = lasso_model.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Print results
print(f"Ridge Mean Squared Error (MSE): {mse_ridge}")
print(f"Ridge R² Score: {r2_ridge}")
print(f"Lasso Mean Squared Error (MSE): {mse_lasso}")
print(f"Lasso R² Score: {r2_lasso}")

# Display Ridge and Lasso coefficients
ridge_coefficients = pd.DataFrame(ridge_model.coef_, X.columns, columns=['Ridge Coefficients'])
lasso_coefficients = pd.DataFrame(lasso_model.coef_, X.columns, columns=['Lasso Coefficients'])
print(ridge_coefficients)
print(lasso_coefficients)

# Plot true vs predicted values for Ridge
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, label="Ridge Predictions", color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Line")
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Ridge)')
plt.legend()
plt.show()

# Plot true vs predicted values for Lasso
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, label="Lasso Predictions", color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Line")
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Lasso)')
plt.legend()
plt.show()

# Polynomial regression (interaction terms)
print("Trial 4")
from sklearn.preprocessing import PolynomialFeatures

# Add polynomial interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and test sets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create and train the polynomial regression model
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_poly)

# Make predictions and evaluate the polynomial model
y_pred_poly = model_poly.predict(X_test_poly)
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
r2_poly = r2_score(y_test_poly, y_pred_poly)

# Print results of polynomial regression
print(f"Polynomial Mean Squared Error (MSE): {mse_poly}")
print(f"Polynomial R² Score: {r2_poly}")

# Plot true vs predicted values for Polynomial regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test_poly, y_pred_poly, label="Polynomial Predictions", color='orange', alpha=0.6)
plt.plot([y_test_poly.min(), y_test_poly.max()], [y_test_poly.min(), y_test_poly.max()], 'k--', lw=2, label="Ideal Line")
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Polynomial)')
plt.legend()
plt.show()


'''
 Code Explanation and Model Trials 

1. Data Loading and Analysis:
   - The dataset is loaded, and initial exploration is performed (missing values, summary statistics, correlation matrix).
   - Missing values are filled with the mean, and data distributions are visualized.

Trial 1: Basic Linear Regression
   - We create a simple linear regression model using all features.
   - Results: 
   - MSE: 25.02, R²: 0.6588.
   - The model explains 65.88% of the variance in housing prices.
   - Conclusion: Model lacks capturing non-linear relationships, leading to limited predictive power.

 Trial 2: Enhanced Linear Regression (Feature Engineering)
   - Feature engineering is applied (`LSTAT_log`, `RM_sq`, and `NOX_INDUS` interaction).
   - Results: 
   - MSE: 19.72, R²: 0.7311, showing improvement.
   - Conclusion: Non-linear transformations and interactions improved the model.
 Trial 3: Ridge and Lasso Regression
   - Ridge and Lasso regularization are applied to prevent overfitting.
   - Results: 
   - Ridge: MSE 19.71, R²: 0.7312.
   - Lasso: MSE 19.49, R²: 0.7342.
   - Conclusion: Regularization slightly improves performance and shrinks less relevant coefficients.

 Trial 4: Polynomial Regression
   - Polynomial regression with degree 2 is applied to capture more complex relationships.
   - Results: 
   - MSE: 14.00, R²: 0.8090, providing the best fit.
   - Conclusion: Polynomial regression significantly improves the model by capturing non-linear patterns.

'''
