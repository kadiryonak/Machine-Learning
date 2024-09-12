# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
file_path = r"...\Machine Learning\DataSet\1-titanic.csv"
titanic_data = pd.read_csv(file_path)

# 1. Check for missing values
missing_values = titanic_data.isnull().sum()

# 2. Calculate survival rate
survival_rate = titanic_data['Survived'].mean()

# 3. Survival rate by gender
survival_by_gender = titanic_data.groupby('Sex')['Survived'].mean()

# 4. Survival rate by class
survival_by_class = titanic_data.groupby('Pclass')['Survived'].mean()

# 5. Age distribution statistics
age_distribution = titanic_data['Age'].describe()

# 6. Survival rate by embarked location
survival_by_embarked = titanic_data.groupby('Embarked')['Survived'].mean()

# 7. Correlation matrix: selecting only numeric columns
numeric_columns = titanic_data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_columns.corr()

# 8. Mean age of survivors and non-survivors
mean_age_survived = titanic_data[titanic_data['Survived'] == 1]['Age'].mean()
mean_age_not_survived = titanic_data[titanic_data['Survived'] == 0]['Age'].mean()

# 9. Survival rate by family size (calculated by SibSp and Parch)
titanic_data['Family_Size'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
survival_by_family_size = titanic_data.groupby('Family_Size')['Survived'].mean()

# 10. Survival rate by ticket fare (grouped into quartiles)
fare_survived = titanic_data.groupby(pd.qcut(titanic_data['Fare'], 4))['Survived'].mean()

# 11. Survival rate by embarked location and class
survival_by_embarked_and_class = titanic_data.groupby(['Embarked', 'Pclass'])['Survived'].mean()

# Fill missing values for 'Age' and 'Fare'
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)

# Fill missing values for 'Embarked' with the most common value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into numeric values (Label Encoding)
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Add 'Is_Alone' feature (1 if alone, 0 otherwise)
titanic_data['Is_Alone'] = (titanic_data['Family_Size'] == 1).astype(int)

# Visualization: Survival rate by gender
plt.figure(figsize=(10, 6))

# Plot 1: Survival rate by gender
plt.subplot(2, 2, 1)
sns.barplot(x='Sex', y='Survived', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')

# Plot 2: Survival rate by class
plt.subplot(2, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=titanic_data)
plt.title('Survival Rate by Class')
plt.ylabel('Survival Rate')

# Plot 3: Age distribution and survival distribution by age
plt.subplot(2, 2, 3)
sns.histplot(titanic_data['Age'].dropna(), kde=True, color='blue', label='All Passengers', bins=30)
sns.histplot(titanic_data[titanic_data['Survived'] == 1]['Age'].dropna(), kde=True, color='green', label='Survivors', bins=30)
plt.legend()
plt.title('Age Distribution and Survivor Age Distribution')

# Plot 4: Survival rate by embarked location
plt.subplot(2, 2, 4)
sns.barplot(x='Embarked', y='Survived', data=titanic_data)
plt.title('Survival Rate by Embarked Location')
plt.ylabel('Survival Rate')

# Show all plots
plt.tight_layout()
plt.show()

# Display the calculated results
print("Missing Values:\n", missing_values)
print("\nSurvival Rate:", survival_rate)
print("\nSurvival Rate by Gender:\n", survival_by_gender)
print("\nSurvival Rate by Class:\n", survival_by_class)
print("\nAge Distribution:\n", age_distribution)
print("\nSurvival Rate by Embarked Location:\n", survival_by_embarked)
print("\nCorrelation Matrix:\n", correlation_matrix)
print("\nMean Age of Survivors:", mean_age_survived)
print("Mean Age of Non-Survivors:", mean_age_not_survived)
print("\nSurvival Rate by Family Size:\n", survival_by_family_size)
print("\nSurvival Rate by Ticket Fare:\n", fare_survived)
print("\nSurvival Rate by Embarked Location and Class:\n", survival_by_embarked_and_class)

# Drop unnecessary columns for modeling
titanic_data_cleaned = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Features and target variable
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family_Size', 'Is_Alone']
X = titanic_data_cleaned[features]
y = titanic_data_cleaned['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

# Calculate feature importance
importances = rf_model.feature_importances_

# Display feature importance
feature_importance_df = pd.DataFrame({
    'Features': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", feature_importance_df)


'''
This code analyzes the Titanic dataset to extract useful insights and train a machine learning model.

1. Data Loading and Preprocessing: 
   - The Titanic dataset is loaded, and missing values in the 'Age', 'Fare', and 'Embarked' columns are filled.
   - Categorical variables like 'Sex' and 'Embarked' are encoded as numeric values for machine learning models.
   - New features like 'Family_Size' (number of family members) and 'Is_Alone' (whether the passenger was alone) are created.

2. Exploratory Data Analysis (EDA):
   - Several statistics are computed: missing values, survival rates by gender, class, and embarked location.
   - Visualization: Bar plots and histograms are created to show survival rates by gender, class, age, and embarked location.
   - Correlation matrix: The correlation between numeric features is calculated to understand their relationships.

3. Model Training:
   - A Random Forest classifier is trained on the data to predict the 'Survived' target variable.
   - The data is split into training and test sets, and the model is trained using the training data.
   - The model's accuracy is evaluated using the test data, and it outputs the accuracy score.

4. Feature Importance:
   - The feature importance is calculated and displayed, showing which features have the most impact on predicting survival.

Through this analysis, we learn that gender (especially being female), class, and ticket fare are strong indicators of survival, while family size and age also play a role. This insight allows us to create predictive models that can accurately estimate survival rates based on these features.
'''
