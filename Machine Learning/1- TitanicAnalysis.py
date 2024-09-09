# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "1-titanic.csv"
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
titanic_data['Family_Size'] = titanic_data['SibSp'] + titanic_data['Parch']
survival_by_family_size = titanic_data.groupby('Family_Size')['Survived'].mean()

# 10. Survival rate by ticket fare (grouped into quartiles)
fare_survived = titanic_data.groupby(pd.qcut(titanic_data['Fare'], 4))['Survived'].mean()

# 11. Survival rate by embarked location and class
survival_by_embarked_and_class = titanic_data.groupby(['Embarked', 'Pclass'])['Survived'].mean()

# Data Preprocessing

# Fill missing values for 'Age' and 'Fare'
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)

# Fill missing values for 'Embarked' with the most common value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into numeric values (Label Encoding)
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

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
