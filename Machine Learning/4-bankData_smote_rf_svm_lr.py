# =============================================================================
# Bank Marketing Dataset - Classification with SMOTE, RF, SVM, and Logistic Regression
# This script demonstrates handling imbalanced data and comparing classifiers
# =============================================================================

# Import all necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report)
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. Data Loading and Exploration
# =============================================================================

# File Path
file_path = os.path.join(os.path.dirname(__file__), "DataSet", "4-bank-full.csv")

# Read Data
df = pd.read_csv(file_path, sep=';')

# Learn about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# =============================================================================
# 2. Data Preprocessing
# =============================================================================

# Encode categorical attributes using LabelEncoder
df_encoded = df.copy()
label_encoders = {}
categorical_columns = df_encoded.select_dtypes(include=['object']).columns

for column in categorical_columns:
    if column != 'y':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le

# Separate features (X) and target variable (y)
X = df_encoded.drop('y', axis=1)
y = LabelEncoder().fit_transform(df_encoded['y'])

# Split into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================================================================
# 3. Apply SMOTE to Handle Class Imbalance
# =============================================================================

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nClass distribution after SMOTE: {np.bincount(y_train_smote)}")

# =============================================================================
# 4. Random Forest Classifier with SMOTE
# =============================================================================
print("\n" + "="*50)
print("Random Forest with SMOTE")
print("="*50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# =============================================================================
# 5. SVM Classifier with SMOTE
# =============================================================================
print("\n" + "="*50)
print("SVM with SMOTE")
print("="*50)

# Scale data for SVM
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(probability=True, random_state=42, class_weight='balanced')
svm_model.fit(X_train_smote_scaled, y_train_smote)

y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_svm):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_svm):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# =============================================================================
# 6. Feature Importance Analysis
# =============================================================================
print("\n" + "="*50)
print("Feature Importance Analysis")
print("="*50)

# Train a new RF model on original (unbalanced) data for feature importance
rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_importance.fit(X_train, y_train)

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_importance.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# =============================================================================
# 7. Model with Top 5 Important Features
# =============================================================================
print("\n" + "="*50)
print("Random Forest with Top 5 Features")
print("="*50)

important_features = feature_importance_df['Feature'].head(5).values
X_important_train = X_train[important_features]
X_important_test = X_test[important_features]

rf_important = RandomForestClassifier(n_estimators=100, random_state=42)
rf_important.fit(X_important_train, y_train)

y_pred_important = rf_important.predict(X_important_test)
y_pred_proba_important = rf_important.predict_proba(X_important_test)[:, 1]

print(f"Precision: {precision_score(y_test, y_pred_important):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_important):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_important):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_important):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_important))

# =============================================================================
# 8. Logistic Regression with Important Features
# =============================================================================
print("\n" + "="*50)
print("Logistic Regression with Top 5 Features")
print("="*50)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_important_train, y_train)

y_pred_lr = lr_model.predict(X_important_test)
y_pred_proba_lr = lr_model.predict_proba(X_important_test)[:, 1]

print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# =============================================================================
# Summary and Explanation
# =============================================================================
'''
This code demonstrates classification on an imbalanced banking dataset:

1. Data Preprocessing:
   - Load bank marketing data and encode categorical variables
   - Split into training (70%) and test (30%) sets

2. Handling Class Imbalance:
   - SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples
   - Balances the minority class for better model training

3. Model Comparison:
   - Random Forest with SMOTE: Good baseline performance
   - SVM with SMOTE: Slightly improved performance after scaling
   - Feature Selection: Using top 5 features slightly improves results
   - Logistic Regression: Simple but effective, ROC-AUC improved

Key Insights:
- SMOTE helps improve recall for the minority class
- Feature selection can reduce complexity without losing performance
- Different algorithms have different strengths for imbalanced data
- For better results, try hyperparameter optimization or ensemble methods
'''
