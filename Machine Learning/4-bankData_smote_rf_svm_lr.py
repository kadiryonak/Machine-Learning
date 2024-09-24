import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE  # SMOTE kütüphanesini ekledik
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# File Path
file_path = r"...Machine Learning\DataSet\bank-full.csv"
# Read Data
df = pd.read_csv(file_path, sep=';')

# Learn about the data set
print(df.info())
print(df.head())

# We use LabelEncoder to encode categorical attributes
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

# Separation into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Let's apply SMOTE to multiply the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Let's examine the distribution of classes after SMOTE is applied
print(f"SMOTE sonrası sınıfların dağılımı: {np.bincount(y_train_smote)}")

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with training data replicated with SMOTE
rf_model.fit(X_train_smote, y_train_smote)

# Make predictions on test data
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # For ROC-AUC we will need

# Calculate Sensitivity, Recall, F1-Score and ROC-AUC
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# A more detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Let's scale the data for the SVM model 
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Initialize SVM model
svm_model = SVC(probability=True, random_state=42, class_weight='balanced')

# Train the model with SMOTE with replicated and scaled training data
svm_model.fit(X_train_smote_scaled, y_train_smote)

# Make predictions on test data
y_pred = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]  # ROC-AUC için ihtiyacımız olacak

# Calculate Sensitivity, Recall, F1-Score and ROC-AUC
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# A more detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Show feature importance again
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
rf_model.fit(X_train, y_train)

# Take feature importance
feature_importances = rf_model.feature_importances_

# Visualize feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# The most important features: Let's pick the top 5 features
important_features = feature_importance_df['Feature'].head(5).values
X_important = X_train[important_features]

# Let's use the same properties in the test data
X_test_important = X_test[important_features]

# New model training (with important features)
rf_model_important = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_important.fit(X_important, y_train)

# Forecasting with advanced metrics and evaluating results
y_pred_important = rf_model_important.predict(X_test_important)
y_pred_proba_important = rf_model_important.predict_proba(X_test_important)[:, 1]

# Advanced metrics
precision = precision_score(y_test, y_pred_important, average='binary')
recall = recall_score(y_test, y_pred_important, average='binary')
f1 = f1_score(y_test, y_pred_important, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba_important)

# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_important))


from sklearn.linear_model import LogisticRegression

# Create and train the logistic regression model
model_loj = LogisticRegression()
model_loj.fit(X_important, y_train)

y_pred_important = model_loj.predict(X_test_important)
y_pred_proba_important = model_loj.predict_proba(X_test_important)[:, 1]

# Advanced metrics
precision = precision_score(y_test, y_pred_important, average='binary')
recall = recall_score(y_test, y_pred_important, average='binary')
f1 = f1_score(y_test, y_pred_important, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba_important)


# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_important))
