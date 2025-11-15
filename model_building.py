
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import os

# --- 1. Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
try:
    df = pd.read_csv("diabetes_prediction_dataset.csv")
except FileNotFoundError:
    print("Error: 'diabetes_prediction_dataset.csv' not found.")
    exit()

df.drop_duplicates(inplace=True)
df = df[df['gender'] != 'Other']
print("Initial preprocessing complete.")

# --- 2. Feature Selection ---
# Based on the recommendation from the previous step
features = [
    'HbA1c_level',
    'blood_glucose_level',
    'bmi',
    'age',
    'hypertension',
    'heart_disease'
]
target = 'diabetes'

X = df[features]
y = df[target]

print(f"Selected {len(features)} features for model building.")

# --- 3. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into 80% training and 20% testing sets.")

# --- 4. Data Scaling ---
# Scaling is important for Logistic Regression and Neural Networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled using StandardScaler.")

# --- 5. Model Building & Training ---
print("\nStarting model training...")

# a. Logistic Regression
print("- Training Logistic Regression model...")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# b. Random Forest
print("- Training Random Forest model...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train) # RF doesn't strictly require scaling, can be trained on original data

# c. Neural Network (MLPClassifier)
print("- Training Neural Network (MLP) model...")
mlp = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50))
mlp.fit(X_train_scaled, y_train)

print("All models have been trained successfully.")

# --- 6. Save Models ---
output_dir = "trained_models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the scaler as well, it's needed for future predictions
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
joblib.dump(log_reg, os.path.join(output_dir, 'logistic_regression_model.joblib'))
joblib.dump(rf, os.path.join(output_dir, 'random_forest_model.joblib'))
joblib.dump(mlp, os.path.join(output_dir, 'neural_network_model.joblib'))

print(f"\nAll models and the scaler have been saved to the '{output_dir}' directory.")
