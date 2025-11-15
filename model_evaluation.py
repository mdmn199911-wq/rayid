
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Use a non-interactive backend for matplotlib
plt.switch_backend('Agg')

# --- 1. Load Data and Recreate Test Set ---
print("Loading data and recreating test set...")
try:
    df = pd.read_csv("diabetes_prediction_dataset.csv")
except FileNotFoundError:
    print("Error: 'diabetes_prediction_dataset.csv' not found.")
    exit()

df.drop_duplicates(inplace=True)
df = df[df['gender'] != 'Other']

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

# Use the same random_state to get the exact same split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Test set recreated successfully.")

# --- 2. Load Models and Scaler ---
models_dir = "trained_models"
scaler_path = os.path.join(models_dir, 'scaler.joblib')
log_reg_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
rf_path = os.path.join(models_dir, 'random_forest_model.joblib')
mlp_path = os.path.join(models_dir, 'neural_network_model.joblib')

try:
    scaler = joblib.load(scaler_path)
    log_reg = joblib.load(log_reg_path)
    rf = joblib.load(rf_path)
    mlp = joblib.load(mlp_path)
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Make sure you have run the training script first.")
    exit()

print("Models and scaler loaded successfully.")

# --- 3. Scale Test Data ---
X_test_scaled = scaler.transform(X_test)

# --- 4. Evaluate Models ---
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "Neural Network (MLP)": mlp
}

results = {}

# Create directory for confusion matrix plots
output_dir = "evaluation_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    
    # Use scaled data for LR and MLP, original for RF (as it was trained)
    if name == "Random Forest":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'{name.replace(" ", "_")}_confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix plot saved for {name}.")

print("\n--- Overall Results Summary ---")
results_df = pd.DataFrame(results).T
print(results_df.round(4))

