
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier

# Use a non-interactive backend for matplotlib
plt.switch_backend('Agg')

# Create a directory to save plots if it doesn't exist
output_dir = "analysis_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading and Preprocessing (same as before) ---
print("Loading and preprocessing data...")
try:
    df = pd.read_csv("diabetes_prediction_dataset.csv")
except FileNotFoundError:
    print("Error: 'diabetes_prediction_dataset.csv' not found.")
    exit()

df.drop_duplicates(inplace=True)
df = df[df['gender'] != 'Other']
df_processed = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
print("Initial preprocessing complete.")

# --- 2. Feature Engineering ---
print("Starting Feature Engineering...")

# Create BMI categories
bins = [0, 18.5, 25, 30, float('inf')]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df_processed['bmi_category'] = pd.cut(df_processed['bmi'], bins=bins, labels=labels)

# One-hot encode the new BMI category
df_engineered = pd.get_dummies(df_processed, columns=['bmi_category'], drop_first=True)

print("- Engineered 'bmi_category' feature.")
print("Feature Engineering Complete.")

# --- 3. Feature Selection ---
print("Starting Feature Selection...")

# Separate features (X) and target (y)
X = df_engineered.drop('diabetes', axis=1)
y = df_engineered['diabetes']

# Use RandomForestClassifier to find feature importances
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("\nFeature Importances (from most to least important):")
print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance for Diabetes Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()

# Save the plot
plot_path = os.path.join(output_dir, 'feature_importances.png')
plt.savefig(plot_path)

print(f"\nFeature importance plot saved to '{plot_path}'.")
print("Feature Selection Complete.")

