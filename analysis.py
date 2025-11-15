
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to save plots
output_dir = "analysis_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading and Initial Inspection ---
try:
    df = pd.read_csv("diabetes_prediction_dataset.csv")
except FileNotFoundError:
    print("Error: 'diabetes_prediction_dataset.csv' not found. Make sure the file is in the correct directory.")
    exit()

# --- 2. Data Preprocessing ---
print("Starting Data Preprocessing...")

# Drop duplicate rows
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"- Removed {initial_rows - len(df)} duplicate rows.")

# Handle 'Other' in 'gender' column by removing it
if 'gender' in df.columns:
    initial_rows = len(df)
    df = df[df['gender'] != 'Other']
    print(f"- Removed {initial_rows - len(df)} rows where gender is 'Other'.")

# Convert categorical columns to numerical using one-hot encoding
# 'smoking_history' and 'gender' are categorical
df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
print("- Converted 'gender' and 'smoking_history' to numerical format using one-hot encoding.")

# The target variable 'diabetes' is already numerical (0 or 1)

print("Data Preprocessing Complete.")
print("\n" + "="*50 + "\n")


# --- 3. Exploratory Data Analysis (EDA) ---
print("Starting Exploratory Data Analysis (EDA)...")

# a. Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())

# b. Target Variable Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='diabetes', data=df)
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Diabetes (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'diabetes_distribution.png'))
plt.close()
print("- Generated plot for diabetes distribution.")

# c. Numerical Feature Distributions (Histograms)
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
df[numerical_features].hist(bins=20, figsize=(14, 10))
plt.suptitle('Distribution of Numerical Features')
plt.savefig(os.path.join(output_dir, 'numerical_features_histograms.png'))
plt.close()
print("- Generated histograms for numerical features.")

# d. Correlation Matrix
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()
print("- Generated correlation matrix heatmap.")

# e. Age vs. Diabetes
plt.figure(figsize=(10, 7))
sns.boxplot(x='diabetes', y='age', data=df)
plt.title('Age Distribution by Diabetes Outcome')
plt.savefig(os.path.join(output_dir, 'age_vs_diabetes.png'))
plt.close()
print("- Generated boxplot for Age vs. Diabetes.")

# f. BMI vs. Diabetes
plt.figure(figsize=(10, 7))
sns.boxplot(x='diabetes', y='bmi', data=df)
plt.title('BMI Distribution by Diabetes Outcome')
plt.savefig(os.path.join(output_dir, 'bmi_vs_diabetes.png'))
plt.close()
print("- Generated boxplot for BMI vs. Diabetes.")

print("Exploratory Data Analysis Complete.")
print(f"\nAll plots have been saved in the '{output_dir}' directory.")

