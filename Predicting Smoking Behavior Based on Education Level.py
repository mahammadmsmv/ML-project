# Project: Predicting Smoking Behavior Based on Education Level
# Author: Mahammad Masimov
# Date: January 2025

# Importing Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load and Inspect the Dataset
# Loading the dataset
file_path = '/workspaces/ML-project/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset Head:")
print(data.head())

# Step 2: Data Cleaning
# Selecting relevant columns for the project
# Renaming columns for easier reference
data = data.rename(columns={
    'Education': 'Education_Level',
    'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)': 'Smoking_Status'
})

# Selecting relevant columns
cleaned_data = data[['Education_Level', 'Smoking_Status']].copy()

# Dropping rows with missing values
cleaned_data = cleaned_data.dropna()

# Encoding categorical data in 'Education_Level'
cleaned_data['Education_Level'] = cleaned_data['Education_Level'].astype('category').cat.codes

# Converting 'Smoking_Status' to integer type
cleaned_data['Smoking_Status'] = cleaned_data['Smoking_Status'].astype(int)

print("\nCleaned Data Head:")
print(cleaned_data.head())

# Step 3: Splitting Data
# Splitting data into features (X) and target (y)
X = cleaned_data[['Education_Level']]
y = cleaned_data['Smoking_Status']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Model Building and Evaluation
# Logistic Regression Model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

# Decision Tree Model
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
dec_tree_preds = dec_tree.predict(X_test)

# Random Forest Model
rand_forest = RandomForestClassifier(random_state=42, n_estimators=100)
rand_forest.fit(X_train, y_train)
rand_forest_preds = rand_forest.predict(X_test)

# Generating and displaying classification reports
log_reg_report = classification_report(y_test, log_reg_preds)
dec_tree_report = classification_report(y_test, dec_tree_preds)
rand_forest_report = classification_report(y_test, rand_forest_preds)

print("\nLogistic Regression Report:")
print(log_reg_report)

print("\nDecision Tree Report:")
print(dec_tree_report)

print("\nRandom Forest Report:")
print(rand_forest_report)