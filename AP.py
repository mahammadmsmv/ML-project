
#Project: Predicting Smoking Behavior Based on Education Level
# Author: Arzu Ibadullayeva
# Date: January 2025

# Importing Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load and Inspect the Dataset
file_path = '/workspaces/ML-project/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
data = pd.read_csv(file_path)

print("Dataset Head:")
print(data.head())

# Step 2: Data Cleaning
data = data.rename(columns={
    'Education': 'Education_Level',
    'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)': 'Smoking_Status'
})

cleaned_data = data[['Education_Level', 'Smoking_Status']].copy()
cleaned_data = cleaned_data.dropna()
cleaned_data['Education_Level'] = cleaned_data['Education_Level'].astype('category').cat.codes
cleaned_data['Smoking_Status'] = cleaned_data['Smoking_Status'].astype(int)

print("\nCleaned Data Head:")
print(cleaned_data.head())

# Step 3: Splitting Data
X = cleaned_data[['Education_Level']]
y = cleaned_data['Smoking_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Model Building and Evaluation
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
dec_tree_preds = dec_tree.predict(X_test)

rand_forest = RandomForestClassifier(random_state=42, n_estimators=100)
rand_forest.fit(X_train, y_train)
rand_forest_preds = rand_forest.predict(X_test)

print("\nDecision Tree Report:")
print(classification_report(y_test, dec_tree_preds))

print("\nRandom Forest Report:")
print(classification_report(y_test, rand_forest_preds))