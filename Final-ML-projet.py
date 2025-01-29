import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path = "/workspaces/ML-project/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
data = pd.read_csv(file_path)

# Display the first few rows and general information to understand the structure
data_info = data.info()
data_head = data.head()

data_info, data_head

# Select relevant columns for the analysis
relevant_columns = ['Education', 'Smoking', 'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)']
data_relevant = data[relevant_columns]

# Check for missing values in the selected columns
missing_values = data_relevant.isnull().sum()

# Drop rows with missing values in the selected columns
data_cleaned = data_relevant.dropna()

# Display the cleaned data and missing values report
missing_values, data_cleaned.head(), data_cleaned.info()

# Correct filtering by exact matching for specific unwanted Education values
data_filtered = data_cleaned[~data_cleaned['Education'].isin(['Somato meeting', 'none (Hauptschule not finished)'])]

# Display filtered data and unique Education values to confirm correctness
data_filtered.info(), data_filtered['Education'].unique(), data_filtered.head()

# Correct the typo "Gymansium" to "Gymnasium" in the Education column
data_filtered['Education'] = data_filtered['Education'].replace('Gymansium', 'Gymnasium')

# Display unique Education values to confirm the correction
data_filtered['Education'].unique()

# Encode the Education column into numeric values
education_mapping = {'Hauptschule': 1, 'Realschule': 2, 'Gymnasium': 3}
data_filtered['Education_encoded'] = data_filtered['Education'].map(education_mapping)

# Drop the original Education column
data_encoded = data_filtered.drop(columns=['Education'])

# Display the updated dataset with encoded education values
data_encoded.head()

# Set up the plotting environment
sns.set(style="whitegrid")

# Distribution of Education levels
plt.figure(figsize=(8, 6))
sns.countplot(x='Education_encoded', data=data_encoded, palette='viridis')
plt.title('Distribution of Education Levels', fontsize=14)
plt.xlabel('Education Level (Encoded)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(ticks=[0, 1, 2], labels=['Hauptschule (1)', 'Realschule (2)', 'Gymnasium (3)'])
plt.show()

# Distribution of Smoking behavior
plt.figure(figsize=(8, 6))
sns.countplot(x='Smoking', data=data_encoded, palette='magma', order=data_encoded['Smoking'].value_counts().index)
plt.title('Distribution of Smoking Behavior', fontsize=14)
plt.xlabel('Smoking Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Correlation heatmap
correlation_data = data_encoded.drop(columns=['Smoking'])  # Exclude non-numeric columns
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=14)
plt.show()

# Define features and target
X = data_encoded[['Education_encoded']]  # Predictor
y = data_encoded['Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10,  # Number of random combinations to try
    scoring='accuracy',
    cv=5,  # Number of cross-validation folds
    n_jobs=-1,  # Use all available CPU cores
    random_state=42,
    verbose=1
)

# Perform hyperparameter tuning
random_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("Best hyperparameters:", best_params)

# Evaluate the best model
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy of the best model: {accuracy_best}")

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, conf_matrix, classification_rep

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=rf_model,  # RandomForestClassifier instance
    X=X,  # Feature data
    y=y,  # Target variable
    train_sizes=np.linspace(0.1, 1.0, 10),  # Training set sizes to evaluate
    cv=5,  # Number of cross-validation folds
    scoring='accuracy',  # Scoring metric
)

# Calculate mean and standard deviation of scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, label='Cross-validation Score', color='green')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.title('Learning Curves for Random Forest Classifier')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
plt.grid()
plt.show()

# Display evaluation metrics for the initial model
print("Initial Model Evaluation:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Compare the accuracy of the initial and tuned models
print("\nAccuracy Comparison:")
print("Initial Model Accuracy:", accuracy)
print("Tuned Model Accuracy:", accuracy_best)
print("Accuracy Improvement:", accuracy_best - accuracy)