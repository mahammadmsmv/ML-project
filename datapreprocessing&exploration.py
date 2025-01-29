import pandas as pd

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

import matplotlib.pyplot as plt
import seaborn as sns

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





