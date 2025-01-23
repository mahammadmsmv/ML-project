import pandas as pd

# Load the dataset
file_path = '/workspaces/ML-project/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
data = pd.read_csv(file_path)
print("Dataset loaded successfully.")

# Select relevant columns
data = data.rename(columns={
    'Education': 'Education_Level',
    'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)': 'Smoking_Status'
})
relevant_data = data[['Education_Level', 'Smoking_Status']]
print("Relevant columns selected.")

# Drop rows with missing values
cleaned_data = relevant_data.dropna()


# Encode categorical variables
cleaned_data['Education_Level'] = cleaned_data['Education_Level'].astype('category').cat.codes
cleaned_data['Smoking_Status'] = cleaned_data['Smoking_Status'].astype(int)
print("Categorical variables encoded.")

# Verify the cleaned data
print("Cleaned Data Sample:")
print(cleaned_data.head())
