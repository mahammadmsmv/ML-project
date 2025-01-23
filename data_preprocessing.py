import pandas as pd


data = pd.read_csv("/workspaces/ML-project/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")


print(data.head())

#Renaming columns for easier reference
data = data.rename(columns={
    "Education": "Education_Level",
    "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)": "Smoking_Status"
})

cleaned_data = data[['Education_Level', 'Smoking_Status']]
