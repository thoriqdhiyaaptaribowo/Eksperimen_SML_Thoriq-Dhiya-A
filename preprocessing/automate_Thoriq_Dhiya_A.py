import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

le = LabelEncoder()
scaler = StandardScaler()

def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Handle missing values
    df.dropna()

    # Remove Outliers
    num_var = ['Age', 'Height', 'Weight']
    for var in num_var:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)]

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Normalize numerical features
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Print into csv
    df.to_csv('preprocessed_data.csv', index=False)
    return df, label_encoders

preprocess_data('ObesityDataSet_raw_and_data_sinthetic.csv')

os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(le, f)