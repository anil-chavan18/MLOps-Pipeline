#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def data_preparation():
    print("Starting data preparation...")
    df = pd.read_excel('/app/large_bank_data.xlsm')
    df.dropna(inplace=True)
    df['age_loan_ratio'] = df['age'] / (df['loan_status'] + 1e-6)
    output_path = '/tmp/processed_data.csv'
    df.to_csv(output_path, index=False)
    print("Data preparation complete.")
    return output_path

def model_training(data_path):
    print("Starting model training...")
    df = pd.read_csv(data_path)
    X = df[['age', 'balance', 'loan_status', 'campaign_calls']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    model_path = '/tmp/model.pkl'
    joblib.dump(model, model_path)
    print("Model training complete.")
    return model_path

def model_validation(model_path, data_path):
    print("Starting model validation...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df[['age', 'balance', 'loan_status', 'campaign_calls']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    if accuracy >= 0.85:
        print("Model meets the 85% accuracy threshold. 🎉")
    else:
        print("Model accuracy is below the 85% threshold. 😥")

if __name__ == '__main__':
    data_path = data_preparation()
    model_path = model_training(data_path)
    model_validation(model_path, data_path)