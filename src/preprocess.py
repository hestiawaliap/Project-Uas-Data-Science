
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import config

def load_data(filepath=None):
    """Load dataset Parkinsons"""
    if filepath is None:
        filepath = config.DATASET_PATH
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    return df

def prepare_features(df):
    """Pisahkan fitur dan target"""
    X = df.drop([config.TARGET_COLUMN] + [col for col in config.EXCLUDE_COLUMNS 
                                         if col in df.columns], axis=1)
    y = df[config.TARGET_COLUMN]
    return X, y

def split_data(X, y):
    """Split data menjadi train dan test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """Normalisasi data menggunakan StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved to {config.SCALER_PATH}")
    
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Test fungsi
    print("Testing preprocess module...")
    df = load_data()
    X, y = prepare_features(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
