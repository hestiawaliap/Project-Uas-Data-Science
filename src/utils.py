
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import config

def print_dataset_info(df):
    """Print dataset information"""
    print("📊 DATASET INFORMATION")
    print("="*40)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"
Target distribution:")
    print(df['status'].value_counts())
    print(f"
Missing values: {df.isnull().sum().sum()}")
    print("="*40)

def plot_feature_distributions(df, features, n_cols=4):
    """Plot distributions of features"""
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            df[feature].hist(ax=axes[i], bins=30)
            axes[i].set_title(feature)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout()
    plt.show()

def save_results(results_df, filename='model_results.csv'):
    """Save results to CSV"""
    results_save_path = os.path.join(config.BASE_PATH, config.RESULTS_PATH)
    results_df.to_csv(results_save_path, index=False)
    print(f"Results saved to {results_save_path}")

def load_results(filename='model_results.csv'):
    """Load results from CSV"""
    try:
        results_load_path = os.path.join(config.BASE_PATH, config.RESULTS_PATH)
        results_df = pd.read_csv(results_load_path)
        print(f"Results loaded from {results_load_path}")
        return results_df
    except Exception as e:
        print(f"File {os.path.join(config.BASE_PATH, config.RESULTS_PATH)} not found: {e}")
        return None

if __name__ == "__main__":
    print("Utils module loaded successfully!")
