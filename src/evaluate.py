
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import joblib
import tensorflow as tf
from tensorflow import keras
import config
import os

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model"""
    print(f"Evaluating {model_name}...")

    if model_name in ['Random Forest', 'SVM']:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    else:  # Deep Learning
        y_pred_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_test, y_pred)

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

    return results

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_models(results_list):
    """Compare all models"""
    comparison_data = []

    for result in results_list:
        comparison_data.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'AUC': result['roc_auc']
        })

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def plot_comparison(comparison_df):
    """Plot model comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        ax.bar(comparison_df['Model'], comparison_df[metric])
        ax.set_title(metric)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        for j, v in enumerate(comparison_df[metric]):
            ax.text(j, v + 0.02, f'{v:.3f}', ha='center')

    plt.suptitle('Model Performance Comparison')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Evaluate module loaded successfully!")
