
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import config
import os

def load_models():
    """Load all trained models"""
    models = {}
    
    # All paths will be joined with config.BASE_PATH
    # No need for current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        scaler_path = os.path.join(config.BASE_PATH, config.SCALER_PATH)
        models['scaler'] = joblib.load(scaler_path)
        print("✓ Scaler loaded")
    except Exception as e:
        print(f"✗ Scaler not found: {e}")
        return None
    
    try:
        rf_path = os.path.join(config.BASE_PATH, config.RF_MODEL_PATH)
        models['rf'] = joblib.load(rf_path)
        print("✓ Random Forest loaded")
    except Exception as e:
        print(f"✗ Random Forest not found: {e}")
        models['rf'] = None
    
    try:
        svm_path = os.path.join(config.BASE_PATH, config.SVM_MODEL_PATH)
        models['svm'] = joblib.load(svm_path)
        print("✓ SVM loaded")
    except Exception as e:
        print(f"✗ SVM not found: {e}")
        models['svm'] = None
    
    try:
        dl_path = os.path.join(config.BASE_PATH, config.DL_MODEL_PATH)
        models['dl'] = keras.models.load_model(dl_path)
        print("✓ Deep Learning loaded")
    except Exception as e:
        print(f"✗ Deep Learning not found: {e}")
        models['dl'] = None
    
    return models

def predict_single(models, input_data):
    """Make prediction for single sample"""
    if models is None or 'scaler' not in models or models['scaler'] is None:
        return {"error": "Models or scaler not loaded"}
    
    scaled_data = models['scaler'].transform([input_data])
    
    predictions = {}
    
    if models['rf'] is not None:
        rf_pred = models['rf'].predict(scaled_data)[0]
        rf_prob = models['rf'].predict_proba(scaled_data)[0][1]
        predictions['Random Forest'] = {
            'prediction': int(rf_pred),
            'probability': float(rf_prob),
            'label': 'Parkinson' if rf_pred == 1 else 'Healthy'
        }
    
    if models['svm'] is not None:
        svm_pred = models['svm'].predict(scaled_data)[0]
        svm_prob = models['svm'].predict_proba(scaled_data)[0][1]
        predictions['SVM'] = {
            'prediction': int(svm_pred),
            'probability': float(svm_prob),
            'label': 'Parkinson' if svm_pred == 1 else 'Healthy'
        }
    
    if models['dl'] is not None:
        dl_prob = models['dl'].predict(scaled_data, verbose=0)[0][0]
        dl_pred = 1 if dl_prob > 0.5 else 0
        predictions['Deep Learning'] = {
            'prediction': int(dl_pred),
            'probability': float(dl_prob),
            'label': 'Parkinson' if dl_pred == 1 else 'Healthy'
        }
    
    return predictions

def print_predictions(predictions):
    """Print prediction results"""
    print("
" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    for model_name, result in predictions.items():
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue
        
        status = "🟢 HEALTHY" if result['prediction'] == 0 else "🔴 PARKINSON"
        prob_percent = result['probability'] * 100
        
        print(f"
{model_name}:")
        print(f"  Status: {status}")
        print(f"  Confidence: {prob_percent:.1f}%")

if __name__ == "__main__":
    print("Testing predict module...")
    
    sample_data = [
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554,
        0.01109, 0.04374, 0.426, 0.02182, 0.03130, 0.02971, 0.06545,
        0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482,
        2.301442, 0.284654
    ]
    
    print(f"Sample data length: {len(sample_data)}")
    print("Note: Models need to be trained first for actual prediction")
