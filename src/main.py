
import argparse
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src')) # Ensure src is in path for module imports

from preprocess import load_data, prepare_features, split_data, normalize_data
from train import train_all_models
from evaluate import evaluate_model, compare_models, plot_comparison
from predict import load_models, predict_single, print_predictions
from utils import print_dataset_info, save_results
import config # Import config to access BASE_PATH

def run_pipeline():
    """Run complete pipeline"""
    print("STARTING PARKINSON PREDICTION PIPELINE")
    print("="*60)
    
    print("
STEP 1: Loading data...")
    df = load_data() # preprocess.load_data() will resolve path internally
    print_dataset_info(df)
    
    print("
STEP 2: Preparing features...")
    X, y = prepare_features(df)
    
    print("
STEP 3: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("
STEP 4: Normalizing data...")
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    
    print("
STEP 5: Training models...")
    rf_model, svm_model, dl_model, history = train_all_models(X_train_scaled, y_train)
    
    print("
STEP 6: Evaluating models...")
    models_dict = {
        'Random Forest': rf_model,
        'SVM': svm_model,
        'Deep Learning': dl_model
    }
    
    results_list = []
    for name, model in models_dict.items():
        result = evaluate_model(model, X_test_scaled, y_test, name)
        results_list.append(result)
    
    print("
" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison_df = compare_models(results_list)
    print(comparison_df.to_string(index=False))
    
    save_results(comparison_df)
    
    plot_comparison(comparison_df)
    
    print("
PIPELINE COMPLETED!")

def run_prediction():
    """Run prediction on sample data"""
    print("MAKING PREDICTION")
    print("="*60)
    
    models = load_models() # predict.load_models() will resolve paths internally
    if models is None:
        print("Please train models first!")
        return
    
    sample_data = [
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554,
        0.01109, 0.04374, 0.426, 0.02182, 0.03130, 0.02971, 0.06545,
        0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482,
        2.301442, 0.284654
    ]
    
    predictions = predict_single(models, sample_data)
    print_predictions(predictions)

def main():
    parser = argparse.ArgumentParser(description='Parkinson Prediction System')
    parser.add_argument('--mode', choices=['pipeline', 'predict', 'train'], 
                       default='pipeline', help='Mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'pipeline':
        run_pipeline()
    elif args.mode == 'predict':
        run_prediction()
    elif args.mode == 'train':
        df = load_data()
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, _ = normalize_data(X_train, X_test)
        train_all_models(X_train_scaled, y_train)
        print("
✅ Models trained!")

if __name__ == "__main__":
    main()
