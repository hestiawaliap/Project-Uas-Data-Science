
# Configuration file for Parkinson Prediction System

# Base path for the project
BASE_PATH = '/content/'

# Dataset settings
DATASET_PATH = 'data/parkinsons.data'
TARGET_COLUMN = 'status'
EXCLUDE_COLUMNS = ['name']

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Random Forest settings
RF_N_ESTIMATORS = 100

# SVM settings
SVM_KERNEL = 'rbf'
SVM_PROBABILITY = True

# Deep Learning settings
DL_EPOPOCHS = 100
DL_BATCH_SIZE = 16
DL_LEARNING_RATE = 0.001
DL_DROPOUT_RATE = 0.3

# Paths
SCALER_PATH = 'models/scaler.pkl'
RF_MODEL_PATH = 'models/rf_model.pkl'
SVM_MODEL_PATH = 'models/svm_model.pkl'
DL_MODEL_PATH = 'models/dl_model.h5'
RESULTS_PATH = 'data/results.csv'
