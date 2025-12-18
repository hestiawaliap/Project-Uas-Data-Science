
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import config

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    rf_model_save_path = os.path.join(config.BASE_PATH, config.RF_MODEL_PATH)
    joblib.dump(model, rf_model_save_path)
    print(f"Random Forest model saved to {rf_model_save_path}")

    return model

def train_svm(X_train, y_train):
    """Train SVM model"""
    print("Training SVM...")
    model = SVC(
        kernel=config.SVM_KERNEL,
        probability=config.SVM_PROBABILITY,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    svm_model_save_path = os.path.join(config.BASE_PATH, config.SVM_MODEL_PATH)
    joblib.dump(model, svm_model_save_path)
    print(f"SVM model saved to {svm_model_save_path}")

    return model

def train_deep_learning(X_train, y_train):
    """Train Deep Learning model"""
    print("Training Deep Learning model...")

    model = keras.Sequential([
        layers.Dense(128, activation='relu',
                    input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(config.DL_DROPOUT_RATE),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config.DL_DROPOUT_RATE / 2),

        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.DL_LEARNING_RATE
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE
    )

    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=config.DL_EPOCHS,
        batch_size=config.DL_BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    dl_model_save_path = os.path.join(config.BASE_PATH, config.DL_MODEL_PATH)
    model.save(dl_model_save_path)
    print(f"Deep Learning model saved to {dl_model_save_path}")

    return model, history

def train_all_models(X_train, y_train):
    """Train all three models"""
    print("
" + "="*50)
    print("TRAINING ALL MODELS")
    print("="*50)

    models_dir = os.path.join(config.BASE_PATH, 'models') # Assuming 'models' is always directly under base path
    os.makedirs(models_dir, exist_ok=True)

    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    dl_model, history = train_deep_learning(X_train, y_train)

    print("
✅ All models trained and saved!")
    return rf_model, svm_model, dl_model, history

if __name__ == "__main__":
    print("Testing train module...")
    X_dummy = np.random.randn(100, 20)
    y_dummy = np.random.randint(0, 2, 100)

    print("Creating dummy models...")
    print("Module ready!")
