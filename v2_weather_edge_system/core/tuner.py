import keras_tuner as kt
import numpy as np
import os
from core.hybrid_model import HybridWeatherModel

def run_tuning():
    print("=== Starting Hyperparameter Tuning ===")
    
    # Generate Dummy Dataset (100, 24, 5)
    X_dummy = np.random.rand(100, 24, 5).astype(np.float32)
    # Target dummy: One-hot encoded (3 classes)
    y_dummy_labels = np.random.randint(0, 3, size=(100,))
    y_dummy = np.zeros((100, 3))
    y_dummy[np.arange(100), y_dummy_labels] = 1

    # Split: 70% Train, 15% Val, 15% Test
    X_train, y_train = X_dummy[:70], y_dummy[:70]
    X_val, y_val = X_dummy[70:85], y_dummy[70:85]
    X_test, y_test = X_dummy[85:], y_dummy[85:]
    
    # Clean previous tuning cache to avoid stale errors in simulation
    import shutil
    if os.path.exists('tuning_dir'):
        shutil.rmtree('tuning_dir')
        
    tuner = kt.RandomSearch(
        HybridWeatherModel.build_model,
        objective='val_accuracy',
        max_trials=3,  # Restricted for simulation speed
        executions_per_trial=1,
        directory='tuning_dir',
        project_name='weather_tuner'
    )
    
    tuner.search(X_train, y_train, epochs=2, validation_data=(X_val, y_val), verbose=0)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\n[BEST HYPERPARAMETERS]")
    print(f"LSTM 1 Units: {best_hps.get('lstm1_units')}")
    print(f"LSTM 2 Units: {best_hps.get('lstm2_units')}")
    print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
    print("====================================\n")
