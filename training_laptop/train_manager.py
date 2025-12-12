import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Setup path imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.model_lib import create_hybrid_model
from common.preprocessing import WeatherPreprocessor # Import modul baru kita
from common.advanced_theory import calculate_metrics

# --- KONFIGURASI ---
SEQ_LENGTH = 24
FEATURES = ['temp', 'wind_speed', 'pressure', 'humidity', 'rain_condition']
ARTIFACTS_DIR = '../deployment_rpi/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    print("🚀 MEMULAI PROSES TRAINING SKRIPSI")
    
    # ----------------------------------------
    # STEP 1: LOAD & PREPARE DATA
    # ----------------------------------------
    print("\n[Step 1/5] Loading Data...")
    # Ganti ini dengan pd.read_csv('data_asli_kamu.csv')
    # Disini kita generate dummy untuk contoh
    n = 5000
    t = np.linspace(0, 100, n)
    raw_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1h'),
        'temp': 30 + 5*np.sin(t) + np.random.normal(0, 0.5, n),
        'wind_speed': 5 + 2*np.cos(t) + np.random.normal(0, 0.5, n),
        'pressure': 1010 + 10*np.sin(t/2) + np.random.normal(0, 1, n),
        'humidity': 70 + 15*np.cos(t+1) + np.random.normal(0, 2, n),
        'rain_condition': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })
    
    # Inisialisasi Preprocessor
    preprocessor = WeatherPreprocessor(seq_length=SEQ_LENGTH, features=FEATURES)
    
    # Bersihkan Data
    df_clean = preprocessor.clean_and_resample(raw_df)
    print(f"Data shape after cleaning: {df_clean.shape}")

    # ----------------------------------------
    # STEP 2: SCALING & SEQUENCING
    # ----------------------------------------
    print("\n[Step 2/5] Preprocessing (Scaling & Sequencing)...")
    # Scaling (-1 s/d 1 karena kita pakai Tanh)
    scaled_data = preprocessor.fit_transform(df_clean)
    
    # Simpan Scaler untuk RPi
    preprocessor.save_scaler(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    
    # Buat Sequence (X: Input 24 jam, y: Target jam ke-25)
    X, y = preprocessor.create_sequences(scaled_data)
    
    # Split Train/Test (80% / 20%)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")

    # ----------------------------------------
    # STEP 3: BUILD MODEL (TANH)
    # ----------------------------------------
    print("\n[Step 3/5] Building Hybrid Model (Tanh Activation)...")
    model = create_hybrid_model(
        input_shape=(SEQ_LENGTH, len(FEATURES)), 
        output_shape=len(FEATURES), 
        seq_len=SEQ_LENGTH
    )
    
    # DEFINISI OPTIMIZER ADAM
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()

    # ----------------------------------------
    # STEP 4: TRAINING
    # ----------------------------------------
    print("\n[Step 4/5] Training Process...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15, # Sesuaikan kebutuhan
        batch_size=32,
        verbose=1
    )
    
    # Simpan Model
    model.save(os.path.join(ARTIFACTS_DIR, 'weather_model.keras'))
    print("✅ Model saved successfully.")

    # ----------------------------------------
    # STEP 5: EVALUATION
    # ----------------------------------------
    print("\n[Step 5/5] Final Evaluation...")
    # Prediksi
    y_pred_scaled = model.predict(X_test)
    
    # Kembalikan ke nilai asli menggunakan preprocessor
    y_pred = preprocessor.inverse_transform(y_pred_scaled)
    y_true = preprocessor.inverse_transform(y_test)
    
    # Hitung Metrik
    results = []
    for i, feat in enumerate(FEATURES):
        met = calculate_metrics(y_true[:, i], y_pred[:, i])
        met['Feature'] = feat
        results.append(met)
        
    res_df = pd.DataFrame(results).set_index('Feature')
    print("\n=== HASIL EVALUASI SKRIPSI ===")
    print(res_df)
    res_df.to_csv("hasil_evaluasi_terbaru.csv")

if __name__ == "__main__":
    main()