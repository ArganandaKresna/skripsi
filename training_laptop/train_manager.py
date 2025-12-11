import sys
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Setup path agar bisa baca folder common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.model_lib import create_hybrid_model
from common.advanced_theory import wavelet_denoising, calculate_metrics

# --- KONFIGURASI ---
SEQ_LENGTH = 24  # Window size (misal 24 jam ke belakang)
FEATURES = ['temp', 'wind_speed', 'pressure', 'humidity', 'rain_condition']
ARTIFACTS_DIR = '../deployment_rpi/artifacts' # Simpan langsung ke folder rpi (jika satu PC) atau folder lokal
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 1. GENERATE DATA DUMMY (Ganti dengan pd.read_csv datamu)
print("=== Generating Dummy Data ===")
n = 5000
t = np.linspace(0, 100, n)
df = pd.DataFrame({
    'temp': 30 + 5*np.sin(t) + np.random.normal(0, 0.5, n),
    'wind_speed': 5 + 2*np.cos(t) + np.random.normal(0, 0.5, n),
    'pressure': 1010 + 10*np.sin(t/2) + np.random.normal(0, 1, n),
    'humidity': 70 + 15*np.cos(t+1) + np.random.normal(0, 2, n),
    'rain_condition': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
})

# 2. WAVELET DENOISING (Eksperimen untuk Bab 4)
print("=== Applying Wavelet Denoising ===")
df['temp_clean'] = wavelet_denoising(df['temp'], 'db8', 2)
# Gunakan data clean untuk training (opsional, disini kita pakai raw untuk demo)

# 3. PREPROCESSING
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[FEATURES])

# Simpan Scaler
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split Train/Test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. TRAINING
print(f"=== Training Model (Input Shape: {X_train.shape}) ===")
model = create_hybrid_model((SEQ_LENGTH, 5), 5, SEQ_LENGTH)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15, 
    batch_size=32,
    verbose=1
)

# Simpan Model
model_path = os.path.join(ARTIFACTS_DIR, 'weather_model.keras')
model.save(model_path)
print(f"Model saved to: {model_path}")

# 5. EVALUASI SKRIPSI
print("\n=== Evaluasi Model (Test Set) ===")
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

results = []
for i, feat in enumerate(FEATURES):
    met = calculate_metrics(y_true[:, i], y_pred[:, i])
    met['Feature'] = feat
    results.append(met)

res_df = pd.DataFrame(results).set_index('Feature')
print(res_df)
res_df.to_csv("hasil_evaluasi_skripsi.csv")