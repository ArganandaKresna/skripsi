import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join('..')))
from common.advanced_theory import calculate_metrics, apply_ceemdan

# Load Artifacts
MODEL_PATH = '../deployment_rpi/artifacts/weather_model.keras'
SCALER_PATH = '../deployment_rpi/artifacts/scaler.pkl'
FEATURES = ['temp', 'wind_speed', 'pressure', 'humidity', 'rain_condition']

print("Loading Model & Data...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- NOTE: Disini harus load data test X_test, y_test yang sama dengan notebook ---
# Untuk simulasi file ini standalone, kita generate ulang (ideally save np array di notebook)
# (Kode generate dummy data disingkat disini, asumsikan X_test, y_test tersedia)
# ... [Insert code to load/generate X_test] ...
# ---

# Prediksi
print("Melakukan Inference pada Data Uji...")
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# Evaluasi Per Parameter
print("\n=== HASIL EVALUASI MODEL (METRIK) ===")
results = []
for i, feature in enumerate(FEATURES):
    metrics = calculate_metrics(y_true[:, i], y_pred[:, i])
    metrics['Feature'] = feature
    results.append(metrics)

results_df = pd.DataFrame(results).set_index('Feature')
print(results_df)

# Export ke CSV untuk Skripsi
results_df.to_csv("hasil_evaluasi_skripsi.csv")

# Contoh Analisis CEEMDAN pada Error (Residual)
# Analisis sisa error menggunakan CEEMDAN untuk melihat apakah errornya acak atau berpola
error_temp = y_true[:, 0] - y_pred[:, 0] # Error pada temperatur
imfs, residue = apply_ceemdan(error_temp[:200]) # Ambil sampel 200 data

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(error_temp[:200], 'k')
plt.title("Residual Error (Temperatur)")
plt.subplot(2, 1, 2)
plt.plot(imfs[0], 'r')
plt.title("IMF 1 dari Error (High Frequency Noise)")
plt.tight_layout()
plt.show()