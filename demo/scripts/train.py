import os
import sys
import types
if 'imp' not in sys.modules:
    sys.modules['imp'] = types.ModuleType('imp')

import pandas as pd
import numpy as np
import tensorflow as tf
from core.hybrid_model import HybridWeatherModel
from core.preprocessor import DataPreprocessor

def create_sliding_windows(data, labels, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

def train_model():
    print("=== Memulai Pelatihan Model Hybrid LSTM-Transformer ===")
    
    csv_path = "database/historical_surabaya.csv"
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan! Harap jalankan fetch_data.py terlebih dahulu.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Labeling Logic
    # 0 = Hujan, 1 = Panas, 2 = Cerah
    print("Memproses pelabelan ground truth...")
    labels = []
    for idx, row in df.iterrows():
        if row['hujan'] > 0.1:
            labels.append(0)
        elif row['suhu'] > 33.0:
            labels.append(1)
        else:
            labels.append(2)
            
    df['label'] = labels
    
    features = df[['suhu', 'kelembaban', 'tekanan', 'angin', 'hujan']]
    
    print("Mempersiapkan DataPreprocessor...")
    preprocessor = DataPreprocessor()
    # We fit the scaler on the entire real historical data!
    scaled_features = preprocessor.scale_data(features, fit=True)
    
    # The output of scale_data is (1, N, 5). We need to reshape it to (N, 5)
    scaled_features = scaled_features.reshape(-1, 5)
    
    print("Membuat sliding windows (24 jam)...")
    X, y = create_sliding_windows(scaled_features, df['label'].values, window_size=24)
    
    # One-hot encode labels
    y_cat = tf.keras.utils.to_categorical(y, num_classes=3)
    
    # Split Train / Test (80% / 20%)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y_cat[:split_idx]
    X_test, y_test = X[split_idx:], y_cat[split_idx:]
    
    print(f"Distribusi Data Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Compile Model
    model = HybridWeatherModel.build_model()
    
    print("Memulai proses training (Epochs=10)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nEvaluasi Model pada Data Uji Asli Surabaya -> Akurasi: {acc*100:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    model.save("models/trained_model.keras")
    print("Model terlatih telah disimpan di models/trained_model.keras")

if __name__ == "__main__":
    train_model()
