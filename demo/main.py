import os
import sys
import types
if 'imp' not in sys.modules:
    sys.modules['imp'] = types.ModuleType('imp')

import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from core.preprocessor import DataPreprocessor
from database.db_manager import DBManager

def main():
    print("=== Localized Smart Weather Station (DEMO: Real Data) ===")
    
    # 1. Setup DB
    db = DBManager()
    
    # 2. Load TRAINED Model
    model_path = "models/trained_model.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model terlatih tidak ditemukan di {model_path}. Jalankan scripts/train.py dahulu!")
        sys.exit(1)
        
    print("\n[1] Loading Trained HybridWeatherModel...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Continuous Mode using real data stream
    print("\n[2] Starting Continuous Edge Inference (Simulating Real Data Stream)...")
    preprocessor = DataPreprocessor()
    classes = ["Hujan", "Panas", "Cerah"]
    
    # Load actual historical data for simulation stream
    csv_path = "database/historical_surabaya.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Data historis tidak ditemukan di {csv_path}. Jalankan scripts/fetch_data.py dahulu!")
        sys.exit(1)
        
    df_real = pd.read_csv(csv_path)
    # We fit the preprocessor on the whole dataset as done in training
    preprocessor.scale_data(df_real[['suhu', 'kelembaban', 'tekanan', 'angin', 'hujan']], fit=True)
    
    # Start simulating from day 20 onwards
    current_idx = 20 * 24
    
    try:
        while True:
            if current_idx >= len(df_real) - 24:
                # Loop back if we reach the end
                current_idx = 20 * 24
                
            # Take a 24-hour window from real data
            window_df = df_real.iloc[current_idx : current_idx + 24].copy()
            
            # The AutoPreprocessor expects a raw dataframe, we don't aggregate because the API data is already hourly!
            # So we skip aggregate_data and go straight to scale_data.
            features = window_df[['suhu', 'kelembaban', 'tekanan', 'angin', 'hujan']]
            processed_tensor = preprocessor.scale_data(features, fit=False)
            
            # Predict next hour
            prediction_proba = model.predict(processed_tensor, verbose=0)
            pred_class_idx = np.argmax(prediction_proba, axis=1)[0]
            
            # Log it
            latest_reading = features.iloc[-1]
            timestamp = datetime.datetime.now().isoformat()
            
            suhu = float(latest_reading["suhu"])
            kelembaban = float(latest_reading["kelembaban"])
            tekanan = float(latest_reading["tekanan"])
            angin = float(latest_reading["angin"])
            hujan = float(latest_reading["hujan"])
            
            # Save to DB
            db.insert_weather(timestamp, suhu, kelembaban, tekanan, angin, hujan, int(pred_class_idx))
            
            print(f"[{timestamp}] Prediksi: {classes[pred_class_idx]:<5} | Probabilitas: {prediction_proba[0]} | Suhu Asli Terakhir: {suhu} C")
            
            current_idx += 1
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping inference...")
        print("System shutdown.")

if __name__ == "__main__":
    main()
