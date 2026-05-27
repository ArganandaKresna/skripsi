import os
import sys
import types
if 'imp' not in sys.modules:
    sys.modules['imp'] = types.ModuleType('imp')

import urllib.request
import json
import time
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf

from core.preprocessor import DataPreprocessor
from database.db_manager import DBManager

def get_live_data():
    lat = -7.2504
    lon = 112.7688
    # Fetch past 24 hours up to today using Open-Meteo's Live Forecast API
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_hours=24&forecast_days=1&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation&timezone=Asia%2FJakarta"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    hourly = data['hourly']
    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "suhu": hourly["temperature_2m"],
        "kelembaban": hourly["relative_humidity_2m"],
        "tekanan": hourly["surface_pressure"],
        "angin": hourly["wind_speed_10m"],
        "hujan": hourly["precipitation"]
    })
    
    # Get current time rounded down to the current hour
    now = datetime.datetime.now()
    current_hour_str = now.strftime("%Y-%m-%dT%H:00")
    
    # Filter to get exactly 24 hours leading up to the CURRENT hour
    df = df[df['timestamp'] <= current_hour_str].tail(24)
    
    # Fill any NaNs
    df = df.ffill().bfill()
    return df

def main():
    print("=== LIVE INFERENCE: Real-time Surabaya Weather ===")
    db = DBManager()
    
    model_path = "models/trained_model.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} tidak ditemukan! Pastikan telah menjalankan scripts/train.py")
        sys.exit(1)
        
    print("\n[1] Memuat Model Terlatih...")
    model = tf.keras.models.load_model(model_path)
    preprocessor = DataPreprocessor()
    classes = ["Hujan", "Panas", "Cerah"]
    
    # Ensure scaling mechanism strictly follows the historical training distribution
    csv_path = "database/historical_surabaya.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Basis data historis di {csv_path} hilang.")
        sys.exit(1)
        
    df_real = pd.read_csv(csv_path)
    preprocessor.scale_data(df_real[['suhu', 'kelembaban', 'tekanan', 'angin', 'hujan']], fit=True)
    
    print("\n[2] Memulai Pembacaan Satelit Waktu Nyata...")
    try:
        while True:
            try:
                # 1. Fetch
                df_live = get_live_data()
                if len(df_live) < 24:
                    print("Data tidak lengkap (kurang dari 24 jam). Mengulang...")
                    time.sleep(10)
                    continue
                
                # 2. Extract & Preprocess
                features = df_live[['suhu', 'kelembaban', 'tekanan', 'angin', 'hujan']]
                processed_tensor = preprocessor.scale_data(features, fit=False)
                
                # 3. Predict 
                prediction_proba = model.predict(processed_tensor, verbose=0)
                pred_class_idx = np.argmax(prediction_proba, axis=1)[0]
                
                # 4. Store
                latest_reading = features.iloc[-1]
                timestamp_live = df_live.iloc[-1]["timestamp"]
                
                suhu = float(latest_reading["suhu"])
                kelembaban = float(latest_reading["kelembaban"])
                tekanan = float(latest_reading["tekanan"])
                angin = float(latest_reading["angin"])
                hujan = float(latest_reading["hujan"])
                
                # Masukkan ke SQLite agar Streamlit bisa membaca Real-Time
                db.insert_weather(timestamp_live, suhu, kelembaban, tekanan, angin, hujan, int(pred_class_idx))
                
                # Terminal output
                print(f"[{timestamp_live}] SUHU SAAT INI: {suhu}°C | PREDIKSI 1 JAM KEDEPAN: {classes[pred_class_idx]:<5} | Prob: {prediction_proba[0]}")
                
            except Exception as e:
                print(f"Gagal mengambil data satelit: {e}")
            
            # Wait 30 seconds before pulling again (for demo, real implementation could be 15-60 minutes)
            print("Menunggu update satelit...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nSistem Ingerensi Live Dihentikan.")

if __name__ == "__main__":
    main()
