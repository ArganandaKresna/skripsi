import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

# --- CONFIG ---
DB_FILE = "weather_data.db"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "weather_model.keras")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

# --- 1. GENERATE FAKE DATABASE (7 HARI KEBELAKANG) ---
def generate_db():
    print(f"🛠️  Membuat database simulasi di {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS measurements")
    c.execute('''CREATE TABLE measurements 
                 (timestamp DATETIME, temp REAL, wind_speed REAL, 
                  pressure REAL, humidity REAL, rain_condition INTEGER)''')
    
    # Generate data dari 7 hari lalu sampai sekarang, interval 10 menit
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Buat rentang waktu per 10 menit
    timestamps = pd.date_range(start=start_time, end=end_time, freq='10T')
    n = len(timestamps)
    
    # Buat pola data sinus agar grafik terlihat cantik
    t = np.linspace(0, 100, n)
    temps = 30 + 5 * np.sin(t) + np.random.normal(0, 0.5, n) # Suhu naik turun
    hums = 70 + 10 * np.cos(t) + np.random.normal(0, 1, n)   # Kelembaban
    press = 1010 + 2 * np.sin(t/2)
    winds = np.abs(5 * np.sin(t/3))
    rains = np.random.choice([0, 1], size=n, p=[0.9, 0.1]) # 10% hujan
    
    data_to_insert = []
    for i in range(n):
        data_to_insert.append((
            timestamps[i].to_pydatetime(),
            float(temps[i]),
            float(winds[i]),
            float(press[i]),
            float(hums[i]),
            int(rains[i])
        ))
        
    c.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?)", data_to_insert)
    conn.commit()
    conn.close()
    print(f"✅ Berhasil mengisi {n} data dummy (7 hari).")

# --- 2. GENERATE DUMMY MODEL (JIKA BELUM ADA) ---
def generate_dummy_artifacts():
    # Cek apakah model asli sudah ada?
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("ℹ️  Model asli ditemukan, melewati pembuatan dummy model.")
        return

    print("⚠️  Model asli tidak ditemukan. Membuat DUMMY model untuk testing UI...")
    
    # Buat Scaler Dummy
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # Fit dengan data random
    dummy_data = np.random.rand(100, 5) 
    scaler.fit(dummy_data)
    joblib.dump(scaler, SCALER_PATH)
    
    # Buat Model Keras Dummy (Input: 24x5 -> Output: 5)
    # Kita tidak pakai Time2Vector disini agar simpel, hanya untuk tes UI
    # Dashboard akan tetap jalan tapi prediksinya angka acak
    inputs = tf.keras.Input(shape=(24, 5))
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    outputs = tf.keras.layers.Dense(5)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Simpan dengan format .keras
    model.save(MODEL_PATH)
    print("✅ Dummy Model & Scaler berhasil dibuat.")

if __name__ == "__main__":
    generate_db()
    generate_dummy_artifacts()
    print("\n🎉 Siap! Jalankan: streamlit run dashboard.py")