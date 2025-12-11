import sys
import os
import time
import sqlite3
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go # Kita pakai Plotly agar grafik lebih interaktif

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.model_lib import Time2Vector 

# --- CONFIG ---
DB_PATH = os.path.join(current_dir, "weather_data.db")
MODEL_PATH = os.path.join(current_dir, "artifacts/weather_model.keras")
SCALER_PATH = os.path.join(current_dir, "artifacts/scaler.pkl")
SEQ_LENGTH = 24  # Model butuh 24 'langkah' ke belakang
PREDICT_HOURS = 72 # 3 Hari x 24 Jam

st.set_page_config(page_title="Weather AI 3-Day Forecast", layout="wide")

@st.cache_resource
def load_ai_system():
    if not os.path.exists(MODEL_PATH): return None, None
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'Time2Vector': Time2Vector})
    return model, scaler

def get_history_data():
    """Ambil data 7 hari terakhir"""
    conn = sqlite3.connect(DB_PATH)
    # Query data seminggu terakhir
    query = """
        SELECT timestamp, temp, wind_speed, pressure, humidity, rain_condition 
        FROM measurements 
        WHERE timestamp >= datetime('now', '-7 days')
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # PENTING: Resample data menjadi per JAM (Rata-rata)
        # Agar 1 langkah prediksi AI = 1 Jam
        df_hourly = df.resample('1H').mean().interpolate() # Interpolate isi data bolong
        return df_hourly.dropna()
    return pd.DataFrame()

model, scaler = load_ai_system()

# --- UI ---
st.title("🌦️ 3-Day Autonomous Weather Forecast")

placeholder = st.empty()

while True:
    df = get_history_data()
    
    with placeholder.container():
        if len(df) < SEQ_LENGTH:
            st.warning(f"Menunggu data terkumpul minimal {SEQ_LENGTH} jam... (Saat ini: {len(df)} jam)")
            st.info("Sistem membutuhkan data historis 24 jam pertama untuk mulai memprediksi 3 hari ke depan.")
        else:
            # Data terakhir (Realtime dari DB)
            latest = df.iloc[-1]
            
            # --- MONITORING SAAT INI ---
            st.subheader(f"Kondisi Terkini")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌡️ Suhu", f"{latest['temp']:.1f} °C")
            c2.metric("💧 Kelembaban", f"{latest['humidity']:.1f} %")
            c3.metric("⏲️ Tekanan", f"{latest['pressure']:.0f} hPa")
            c4.metric("💨 Angin", f"{latest['wind_speed']:.1f} m/s")

            # --- PREDIKSI 3 HARI (AUTOREGRESSIVE LOOP) ---
            if model:
                # 1. Siapkan input awal (24 jam terakhir)
                input_df = df.tail(SEQ_LENGTH)[['temp', 'wind_speed', 'pressure', 'humidity', 'rain_condition']]
                input_scaled = scaler.transform(input_df) # Shape (24, 5)
                
                # Bentuk tensor (1, 24, 5)
                current_batch = input_scaled.reshape(1, SEQ_LENGTH, 5)
                
                predictions_scaled = []
                
                # 2. LOOPING 72 kali (3 Hari)
                # Prediksi jam ke-1 -> Masukkan jadi input -> Prediksi jam ke-2 -> dst
                for i in range(PREDICT_HOURS):
                    # Prediksi 1 langkah
                    pred = model.predict(current_batch, verbose=0) # Shape (1, 5)
                    predictions_scaled.append(pred[0])
                    
                    # Update batch input:
                    # Geser window: buang data terlama (index 0), masukkan prediksi baru ke paling belakang
                    # Reshape pred jadi (1, 1, 5) agar bisa diappend
                    pred_reshaped = pred.reshape(1, 1, 5)
                    current_batch = np.append(current_batch[:, 1:, :], pred_reshaped, axis=1)
                
                # 3. Kembalikan ke nilai asli
                predictions_actual = scaler.inverse_transform(np.array(predictions_scaled))
                
                # Buat DataFrame Prediksi
                last_time = df.index[-1]
                future_dates = [last_time + pd.Timedelta(hours=x+1) for x in range(PREDICT_HOURS)]
                
                df_pred = pd.DataFrame(predictions_actual, index=future_dates, columns=input_df.columns)
                
                # --- VISUALISASI ---
                st.divider()
                st.subheader("🔮 Grafik Prediksi 3 Hari ke Depan")
                
                # Kita gabung History (7 hari) + Prediksi (3 hari)
                fig = go.Figure()
                
                # Plot History
                fig.add_trace(go.Scatter(x=df.index, y=df['temp'], name='History (7 Hari)', line=dict(color='gray')))
                
                # Plot Prediksi
                fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['temp'], name='Prediksi (3 Hari)', line=dict(color='red', width=3)))
                
                fig.update_layout(title="Trend Suhu: History vs Forecast", xaxis_title="Waktu", yaxis_title="Suhu (°C)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tampilkan Tabel Ringkasan Harian
                st.subheader("Ringkasan Prakiraan Harian")
                df_pred['Day'] = df_pred.index.date
                daily_summary = df_pred.groupby('Day').agg({
                    'temp': 'mean',
                    'rain_condition': lambda x: (x > 0.5).mean() * 100 # Persentase jam hujan dalam sehari
                }).rename(columns={'temp': 'Rata-rata Suhu', 'rain_condition': 'Peluang Hujan (%)'})
                
                st.table(daily_summary)

    time.sleep(10) # Refresh tiap 10 detik (karena data diresample per jam, tidak perlu terlalu cepat)