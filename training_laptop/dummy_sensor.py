import sys
import os
import time
import sqlite3
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf

# --- SETUP PATH MODULE ---
# Agar RPi bisa membaca folder 'common' yang ada di level atasnya
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Library Sendiri
from common.model_lib import Time2Vector 

# --- KONFIGURASI ---
DB_PATH = os.path.join(current_dir, "weather_data.db")
MODEL_PATH = os.path.join(current_dir, "artifacts/weather_model.keras")
SCALER_PATH = os.path.join(current_dir, "artifacts/scaler.pkl")
SEQ_LENGTH = 24 # Wajib sama dengan saat training (misal 24 data terakhir)

# --- SETUP STREAMLIT ---
st.set_page_config(
    page_title="Weather AI System",
    page_icon="🌦️",
    layout="wide"
)

# Custom CSS untuk tampilan Skripsi yang rapi
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;}
    .stAlert {margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# --- FUNGSI UTAMA ---

@st.cache_resource
def load_ai_system():
    """Load Model dan Scaler sekali saja saat start"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    
    try:
        scaler = joblib.load(SCALER_PATH)
        # Load model dengan memberitahu Keras tentang Custom Layer kita
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={'Time2Vector': Time2Vector}
        )
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

def get_latest_data(limit=100):
    """Ambil data dari SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Ambil data rain_condition sebagai angka (0/1)
        query = f"""
            SELECT timestamp, temp, wind_speed, pressure, humidity, rain_condition 
            FROM measurements 
            ORDER BY timestamp DESC LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Konversi timestamp ke datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Urutkan dari terlama ke terbaru (Ascending) untuk plotting & AI
        return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# --- LOAD RESOURCES ---
model, scaler = load_ai_system()

# --- UI LAYOUT ---
st.title("🌦️ Skripsi: Autonomous Weather Station with LSTM-Transformer")
st.markdown("Sistem monitoring cuaca hiperlokal dengan prediksi berbasis *Deep Learning*.")

placeholder = st.empty()

while True:
    # 1. Fetch Data
    df = get_latest_data(limit=SEQ_LENGTH + 20) # Ambil buffer lebih
    
    with placeholder.container():
        if df.empty:
            st.warning("⚠️ Belum ada data di Database. Jalankan 'collector.py' dan nyalakan sensor.")
        else:
            latest = df.iloc[-1]
            
            # --- SECTION 1: REALTIME MONITORING ---
            st.subheader(f"📍 Kondisi Terkini ({latest['timestamp'].strftime('%H:%M:%S')})")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🌡️ Temperatur", f"{latest['temp']:.1f} °C")
            c2.metric("💧 Kelembaban", f"{latest['humidity']:.1f} %")
            c3.metric("⏲️ Tekanan", f"{latest['pressure']:.1f} hPa")
            c4.metric("💨 Angin", f"{latest['wind_speed']:.1f} m/s")
            
            rain_status = "Hujan" if latest['rain_condition'] > 0.5 else "Cerah"
            c5.metric("🌧️ Kondisi", rain_status)
            
            st.divider()

            # --- SECTION 2: AI INFERENCE ---
            st.subheader("🔮 Prediksi Cuaca (Next Step Inference)")
            
            # Cek kecukupan data
            if len(df) < SEQ_LENGTH:
                st.info(f"Mengumpulkan data untuk prediksi AI... ({len(df)}/{SEQ_LENGTH})")
                st.progress(len(df)/SEQ_LENGTH)
            elif model is not None:
                # Siapkan Input
                input_df = df.tail(SEQ_LENGTH)[['temp', 'wind_speed', 'pressure', 'humidity', 'rain_condition']]
                input_scaled = scaler.transform(input_df)
                
                # Reshape: (1, 24, 5)
                input_tensor = input_scaled.reshape(1, SEQ_LENGTH, 5)
                
                # Predict
                pred_scaled = model.predict(input_tensor, verbose=0)
                pred_actual = scaler.inverse_transform(pred_scaled)[0]
                
                # Tampilkan Hasil
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.success(f"**Prediksi Suhu:** {pred_actual[0]:.2f} °C")
                    st.info(f"**Prediksi Kelembaban:** {pred_actual[3]:.2f} %")
                    
                    chance_rain = np.clip(pred_actual[4], 0, 1) * 100
                    status_pred = "Hujan" if chance_rain > 50 else "Cerah"
                    st.warning(f"**Potensi Hujan:** {chance_rain:.1f}% ({status_pred})")

                with col_res2:
                    # Gabungkan data history terakhir + prediksi untuk grafik
                    # Buat index waktu baru untuk prediksi
                    last_time = df['timestamp'].iloc[-1]
                    next_time = last_time + pd.Timedelta(hours=1) # Asumsi interval 1 jam
                    
                    chart_data = pd.DataFrame({
                        'Waktu': [last_time, next_time],
                        'Suhu': [latest['temp'], pred_actual[0]],
                        'Tipe': ['Aktual', 'Prediksi']
                    })
                    st.caption("Perbandingan Aktual vs Prediksi")
                    st.dataframe(chart_data) # Bisa diganti st.line_chart jika history panjang

            # --- SECTION 3: GRAFIK HISTORIS ---
            st.subheader("📈 Tren Data Historis")
            st.line_chart(df.set_index('timestamp')[['temp', 'humidity', 'pressure']])

    # Auto Refresh rate
    time.sleep(5)