import sqlite3
import pandas as pd
import streamlit as st
import numpy as np
import datetime
from core.hybrid_model import HybridWeatherModel
from core.preprocessor import DataPreprocessor

st.set_page_config(page_title="Weather Edge Dashboard v2", layout="wide")

st.title("🌤️ Edge Computing Weather Dashboard (v2)")
st.markdown("Real-time telemetry and hyperlocal weather classification (LSTM-Transformer).")

# Connect to database
DB_PATH = "database/weather_data.db"
# Target names from v2 evaluator: ["Hujan", "Panas", "Cerah"]
LABEL_MAP = {0: "Hujan", 1: "Panas", 2: "Cerah"}

def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM weather ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        
        if not df.empty:
            df["prediksi_label"] = df["prediksi"].map(LABEL_MAP)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        return pd.DataFrame()

# Create Tabs
tab_realtime, tab_forecast = st.tabs(["🔴 Real-time Edge Telemetry", "🔮 Forecast 24-Jam (Surabaya)"])

with tab_realtime:
    st.header("Live Sensor Data & Predictions")
    df = load_data()
    
    if df.empty:
        st.warning("Menunggu data dari sistem inference edge (main.py)...")
    else:
        # Top metrics
        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediksi Saat Ini", latest["prediksi_label"])
        col2.metric("Suhu", f"{latest['suhu']:.1f} °C")
        col3.metric("Kelembaban", f"{latest['kelembaban']:.1f} %")
        col4.metric("Tekanan", f"{latest['tekanan']:.1f} hPa")
        
        st.markdown("---")
        
        # Charts
        st.subheader("📈 Tren Sensor Terakhir")
        chart_data = df[["timestamp", "suhu", "kelembaban", "angin", "hujan"]].set_index("timestamp")
        st.line_chart(chart_data)
        
        # Raw Data
        st.subheader("📋 Data Log Terbaru")
        st.dataframe(df[["timestamp", "suhu", "kelembaban", "tekanan", "angin", "hujan", "prediksi_label"]].sort_values("timestamp", ascending=False).head(10))

with tab_forecast:
    st.header("Forecast 24 Jam ke Depan (Live API Open-Meteo)")
    st.markdown("Menarik ramalan cuaca Surabaya langsung dari satelit, lalu diklasifikasikan oleh Hybrid LSTM-Transformer.")
    
    if st.button("Tarik Data & Jalankan Prediksi"):
        with st.spinner("Mengunduh data satelit & Menjalankan Model..."):
            try:
                import urllib.request
                import json
                import tensorflow as tf
                
                lat, lon = -7.2504, 112.7688
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_hours=24&forecast_hours=48&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation&timezone=Asia%2FJakarta"
                
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode())
                
                hourly = data['hourly']
                df_api = pd.DataFrame({
                    "timestamp": hourly["time"],
                    "suhu": hourly["temperature_2m"],
                    "kelembaban": hourly["relative_humidity_2m"],
                    "tekanan": hourly["surface_pressure"],
                    "angin": hourly["wind_speed_10m"],
                    "hujan": hourly["precipitation"]
                }).ffill().bfill()
                
                now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:00")
                start_idx = 24
                for idx, row in df_api.iterrows():
                    if row['timestamp'] == now_str:
                        start_idx = idx
                        break
                
                model = tf.keras.models.load_model("models/trained_model.keras")
                preprocessor = DataPreprocessor()
                df_real = pd.read_csv("database/historical_surabaya.csv")
                preprocessor.scale_data(df_real[['suhu', 'kelembaban', 'tekanan', 'angin', 'hujan']], fit=True)
                
                forecast_results = []
                
                for i in range(1, 25):
                    if start_idx + i >= len(df_api):
                        break
                        
                    # Ambil 24 jam sebelum target waktu prediksi
                    window = df_api.iloc[start_idx + i - 24 : start_idx + i].copy()
                    
                    features = window[['suhu', 'kelembaban', 'tekanan', 'angin', 'hujan']]
                    processed_tensor = preprocessor.scale_data(features, fit=False)
                    
                    prediction = model.predict(processed_tensor, verbose=0)
                    pred_class = int(np.argmax(prediction, axis=1)[0])
                    pred_label = LABEL_MAP.get(pred_class, "Unknown")
                    
                    target_row = df_api.iloc[start_idx + i]
                    
                    forecast_results.append({
                        "Waktu": target_row["timestamp"],
                        "Prediksi Model": pred_label,
                        "Suhu (°C)": target_row["suhu"],
                        "Kelembaban (%)": target_row["kelembaban"],
                        "Hujan (mm)": target_row["hujan"]
                    })
                
                forecast_df = pd.DataFrame(forecast_results)
                st.success("Prediksi sukses menggunakan data Live Open-Meteo!")
                
                col_chart, col_table = st.columns([2, 1])
                with col_chart:
                    st.subheader("Tren Suhu 24 Jam Ke Depan")
                    st.line_chart(forecast_df.set_index("Waktu")["Suhu (°C)"])
                    
                with col_table:
                    st.subheader("Detail Prediksi")
                    st.dataframe(forecast_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Gagal menarik data atau memprediksi: {e}")
