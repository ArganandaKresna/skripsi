import sqlite3
import pandas as pd
import streamlit as st
import numpy as np
import datetime
from core.inference import WeatherInference

st.set_page_config(page_title="Weather Edge Dashboard", layout="wide")

st.title("🌤️ Edge Computing Weather Dashboard")
st.markdown("Real-time telemetry and hyperlocal weather classification from Edge Device.")

# Connect to database
DB_PATH = "database/weather_data.db"
LABEL_MAP = {0: "Cerah", 1: "Panas", 2: "Hujan"}

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
        st.error(f"Error loading database: {e}")
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
    st.header("Simulasi Prediksi 24 Jam ke Depan")
    st.markdown("Wilayah: **Surabaya** (Karakteristik: Suhu Panas 31-36°C, Kelembaban Sedang-Rendah)")
    
    if st.button("Jalankan Prediksi 24 Jam"):
        with st.spinner("Memuat Model Edge dan Mengolah Data..."):
            inference = WeatherInference()
            
            forecast_results = []
            current_time = datetime.datetime.now()
            
            # Generate Base Surabaya Dummy Data (1, 288, 5) float32
            # Features shape logic: we can just manipulate the last timestep or mean for visualization
            
            for hour in range(1, 25):
                # Generate custom tensor for Surabaya:
                # Feature scaling in main.py (for reference):
                # suhu = data[0] * 40
                # kelembaban = data[1] * 100
                # tekanan = data[2] * 50 + 980
                # angin = data[3] * 20
                # hujan = data[4] * 50
                
                base_data = np.random.rand(1, 288, 5).astype(np.float32)
                
                # Force Surabaya characteristics on the input tensor
                base_data[:, :, 0] = np.random.uniform(0.77, 0.90, (1, 288)) # Suhu: 31 - 36 C
                base_data[:, :, 1] = np.random.uniform(0.40, 0.60, (1, 288)) # Kelembaban: 40 - 60%
                base_data[:, :, 2] = np.random.uniform(0.40, 0.60, (1, 288)) # Tekanan normal
                base_data[:, :, 3] = np.random.uniform(0.10, 0.30, (1, 288)) # Angin ringan
                base_data[:, :, 4] = np.random.uniform(0.00, 0.05, (1, 288)) # Hujan hampir nol
                
                # Predict
                pred_class = inference.predict(base_data)
                pred_label = LABEL_MAP.get(pred_class, "Unknown")
                
                # Extract simulated values for display
                latest_reading = base_data[0, -1, :]
                suhu = float(latest_reading[0] * 40)
                kelembaban = float(latest_reading[1] * 100)
                
                forecast_time = current_time + datetime.timedelta(hours=hour)
                
                forecast_results.append({
                    "Jam": forecast_time.strftime("%H:00"),
                    "Prediksi": pred_label,
                    "Suhu (°C)": round(suhu, 1),
                    "Kelembaban (%)": round(kelembaban, 1)
                })
            
            # Display Results
            forecast_df = pd.DataFrame(forecast_results)
            
            st.success("Prediksi berhasil diselesaikan!")
            
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                st.subheader("Tren Suhu 24 Jam Ke Depan")
                # Using line chart with temperature
                st.line_chart(forecast_df.set_index("Jam")["Suhu (°C)"])
                
            with col_table:
                st.subheader("Detail Prediksi")
                st.dataframe(forecast_df, use_container_width=True)
