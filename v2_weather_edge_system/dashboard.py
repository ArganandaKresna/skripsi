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
    st.header("Simulasi Prediksi 24 Jam ke Depan")
    st.markdown("Wilayah: **Surabaya** (Karakteristik: Suhu Panas 31-36°C, Kelembaban Sedang-Rendah)")
    
    if st.button("Jalankan Prediksi 24 Jam"):
        with st.spinner("Memuat Model Edge dan Mengolah Data..."):
            model = HybridWeatherModel.build_model()
            # Force dummy initialization
            _ = model.predict(np.zeros((1, 24, 5)), verbose=0)
            
            preprocessor = DataPreprocessor()
            
            forecast_results = []
            current_time = datetime.datetime.now()
            
            for hour in range(1, 25):
                # Generate 288 samples simulating Surabaya
                base_data = np.random.rand(288, 5) * 100 
                # 0: Suhu, 1: Kelembaban, 2: Tekanan, 3: Angin, 4: Hujan
                base_data[:, 0] = np.random.uniform(31.0, 36.0, 288)
                base_data[:, 1] = np.random.uniform(40.0, 60.0, 288)
                base_data[:, 4] = np.random.uniform(0.0, 5.0, 288)
                
                df_dummy = pd.DataFrame(base_data, columns=["suhu", "kelembaban", "tekanan", "angin", "hujan"])
                
                # Use v2 pipeline to shape into (1, 24, 5)
                df_imputed = preprocessor.handle_missing_data(df_dummy)
                df_aggregated = preprocessor.aggregate_data(df_imputed)
                processed_tensor = preprocessor.scale_data(df_aggregated, fit=True)
                
                # Predict
                prediction = model.predict(processed_tensor, verbose=0)
                pred_class = int(np.argmax(prediction, axis=1)[0])
                pred_label = LABEL_MAP.get(pred_class, "Unknown")
                
                suhu = base_data[-1, 0]
                kelembaban = base_data[-1, 1]
                
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
                st.line_chart(forecast_df.set_index("Jam")["Suhu (°C)"])
                
            with col_table:
                st.subheader("Detail Prediksi")
                st.dataframe(forecast_df, use_container_width=True)
