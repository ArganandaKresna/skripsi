import sys
import os
import time
import sqlite3
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- SETUP PATH & IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.model_lib import Time2Vector 

# --- KONFIGURASI ---
DB_PATH = os.path.join(current_dir, "weather_data.db")
MODEL_PATH = os.path.join(current_dir, "artifacts/weather_model.keras")
SCALER_PATH = os.path.join(current_dir, "artifacts/scaler.pkl")
SEQ_LENGTH = 24  
PREDICT_HOURS = 72 

# --- SETUP PAGE ---
st.set_page_config(
    page_title="SkyNet BotRobot",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan lebih modern
st.markdown("""
<style>
    .big-font { font-size: 20px !important; font-weight: bold; }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    /* Dark mode support adjustment */
    @media (prefers-color-scheme: dark) {
        .metric-container {
            background-color: #262730;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI ILMIAH (METEOROLOGI) ---
def calculate_dew_point(T, RH):
    """Menghitung Titik Embun (Dew Point) - Rumus Magnus"""
    a = 17.27
    b = 237.7
    alpha = ((a * T) / (b + T)) + np.log(RH/100.0)
    return (b * alpha) / (a - alpha)

def calculate_heat_index(T, RH):
    """Menghitung Heat Index (Terasa Seperti) - Rumus NOAA sederhana"""
    # Rumus regresi sederhana untuk T dalam Celcius
    c1 = -8.78469475556
    c2 = 1.61139411
    c3 = 2.33854883889
    c4 = -0.14611605
    c5 = -0.012308094
    c6 = -0.0164248277778
    c7 = 0.002211732
    c8 = 0.00072546
    c9 = -0.000003582
    
    HI = c1 + (c2 * T) + (c3 * RH) + (c4 * T * RH) + (c5 * T**2) + (c6 * RH**2) + \
         (c7 * T**2 * RH) + (c8 * T * RH**2) + (c9 * T**2 * RH**2)
    return HI

# --- DATA LOADER ---
@st.cache_resource
def load_ai_system():
    if not os.path.exists(MODEL_PATH): return None, None
    scaler = joblib.load(SCALER_PATH)
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'Time2Vector': Time2Vector})
    except:
        # Fallback jika custom objects bermasalah (biasanya krn beda versi TF)
        st.error("Gagal load Custom Layer. Pastikan library common/model_lib.py terbaca.")
        return None, None
    return model, scaler

def get_data_for_dashboard():
    conn = sqlite3.connect(DB_PATH)
    # Ambil data 7 hari terakhir
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
        # Resample per jam
        df_hourly = df.resample('1h').mean().interpolate()
        return df_hourly.dropna()
    return pd.DataFrame()

model, scaler = load_ai_system()

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("⚙️ System Status")
    if model:
        st.success("✅ LSTM Model Loaded")
    else:
        st.error("❌ LSTM Model Missing")
    
    st.info(f"📁 Database: {DB_PATH}")
    st.markdown("---")
    st.write("**Architecture:**")
    st.caption("Hybrid LSTM + Transformer")
    st.markdown("---")
    st.write("© 2025 Skripsi IoT")

# --- MAIN DASHBOARD LOOP ---
placeholder = st.empty()

while True:
    df = get_data_for_dashboard()
    
    with placeholder.container():
        st.title("🌦️ Autonomous Weather Station Dashboard")
        st.markdown(f"**Last Update:** {pd.Timestamp.now().strftime('%d %B %Y, %H:%M:%S')}")

        if len(df) < SEQ_LENGTH:
            st.warning(f"⏳ Menginisialisasi Sistem... Data terkumpul: {len(df)}/{SEQ_LENGTH} jam.")
            st.progress(min(len(df)/SEQ_LENGTH, 1.0))
        else:
            latest = df.iloc[-1]
            prev_hour = df.iloc[-2] if len(df) > 1 else latest

            # Hitung Advanced Metrics
            dew_point = calculate_dew_point(latest['temp'], latest['humidity'])
            heat_index = calculate_heat_index(latest['temp'], latest['humidity'])

            # --- ROW 1: KEY METRICS (KPIs) ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            with kpi1:
                st.metric("🌡️ Temperatur", f"{latest['temp']:.1f} °C", 
                          f"{latest['temp'] - prev_hour['temp']:.2f} °C")
            with kpi2:
                st.metric("💧 Kelembaban", f"{latest['humidity']:.1f} %",
                          f"{latest['humidity'] - prev_hour['humidity']:.2f} %")
            with kpi3:
                st.metric("🥵 Heat Index", f"{heat_index:.1f} °C", delta_color="off")
            with kpi4:
                rain_txt = "Hujan" if latest['rain_condition'] > 0.5 else "Cerah"
                st.metric("🌧️ Kondisi", rain_txt)

            st.divider()

            # --- TABS NAVIGASI ---
            tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📊 Analisis Detail", "📥 Raw Data"])

            # === TAB 1: AI FORECAST ===
            with tab1:
                col_chart, col_gauge = st.columns([3, 1])
                
                # --- PREDIKSI LOGIC ---
                if model:
                    # Prepare Input
                    input_df = df.tail(SEQ_LENGTH)[['temp', 'wind_speed', 'pressure', 'humidity', 'rain_condition']]
                    input_scaled = scaler.transform(input_df.values)
                    current_batch = input_scaled.reshape(1, SEQ_LENGTH, 5)
                    
                    predictions_scaled = []
                    for _ in range(PREDICT_HOURS):
                        pred = model.predict(current_batch, verbose=0)
                        predictions_scaled.append(pred[0])
                        pred_reshaped = pred.reshape(1, 1, 5)
                        current_batch = np.append(current_batch[:, 1:, :], pred_reshaped, axis=1)
                    
                    predictions_actual = scaler.inverse_transform(np.array(predictions_scaled))
                    
                    # Create DF Prediksi
                    last_time = df.index[-1]
                    future_dates = [last_time + pd.Timedelta(hours=x+1) for x in range(PREDICT_HOURS)]
                    df_pred = pd.DataFrame(predictions_actual, index=future_dates, columns=input_df.columns)

                    # --- PLOT FORECAST ---
                    with col_chart:
                        fig_forecast = go.Figure()
                        
                        # Data Masa Lalu (24 jam terakhir saja biar jelas)
                        history_view = df.tail(48)
                        fig_forecast.add_trace(go.Scatter(x=history_view.index, y=history_view['temp'], 
                                                        mode='lines', name='History', 
                                                        line=dict(color='gray', width=2)))
                        
                        # Data Prediksi
                        fig_forecast.add_trace(go.Scatter(x=df_pred.index, y=df_pred['temp'], 
                                                        mode='lines', name='Prediction',
                                                        line=dict(color='#FF4B4B', width=4))) # Merah Streamlit
                        
                        # Garis Vertikal "NOW"
                        fig_forecast.add_vline(x=last_time, line_width=1, line_dash="dash", line_color="white")
                        fig_forecast.add_annotation(x=last_time, y=latest['temp'], text="Sekarang", showarrow=True, arrowhead=1)

                        fig_forecast.update_layout(
                            title="Prediksi Suhu 3 Hari Kedepan",
                            xaxis_title="Waktu", yaxis_title="Suhu (°C)",
                            hovermode="x unified",
                            height=400,
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

                    # --- GAUGE CHART (TEKANAN) ---
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = latest['pressure'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Tekanan (hPa)"},
                            delta = {'reference': 1013, 'increasing': {'color': "green"}},
                            gauge = {
                                'axis': {'range': [980, 1040], 'tickwidth': 1},
                                'bar': {'color': "lightblue"},
                                'steps': [
                                    {'range': [980, 1000], 'color': "lightgray"},
                                    {'range': [1000, 1020], 'color': "gray"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 1013}}))
                        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        st.caption("1013 hPa = Tekanan Standar Laut")

            # === TAB 2: ANALISIS DETAIL ===
            with tab2:
                col_ana1, col_ana2 = st.columns(2)
                
                with col_ana1:
                    # Dual Axis Chart: Temp vs Humidity
                    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig_combo.add_trace(go.Scatter(x=df.index, y=df['temp'], name="Suhu", line=dict(color='orange')), secondary_y=False)
                    fig_combo.add_trace(go.Scatter(x=df.index, y=df['humidity'], name="Kelembaban", line=dict(color='cyan', dash='dot')), secondary_y=True)
                    
                    fig_combo.update_layout(title="Korelasi Suhu & Kelembaban (7 Hari)", height=350)
                    fig_combo.update_yaxes(title_text="Suhu (°C)", secondary_y=False)
                    fig_combo.update_yaxes(title_text="Kelembaban (%)", secondary_y=True)
                    st.plotly_chart(fig_combo, use_container_width=True)
                
                with col_ana2:
                    # Scatter Plot Angin vs Tekanan
                    fig_scatter = px.scatter(df, x="pressure", y="wind_speed", color="temp",
                                           title="Distribusi Kecepatan Angin vs Tekanan",
                                           labels={"pressure": "Tekanan (hPa)", "wind_speed": "Kec. Angin (m/s)"})
                    st.plotly_chart(fig_scatter, use_container_width=True)

            # === TAB 3: RAW DATA ===
            with tab3:
                st.subheader("Database Records")
                st.dataframe(df.sort_index(ascending=False), use_container_width=True, height=300)
                
                # Tombol Download CSV
                csv = df.to_csv().encode('utf-8')
                st.download_button("📥 Download Data CSV", data=csv, file_name="weather_history.csv", mime="text/csv")

    time.sleep(10) # Refresh tiap 10 detik