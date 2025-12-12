import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class WeatherPreprocessor:
    def __init__(self, seq_length=24, features=None):
        self.seq_length = seq_length
        self.features = features or ['temp', 'wind_speed', 'pressure', 'humidity', 'rain_condition']
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) # Tanh suka range -1 s/d 1

    def clean_and_resample(self, df):
        """Membersihkan data mentah dan resample ke per-jam"""
        # Pastikan format datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Resample per jam (mean) dan interpolate data kosong
        df_hourly = df.resample('1h').mean().interpolate()
        return df_hourly.dropna()

    def fit_transform(self, df):
        """Scaling data untuk training"""
        data = df[self.features].values
        return self.scaler.fit_transform(data)

    def transform(self, df):
        """Scaling data untuk inference (pakai scaler yang sudah difit)"""
        data = df[self.features].values
        return self.scaler.transform(data)

    def inverse_transform(self, scaled_data):
        """Kembalikan ke nilai asli"""
        # Handle jika input cuma 1 dimensi (misal hasil prediksi)
        if len(scaled_data.shape) == 1:
             # Kita butuh trik karena scaler mengharapkan 5 fitur
             # Asumsi urutan fitur: temp, wind, press, hum, rain
             # Ini hanya mengembalikan kolom pertama (temp) jika input cuma 1 kolom
             # Untuk skripsi sebaiknya kita inverse matriks lengkap
             pass 
        return self.scaler.inverse_transform(scaled_data)

    def create_sequences(self, data):
        """Membuat Sliding Window untuk LSTM"""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i : i + self.seq_length])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)
    
    def save_scaler(self, path):
        joblib.dump(self.scaler, path)
        
    def load_scaler(self, path):
        self.scaler = joblib.load(path)