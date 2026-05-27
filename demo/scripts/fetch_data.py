import urllib.request
import json
import pandas as pd
import datetime
import os

def fetch_historical_data():
    print("Mengambil data riwayat Surabaya (30 hari terakhir) dari Open-Meteo...")
    
    # Surabaya Coordinates
    lat = -7.2504
    lon = 112.7688
    
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=30)
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation&timezone=Asia%2FJakarta"
    
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
    
    # Handle NaNs from API immediately by ffill/bfill
    df = df.ffill().bfill()
    
    os.makedirs("database", exist_ok=True)
    csv_path = "database/historical_surabaya.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Data berhasil diunduh dan disimpan ke: {csv_path}")
    print(f"Total data: {len(df)} jam")

if __name__ == "__main__":
    fetch_historical_data()
