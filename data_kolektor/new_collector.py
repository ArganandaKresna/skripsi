import os
import time
import sqlite3
import requests
from datetime import datetime

API_KEY = "5d46368f1024970f93a035dab5c95de2"
LAT = -7.266609668436994
LON = 112.78618232416069
URL = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

DB_FILE = "weather_data.db"

def init_db():
    """Inisialisasi database SQLite dan membuat tabel jika belum ada."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            suhu REAL,
            temperatur REAL,
            kelembaban REAL,
            tekanan_udara REAL,
            kondisi_hujan TEXT,
            kecepatan_angin REAL
        )
    ''')
    try:
        cursor.execute('ALTER TABLE weather_log ADD COLUMN kecepatan_angin REAL')
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def fetch_data():
    """Mengambil 1 sampel data dari API OpenWeatherMap dan menyimpannya ke SQLite."""
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        # Ekstrak parameter yang diminta
        suhu = data['main']['temp']
        temperatur_terasa = data['main']['feels_like']
        kelembaban = data['main']['humidity']
        tekanan_udara = data['main']['pressure']
        kecepatan_angin = data.get('wind', {}).get('speed', 0.0)
        
        # Penentuan kondisi hujan
        weather_main = data['weather'][0]['main'].lower()
        
        is_raining = "rain" in weather_main or "drizzle" in weather_main or "thunderstorm" in weather_main
        kondisi_hujan = "Ya" if is_raining else "Tidak"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Simpan ke SQLite
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO weather_log (
                timestamp, suhu, temperatur, 
                kelembaban, tekanan_udara, kondisi_hujan, kecepatan_angin
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, suhu, temperatur_terasa, kelembaban, tekanan_udara, kondisi_hujan, kecepatan_angin))
        conn.commit()
        conn.close()
            
        print(f"[{timestamp}] Data tersimpan -> Suhu: {suhu}°C, Kelembaban: {kelembaban}%, Tekanan: {tekanan_udara}hPa, Hujan: {kondisi_hujan}, Angin: {kecepatan_angin}m/s")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Gagal mengambil data: {e}")

def main():
    init_db()
    print("===================================================")
    print("Program Kolektor Data Cuaca OpenWeatherMap Dimulai")
    print("Database: SQLite (weather_data.db)")
    print("Lokasi: Surabaya (-7.2666, 112.7861)")
    print("Frekuensi: Setiap 5 detik (0.2 Hz) secara terus-menerus")
    print("===================================================\n")
    
    while True:
        # Langsung ambil data
        fetch_data()
        
        # Tidur selama 5 detik untuk menjaga frekuensi 0.2 Hz
        time.sleep(5)

if __name__ == "__main__":
    # Pastikan database tersimpan di folder yang sama dengan script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()