import os
import time
import sqlite3
import requests
from datetime import datetime, timedelta

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

def collect_for_one_minute():
    """Menjalankan pengambilan data selama 1 menit pada 0.2 Hz (12 sampel)."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Memulai pengambilan data 1 menit (12 sampel @ 0.2 Hz)...")
    
    # 0.2 Hz berarti 1 data setiap 5 detik. Total 12 data untuk 1 menit (60 detik)
    for i in range(12):
        fetch_data()
        if i < 11:  # Tidak perlu sleep di iterasi terakhir
            time.sleep(5)
            
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sesi pengambilan data selesai.\n")

def main():
    init_db()
    print("===================================================")
    print("Program Kolektor Data Cuaca OpenWeatherMap Dimulai")
    print("Database: SQLite (weather_data.db)")
    print("Lokasi: Surabaya (-7.2666, 112.7861)")
    print("Frekuensi: 1 menit @ 0.2 Hz, di awal setiap jam")
    print("===================================================\n")
    
    while True:
        now = datetime.now()
        
        # Jika saat ini berada tepat di awal jam (menit 0, detik 0-5)
        # Menambahkan sedikit toleransi detik untuk menghindari missed trigger
        if now.minute == 0 and now.second < 10:
            collect_for_one_minute()
            
            # Tidur selama 59 menit untuk menghindari memicu lagi dalam jam yang sama
            # dan untuk memberikan waktu transisi yang cukup ke jam berikutnya
            time.sleep(59 * 60)
        else:
            # Hitung waktu tunggu hingga awal jam berikutnya (menit 00, detik 00)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            sleep_seconds = (next_hour - now).total_seconds()
            
            print(f"Menunggu sesi berikutnya pada {next_hour.strftime('%Y-%m-%d %H:%M:%S')} (dalam {int(sleep_seconds)} detik)...")
            
            # Sleep menyesuaikan dengan target waktu
            time.sleep(sleep_seconds)

if __name__ == "__main__":
    # Pastikan database tersimpan di folder yang sama dengan script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
