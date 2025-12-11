import paho.mqtt.client as mqtt
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
import threading

# --- CONFIG ---
MQTT_BROKER = "localhost"
MQTT_TOPIC = "cuaca/sensor"
DB_FILE = os.path.join(os.path.dirname(__file__), "weather_data.db")
RETENTION_DAYS = 7  # Simpan data hanya 7 hari ke belakang

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS measurements 
                 (timestamp DATETIME, temp REAL, wind_speed REAL, 
                  pressure REAL, humidity REAL, rain_condition INTEGER)''')
    # Indexing agar query visualisasi 7 hari cepat
    c.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp ON measurements(timestamp)''')
    conn.commit()
    conn.close()

def cleanup_old_data():
    """Menghapus data yang lebih tua dari RETENTION_DAYS"""
    while True:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
            c.execute("DELETE FROM measurements WHERE timestamp < ?", (cutoff_date,))
            deleted = c.rowcount
            conn.commit()
            conn.close()
            if deleted > 0:
                print(f"[Maintenance] Menghapus {deleted} data lama (> {RETENTION_DAYS} hari).")
        except Exception as e:
            print(f"[Error] Cleanup failed: {e}")
        
        # Cek setiap 6 jam sekali saja sudah cukup
        time.sleep(6 * 3600)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        # Validasi data (pastikan key sesuai JSON ESP32 kamu)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?)", 
                  (datetime.now(), data['temp'], data['wind'], 
                   data['press'], data['hum'], data['rain']))
        conn.commit()
        conn.close()
        print(f"Saved: {data}") # Debug log
    except Exception as e:
        print(f"Error processing: {e}")

if __name__ == "__main__":
    init_db()
    
    # Jalankan cleanup di thread terpisah agar tidak mengganggu penerimaan MQTT
    t = threading.Thread(target=cleanup_old_data, daemon=True)
    t.start()

    client = mqtt.Client()
    client.on_connect = lambda c, u, f, rc: c.subscribe(MQTT_TOPIC)
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        print("Collector Service (7-Day Retention) Started...")
        client.loop_forever()
    except Exception as e:
        print(f"Connection Failed: {e}")