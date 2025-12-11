import paho.mqtt.client as mqtt
import sqlite3
import json
import os
from datetime import datetime

# --- CONFIG ---
MQTT_BROKER = "localhost" # Mosquitto di RPi sendiri
MQTT_TOPIC = "cuaca/sensor"
DB_FILE = os.path.join(os.path.dirname(__file__), "weather_data.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Tabel measurements
    c.execute('''CREATE TABLE IF NOT EXISTS measurements 
                 (timestamp DATETIME, temp REAL, wind_speed REAL, 
                  pressure REAL, humidity REAL, rain_condition INTEGER)''')
    conn.commit()
    conn.close()

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        # Validasi sederhana
        required = ['temp', 'wind', 'press', 'hum', 'rain']
        if not all(k in data for k in required):
            print("Data tidak lengkap!")
            return

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Mapping JSON key ke Database column
        # Asumsi JSON dari ESP32: {"temp": 30.2, "wind": 5.1, "press": 1010, "hum": 80, "rain": 0}
        c.execute("INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?)", 
                  (datetime.now(), data['temp'], data['wind'], 
                   data['press'], data['hum'], data['rain']))
        
        conn.commit()
        conn.close()
        print(f"Saved: {data}")
        
    except Exception as e:
        print(f"Error processing message: {e}")

if __name__ == "__main__":
    init_db()
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        print("Collector Service Started...")
        client.loop_forever()
    except Exception as e:
        print(f"Connection Failed: {e}")