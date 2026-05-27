import sys
import types
if 'imp' not in sys.modules:
    sys.modules['imp'] = types.ModuleType('imp')

import time
import datetime
import psutil
import os
import numpy as np

from core.inference import WeatherInference
from database.db_manager import DBManager

def get_memory_usage():
    """Returns the RAM allocation of the current Python process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("=== Initialize Weather Station Edge ===")
    
    # 1. Init Database
    db = DBManager()
    
    # 2. Init Model
    inference = WeatherInference()
    
    # 3. Warm-up Run
    print("Warming up model...")
    dummy_input = np.random.rand(1, 288, 5).astype(np.float32)
    inference.predict(dummy_input)
    
    print("=== Starting Continuous Inference ===")
    
    # Reset CPU counter
    psutil.cpu_percent(interval=None)
    
    try:
        while True:
            # Generate Dummy Data (1, 288, 5) float32
            # Features: [suhu, kelembaban, tekanan, angin, hujan]
            sensor_data = np.random.rand(1, 288, 5).astype(np.float32)
            
            # Record time
            timestamp = datetime.datetime.now().isoformat()
            
            # Predict
            pred_class = inference.predict(sensor_data)
            label_map = {0: "Cerah", 1: "Panas", 2: "Hujan"}
            pred_label = label_map.get(pred_class, "Unknown")
            
            # Get latest mock sensor readings for logging to DB (take from last timestep)
            latest_reading = sensor_data[0, -1, :]
            suhu = float(latest_reading[0] * 40) # Scale to 0-40 C
            kelembaban = float(latest_reading[1] * 100) # Scale to 0-100 %
            tekanan = float(latest_reading[2] * 50 + 980) # Scale to 980-1030 hPa
            angin = float(latest_reading[3] * 20) # Scale to 0-20 m/s
            hujan = float(latest_reading[4] * 50) # Scale to 0-50 mm
            
            # Save to database
            db.insert_weather(timestamp, suhu, kelembaban, tekanan, angin, hujan, pred_class)
            
            # Hardware metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            mem_usage = get_memory_usage()
            
            print(f"[{timestamp}] Prediksi: {pred_label} | CPU: {cpu_usage}% | RAM: {mem_usage:.2f} MB")
            
            # Wait 1 second
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        db.close()
        print("Database connection closed. System shutdown.")

if __name__ == "__main__":
    main()
