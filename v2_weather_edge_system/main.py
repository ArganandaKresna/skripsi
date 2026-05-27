import os
import sys
import types
if 'imp' not in sys.modules:
    sys.modules['imp'] = types.ModuleType('imp')

import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from core.preprocessor import DataPreprocessor
from core.hybrid_model import HybridWeatherModel
from core.tuner import run_tuning
from scripts.evaluator import profile_edge_hardware, evaluate_classification
from database.db_manager import DBManager

def main():
    print("=== Localized Smart Weather Station (Edge System) ===")
    
    # 1. Setup DB
    db = DBManager()
    
    # 2. Build Model
    print("\n[1] Building HybridWeatherModel...")
    model = HybridWeatherModel.build_model()
    # Dummy predicting to initialize weights
    _ = model.predict(np.zeros((1, 24, 5)), verbose=0)
    
    # 3. Hardware Profiling
    print("\n[2] Executing Hardware Profiling...")
    # profile_edge_hardware(model, np.zeros((1, 24, 5)))
    print("Skipping full hardware profile for continuous mode. (Uncomment in code to run)")
    
    # 4. Continuous Mode
    print("\n[3] Starting Continuous Edge Inference...")
    preprocessor = DataPreprocessor()
    classes = ["Hujan", "Panas", "Cerah"]
    
    try:
        while True:
            # Generate 24 hours of dummy raw data (288 samples)
            columns = ["suhu", "kelembaban", "tekanan", "angin", "hujan"]
            raw_data = np.random.rand(288, 5) * 100 
            df_raw = pd.DataFrame(raw_data, columns=columns)
            
            # Preprocess
            df_imputed = preprocessor.handle_missing_data(df_raw)
            df_aggregated = preprocessor.aggregate_data(df_imputed)
            processed_tensor = preprocessor.scale_data(df_aggregated, fit=True)
            
            # Predict
            prediction_proba = model.predict(processed_tensor, verbose=0)
            pred_class_idx = np.argmax(prediction_proba, axis=1)[0]
            
            # Extract latest raw sensor value for logging (representing current state)
            latest_reading = df_raw.iloc[-1]
            timestamp = datetime.datetime.now().isoformat()
            
            suhu = float(latest_reading["suhu"])
            kelembaban = float(latest_reading["kelembaban"])
            tekanan = float(latest_reading["tekanan"])
            angin = float(latest_reading["angin"])
            hujan = float(latest_reading["hujan"])
            
            # Save to DB
            db.insert_weather(timestamp, suhu, kelembaban, tekanan, angin, hujan, int(pred_class_idx))
            
            # Print to Terminal
            print(f"[{timestamp}] Prediksi: {classes[pred_class_idx]} | Probabilitas: {prediction_proba[0]}")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping inference...")
        print("System shutdown.")

if __name__ == "__main__":
    main()
