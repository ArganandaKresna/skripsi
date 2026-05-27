import time
import psutil
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification(y_true, y_pred):
    """Evaluates multi-class classification using Precision, Recall, F1."""
    print("\n--- Classification Evaluation ---")
    target_names = ["Hujan", "Panas", "Cerah"]
    
    # Convert one-hot or proba to labels if needed
    if len(np.shape(y_true)) > 1 and np.shape(y_true)[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(np.shape(y_pred)) > 1 and np.shape(y_pred)[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("---------------------------------")


def profile_edge_hardware(model, dummy_input):
    """Profiles the hardware usage over 100 inferences."""
    print("\n--- Edge Hardware Profiling ---")
    
    # 1. Warm-up
    print("Warming up model...")
    _ = model.predict(dummy_input, verbose=0)
    
    # 2. Setup counters
    latencies = []
    psutil.cpu_percent(interval=None) # reset cpu counter
    
    # 3. Looping Inference 100x
    print("Running 100 consecutive inferences...")
    for _ in range(100):
        start_time = time.time()
        _ = model.predict(dummy_input, verbose=0)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)
    
    # Hardware check
    avg_cpu = psutil.cpu_percent(interval=None)
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    
    avg_latency = np.mean(latencies)
    peak_latency = np.max(latencies)
    
    print(f"Rata-rata Inference Latency : {avg_latency:.2f} ms")
    print(f"Puncak Latency              : {peak_latency:.2f} ms")
    print(f"Rata-rata CPU Usage         : {avg_cpu:.1f} %")
    print(f"RAM Allocation              : {ram_mb:.2f} MB")
    print("-------------------------------\n")
