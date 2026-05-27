# Localized Smart Weather Station (Edge Computing System)

Repositori ini berisi sistem perangkat lunak untuk penelitian skripsi mengenai **Sistem Klasifikasi Cuaca Hiperlokal Berbasis Edge Computing** menggunakan arsitektur *Hybrid Deep Learning (LSTM-Transformer)*. Sistem ini dirancang secara khusus untuk dieksekusi secara otonom pada *hardware* Edge (seperti Raspberry Pi 5).

## Struktur Direktori

Repositori ini memiliki beberapa iterasi pengembangan:
- **`v1_weather-station-edge/`**: Iterasi pertama (Prototype lama).
- **`v2_weather_edge_system/`**: Iterasi kedua (Production-Ready). Memuat arsitektur sistem IoT cerdas yang sangat terstruktur, *auto-preprocessing pipeline*, arsitektur model *Transformer*, penyimpanan *database*, dan *Streamlit Dashboard* terbaru.

## 📦 Dependensi Sistem

Untuk menjalankan seluruh *pipeline* (mulai dari pra-pemrosesan, *hyperparameter tuning*, hingga *hardware profiling*), sistem ini membutuhkan pustaka Python berikut:

- `tensorflow` (>= 2.12.0) - *Core Deep Learning Framework*
- `numpy` - Komputasi matriks/tensor
- `pandas` - Manipulasi dan agregasi dataset *time-series*
- `scikit-learn` - *MinMaxScaler* dan metrik evaluasi klasifikasi (*Precision, Recall, F1*)
- `keras-tuner` - Otomatisasi pencarian *Hyperparameter* terbaik
- `paho-mqtt` - Protokol komunikasi IoT (Pub/Sub data sensor Edge)
- `psutil` - *Hardware profiling* (Pemantauan Latency, RAM, dan CPU)
- `streamlit` - Visualisasi antarmuka web

## ⚠️ Catatan Kekurangan Sistem Saat Ini (Evaluasi Berkelanjutan)

Meskipun arsitektur *pipeline* telah siap pakai (*production-ready*), sistem saat ini masih berupa **Simulasi Perangkat Lunak** dengan beberapa keterbatasan (kekurangan) yang harus diselesaikan untuk implementasi dunia nyata:

1. **Data Sensor Masih Berupa Simulasi (Dummy):** 
   Sistem masih menggunakan `np.random` di dalam `main.py` untuk mengimitasi input lingkungan. Agar sistem berjalan sesungguhnya, modul pembacaan pin GPIO/I2C dari sensor fisik (misalnya DHT22 untuk suhu/kelembaban, BMP280 untuk tekanan udara) perlu dihubungkan.
2. **Ketiadaan Dataset Pelatihan Historis:** 
   *Keras-Tuner* dan *Model Evaluation* saat ini beroperasi pada data dan label sintetis. Oleh karena itu, tingkat akurasi (akurasi ~30%) yang dihasilkan hanyalah kebetulan semata. Model perlu dilatih ulang (*retraining*) secara masif menggunakan minimal 1-2 tahun data riwayat iklim BMKG/lokal.
3. **Ketergantungan *Virtual Environment* Global:** 
   Saat ini sistem menggunakan *shared virtual environment* di root `/home/arga/skripsi/venv`. Pada tahap *deployment* massal di Edge, *requirements.txt* milik v2 harus di-*build* dalam wadah *container* (seperti Docker) secara terisolasi.
4. **Hardcoded Scaling Limits:** 
   Modul prapemrosesan (`DataPreprocessor`) saat ini memiliki rentang minimum-maksimum skala data yang diasumsikan secara statis (contoh: Suhu 15-45°C). Idealnya, *Scaler* ini harus di-*fit* dari statistik *dataset* historis sungguhan sebelum dipakai inferensi.

## 🚀 Cara Menjalankan Sistem v2

Iterasi `v2` adalah sistem lengkap yang memiliki mesin inferensi latar belakang dan *Dashboard Web* interaktif.

### 1. Jalankan Orkestrator Inferensi (Latar Belakang)
Buka terminal baru dan jalankan skrip ini untuk menyalakan "sensor" dan menyimpan log prediksi ke database secara *real-time*:
```bash
cd /home/arga/skripsi/v2_weather_edge_system
source ../venv/bin/activate
python3 main.py
```

### 2. Luncurkan Streamlit Dashboard
Buka terminal *lainnya*, lalu jalankan perintah ini untuk melihat visualisasi data:
```bash
cd /home/arga/skripsi/v2_weather_edge_system
source ../venv/bin/activate
streamlit run dashboard.py
```
*Setelah diluncurkan, buka tautan `http://localhost:8501` di browser Anda.*

---
*Dibuat untuk penelitian Edge AI dan Smart Weather Classification.*
