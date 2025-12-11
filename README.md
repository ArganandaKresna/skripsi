# 🌦️ Autonomous Hyperlocal Weather Station with Hybrid LSTM-Transformer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MQTT](https://img.shields.io/badge/Communication-MQTT-660066?style=for-the-badge&logo=eclipse-mosquitto&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Proyek Skripsi**: Rancang Bangun Stasiun Cuaca Otonom Berbasis IoT untuk Prediksi Cuaca Hiperlokal Menggunakan Long Short-Term Memory (LSTM) dan Transformer.

---

## 📖 Ringkasan Proyek

Sistem ini adalah solusi *End-to-End* untuk memantau dan memprediksi cuaca secara lokal (hiperlokal). Menggunakan **ESP32** sebagai node sensor dan **Raspberry Pi 5** sebagai Edge Server yang menjalankan model Deep Learning Hybrid.

### 🌟 Fitur Utama
* **Hybrid AI Model**: Menggabungkan `LSTM` (Short-term dependencies) dan `Transformer` dengan *Positional Encoding* (Long-range dependencies).
* **Advanced Preprocessing**: Penerapan **Wavelet Transform** untuk *denoising* sinyal sensor.
* **Real-time Inference**: Prediksi cuaca (Suhu, Kelembaban, Kemungkinan Hujan) setiap detik di Edge Device.
* **IoT Architecture**: Komunikasi asinkronus menggunakan protokol MQTT.
* **Interactive Dashboard**: Visualisasi data menggunakan Streamlit yang dapat diakses via jaringan lokal/VPN.

---

## 📂 Struktur Direktori

Pastikan struktur folder Anda seperti berikut agar sistem berjalan lancar:

```text
skripsi_weather_system/
├── common/                  # [SHARED] Library Inti (Arsitektur Model & Teori)
│   ├── model_lib.py         # Definisi Class Model & Time2Vector
│   └── advanced_theory.py   # Algoritma Wavelet & Metrics
├── training_laptop/         # [LAPTOP] Environment Pelatihan
│   ├── train_manager.py     # Script Training Utama
│   ├── evaluate_model.py    # Script Evaluasi Skripsi
│   └── data/                # Dataset CSV
├── deployment_rpi/          # [RPI] Environment Deployment
│   ├── collector.py         # Backend MQTT -> SQLite
│   ├── dashboard.py         # Frontend Streamlit
│   ├── dummy_sensor.py      # Simulator Sensor
│   └── artifacts/           # [AUTO] Model hasil training disimpan disini
│       ├── weather_model.keras
│       └── scaler.pkl
├── requirements_laptop.txt
└── requirements_rpi.txt
```

## ⚡ Instalasi
1. Persiapan di Laptop (Training Environment)

Digunakan untuk melatih model dengan data historis.

```bash
# Clone atau Download repository ini
cd skripsi_weather_system

# Buat Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies Laptop
pip install -r requirements_laptop.txt
```

2. Persiapan di Raspberry Pi 5 (Deployment Environment)

Digunakan untuk menjalankan sistem secara headless.

```bash
# Update System & Install Mosquitto Broker
sudo apt update && sudo apt install mosquitto mosquitto-clients -y

# Buat Virtual Environment
cd ~/skripsi_weather_system
python3 -m venv venv
source venv/bin/activate

# Install Dependencies RPi
pip install -r requirements_rpi.txt
```

## 🚀 Cara Menjalankan Sistem (Step-by-Step)

Sistem ini dirancang modular. Ikuti urutan berikut untuk menjamin data mengalir dengan benar dari Training hingga Inference.

### FASE 1: Pelatihan Model (Di Laptop)
Lakukan ini setiap kali Anda mengubah dataset atau ingin memperbarui kecerdasan model.

1.  **Generate & Train Model**
    Jalankan script manajer pelatihan. Script ini akan melakukan preprocessing (termasuk Wavelet Denoising), training, dan evaluasi.
    ```bash
    cd training_laptop
    python train_manager.py
    ```
    > **Output:** File `weather_model.keras` dan `scaler.pkl` akan otomatis muncul di folder `deployment_rpi/artifacts/`.

2.  **Transfer Artifacts ke Raspberry Pi**
    Jika Anda mengerjakan coding di Laptop, pindahkan seluruh folder `deployment_rpi` (atau minimal folder `artifacts`) ke Raspberry Pi Anda.
    * *Tips: Gunakan `scp` atau `rsync` via Tailscale/SSH.*

---

### FASE 2: Deployment (Di Raspberry Pi 5)
Untuk menjalankan sistem secara utuh, Anda memerlukan **3 Terminal** (atau gunakan `tmux`/`screen` untuk menjalankannya di background).

#### 🟢 Terminal 1: Data Collector (Backend)
Bertugas mendengarkan data dari MQTT dan menyimpannya ke Database SQLite.
```bash
# Pastikan berada di root project
cd ~/skripsi_weather_system/deployment_rpi
source ../venv/bin/activate

python collector.py
```
Tanda Berhasil: Muncul pesan Collector Service Started... dan tidak ada error database.

🔵 Terminal 2: Dashboard AI (Frontend)

Bertugas menampilkan antarmuka visual dan menjalankan prediksi AI.

```bash
cd ~/skripsi_weather_system/deployment_rpi
source ../venv/bin/activate

streamlit run dashboard.py --server.port 8501
Akses: Buka browser di Laptop/HP dan ketik http://<IP-RPi-Anda>:8501.
```

#### 🟠 Terminal 3: Simulasi Sensor (Opsional)

Hanya jalankan jika alat ESP32 Anda sedang mati/belum siap. Script ini akan mengirim data dummy ke MQTT.

```bash
cd ~/skripsi_weather_system/deployment_rpi
source ../venv/bin/activate
```
`python dummy_sensor.py`
Tanda Berhasil: Dashboard akan mulai bergerak dan grafik terupdate setiap 5 detik.

#### 🛑 Cara Mematikan Sistem
Karena sistem berjalan di terminal (foreground), cara mematikannya adalah:

Matikan Dashboard & Sensor: Klik pada terminal yang sedang berjalan, lalu tekan tombol keyboard: `CTRL + C`

Matikan Collector: Sama, tekan `CTRL + C.`

Note: Mematikan collector aman, data di SQLite tidak akan rusak/corrupt karena transaksi bersifat atomik.

Kill Process (Jika Hang/Error): Jika terminal tertutup tapi program masih jalan di background (port 8501 masih terpakai), jalankan:

```bash
pkill -f streamlit
pkill -f python
```
## 📊 Metrik Evaluasi Model

Untuk keperluan Bab 4 Skripsi (Analisis dan Pembahasan), sistem ini menghitung metrik berikut secara otomatis menggunakan script `evaluate_model.py`. Metrik ini digunakan untuk memvalidasi seberapa akurat model Hybrid LSTM-Transformer dalam memprediksi cuaca dibandingkan data aktual.

| Metrik | Nama Lengkap | Fungsi | Target Nilai |
| :--- | :--- | :--- | :--- |
| **RMSE** | *Root Mean Squared Error* | Mengukur rata-rata besaran error prediksi dalam satuan asli data (misal: °C atau %). Memberikan bobot lebih pada error yang besar (outliers). | Semakin **Kecil** (Mendekati 0) semakin baik. |
| **MAE** | *Mean Absolute Error* | Rata-rata selisih mutlak antara nilai prediksi dan nilai aktual. Menunjukkan seberapa jauh rata-rata "melesetnya" prediksi. | Semakin **Kecil** semakin baik. |
| **R²** | *Coefficient of Determination* | Mengukur seberapa baik model dapat menjelaskan variasi data. Menunjukkan kecocokan (*Goodness of Fit*) pola prediksi terhadap pola data asli. | Mendekati **1.0** (100%) semakin akurat. |
| **MPD** | *Mean Percentage Difference* | Persentase kesalahan rata-rata relatif terhadap nilai asli. Berguna untuk membandingkan error antar parameter yang memiliki skala berbeda. | Di bawah **10%** dianggap sangat baik (*High Accuracy*). |

#### 🛠️ Troubleshooting
Q: Dashboard error "Model not found"? A: Pastikan Anda sudah menjalankan python train_manager.py di laptop dan folder artifacts/ berisi file .keras dan .pkl sudah ada di Raspberry Pi.

Q: Data tidak muncul di Dashboard? A: Cek collector.py di Terminal 1. Apakah ada pesan error? Jika hening, pastikan dummy_sensor.py atau ESP32 Anda mengirim ke topik MQTT yang benar (cuaca/sensor).

Q: Raspberry Pi panas? A: RPi 5 cukup kuat, tapi jika suhu > 80°C, pertimbangkan menggunakan Active Cooler. Inference AI berjalan di CPU.