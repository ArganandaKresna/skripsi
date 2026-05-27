# DEMO: Real Historical Data Inference (Surabaya)

Direktori `demo/` ini merupakan versi *fork* mutakhir dari sistem utama (*v2*) yang telah dimodifikasi secara khusus untuk diujikan pada **data cuaca historis dunia nyata**. 

Sistem ini tidak lagi menggunakan pembacaan *dummy random*, melainkan menarik riwayat klimatologi sungguhan dari Open-Meteo, melatih model AI dari nol, lalu menyimulasikan aliran data riwayat tersebut seolah-olah terjadi secara *real-time*.

## 🚀 Urutan Cara Menjalankan Eksekusi Demo

Ikuti langkah-langkah di bawah ini secara berurutan agar *pipeline* berjalan sempurna. Pastikan Anda berada di direktori `/home/arga/skripsi/demo/` dan virtual environment sudah aktif!

```bash
cd /home/arga/skripsi/demo
source ../venv/bin/activate
```

### Langkah 1: Mengunduh Dataset Cuaca Asli
Skrip ini akan menarik data suhu, kelembaban, tekanan, angin, dan curah hujan Surabaya 30 hari ke belakang dari Open-Meteo API.
```bash
python3 scripts/fetch_data.py
```
*(Data akan tersimpan di `database/historical_surabaya.csv`)*

### Langkah 2: Pembersihan Data & Pelatihan Model (Training)
Skrip ini membagi dataset menjadi tensor *sliding window* 24 jam, menentukan label *ground-truth* (Hujan/Panas/Cerah), lalu mulai melatih arsitektur Hybrid LSTM-Transformer hingga mencapai akurasi optimal (biasanya ~96%).
```bash
PYTHONPATH=. python3 scripts/train.py
```
*(Bobot model pintar ini akan direkam permanen ke `models/trained_model.keras`)*

### Langkah 3: Menjalankan Orkestrator Inferensi (Waktu Nyata)
Setelah model pintar dilatih, jalankan mesin utama untuk menyimulasikan klasifikasi *real-time* berbasis data uji (*test set*). Skrip ini berjalan terus-menerus dan memompa hasil prediksinya ke file `weather_data.db`.
```bash
python3 main.py
```

### Langkah 4: Pemantauan Visual (Dashboard Streamlit)
Untuk melihat pergerakan sensor dan visualisasi secara interaktif layaknya BMKG modern, biarkan `main.py` berjalan, lalu buka terminal baru:
```bash
cd /home/arga/skripsi/demo
source ../venv/bin/activate
streamlit run dashboard.py
```
*(Buka URL `http://localhost:8501` di browser untuk menikmati antarmukanya)*

### Langkah 5: Mode Deteksi Ekstrem (LIVE Telemetry Waktu Nyata)
Jika Anda ingin mendemokan ke dosen penguji bahwa sistem **benar-benar menyedot data Surabaya SAAT INI**, hentikan skrip `main.py` di atas, dan jalankan varian cerdas ini:
```bash
python3 live_inference.py
```
*(Skrip ini akan mengontak API satelit cuaca setiap 30 detik, mengambil rekam jejak 24 jam ke belakang **hingga detik ini**, menjejalinya ke arsitektur Hybrid, dan memvisualisasikan hasilnya secara serentak ke Streamlit!)*

---
> [!TIP]  
> **Modifikasi Lokasi:** Ingin menguji keampuhan cuaca Jakarta atau Bandung? Cukup buka `scripts/fetch_data.py` dan `live_inference.py`, ubah angka koordinat `lat` (latitude) dan `lon` (longitude) miliknya, lalu jalankan seluruh siklus dari Langkah 1 lagi!
