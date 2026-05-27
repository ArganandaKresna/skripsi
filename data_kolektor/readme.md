# OpenWeatherMap Data Collector untuk Skripsi

Direktori ini berisi program otonom (latar belakang) untuk mengumpulkan data cuaca real-time dari API OpenWeatherMap. Program dirancang khusus untuk memenuhi parameter pengumpulan dataset pada skripsi.

## Spesifikasi Pengumpulan Data
- **Lokasi**: Surabaya (Latitude: `-7.2666`, Longitude: `112.7861`)
- **Frekuensi (Sampling Rate)**: `0.2 Hz` (1 sampel setiap 5 detik)
- **Durasi Sesi**: `1 Menit` (Total 12 sampel per sesi)
- **Siklus Pengambilan**: Dilakukan tepat di **awal setiap jam** (menit `00`).
- **Penyimpanan**: Menggunakan database relasional `SQLite` (`weather_data.db`).

## Parameter (Fitur) yang Disimpan
1. `suhu` (Suhu aktual dalam Celcius)
2. `temperatur` (Suhu yang dirasakan oleh tubuh dalam Celcius)
3. `kelembaban` (Kelembaban udara / *Humidity* dalam Persen)
4. `tekanan_udara` (Tekanan udara tingkat permukaan laut dalam hPa)
5. `kondisi_hujan` (Label Kondisi Hujan; `Ya` jika status cuaca berupa rain/drizzle/thunderstorm, sebaliknya `Tidak`)

## Struktur File
```text
data_kolektor/
├── collector.py       # Script utama Python untuk mengambil data dari API
├── weather_data.db    # Database SQLite tempat data disimpan (Otomatis dibuat)
├── log_kolektor.txt   # File teks yang berisi log rekam jejak program berjalan
└── readme.md          # File dokumentasi ini
```

## Cara Menjalankan Program (Di Latar Belakang)
Untuk menjalankan program tanpa henti agar bisa ditinggal selama sebulan:
1. Buka terminal
2. Masuk ke folder direktori ini:
   ```bash
   cd /home/arga/skripsi/data_kolektor
   ```
3. Eksekusi program di latar belakang dengan perintah `nohup`:
   ```bash
   nohup python -u collector.py > log_kolektor.txt 2>&1 &
   ```

## Cara Memantau Log
Untuk melihat apakah program masih berjalan normal dan menunggu jadwal berikutnya:
```bash
tail -f log_kolektor.txt
```
*(Tekan `Ctrl+C` untuk keluar dari mode pemantauan, ini tidak akan mematikan program)*.

## Cara Menghentikan Program
Jika pengumpulan data sudah selesai dilakukan sesuai durasi penelitian, berhentikan program dengan:
```bash
pkill -f collector.py
```

## Cara Mengekstrak/Melihat Data SQLite
Database (`weather_data.db`) berisi satu tabel bernama `weather_log`.
Jika Anda menggunakan Python (misalnya di Jupyter Notebook / script evaluasi model), Anda bisa mengambil datanya ke dalam format *Pandas DataFrame* dengan cara:

```python
import sqlite3
import pandas as pd

# Buat koneksi ke database SQLite
conn = sqlite3.connect('weather_data.db')

# Ambil seluruh isi tabel menjadi DataFrame
df = pd.read_sql_query("SELECT * FROM weather_log", conn)

# Tutup koneksi
conn.close()

# Tampilkan 5 data teratas
print(df.head())
```

Atau jika ingin melihat isi database secara langsung via terminal (pastikan sqlite3 ter-install):
```bash
sqlite3 weather_data.db "SELECT * FROM weather_log;"
```
