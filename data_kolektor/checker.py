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