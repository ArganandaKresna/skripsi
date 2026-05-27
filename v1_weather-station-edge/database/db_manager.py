import sqlite3
import os

class DBManager:
    def __init__(self, db_path="database/weather_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.setup_tables()

    def setup_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                suhu REAL,
                kelembaban REAL,
                tekanan REAL,
                angin REAL,
                hujan REAL,
                prediksi INTEGER
            )
        """)
        self.conn.commit()

    def insert_weather(self, timestamp, suhu, kelembaban, tekanan, angin, hujan, prediksi):
        self.cursor.execute("""
            INSERT INTO weather (timestamp, suhu, kelembaban, tekanan, angin, hujan, prediksi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, suhu, kelembaban, tekanan, angin, hujan, prediksi))
        self.conn.commit()

    def close(self):
        self.conn.close()
