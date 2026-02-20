import csv
import json
import os
import sqlite3
from pathlib import Path


def create_data(path):
    # Define paths
    data_dir = Path(path)
    data_dir.mkdir(exist_ok=True)

    sqlite_path = data_dir / 'invoice.sqlite3'
    csv_path = data_dir / 'invoice.csv'
    json_path = data_dir / 'invoice.json'

    # Remove existing files if they exist
    for path in [sqlite_path, csv_path, json_path]:
        if os.path.exists(path):
            os.remove(path)

    # Sample data with versioned products
    data = [
        ('RES-CF-10K-0.25W-5%', 100, 0.05, '2024-01-01'),
        ('CAP-ELEC-100UF-25V', 50, 0.25, '2024-01-02'),
        ('MCU-ARDUINO-UNO-R3', 5, 22.50, '2024-01-03'),
        ('MCU-ARDUINO-UNO-R3-v1', 3, 21.00, '2024-01-04'),
        ('MCU-ARDUINO-UNO-R3-v2', 7, 24.00, '2024-01-05'),
        ('LED-5MM-RED-20MA', 200, 0.10, '2024-01-06'),
        ('LED-5MM-RED-20MA-v1', 150, 0.09, '2024-01-07'),
        ('MCU-ESP32-DEVKIT-V1', 10, 8.75, '2024-01-08'),
        ('MCU-ESP32-DEVKIT-V2', 8, 9.25, '2024-01-09'),
        ('TRANS-NPN-2N2222A-TO92', 150, 0.15, '2024-01-10'),
        ('SBC-RPI4-4GB-MODEL-B', 3, 45.00, '2024-01-11'),
        ('SBC-RPI4-4GB-MODEL-B-v1', 2, 44.00, '2024-01-12'),
        ('SBC-RPI4-4GB-MODEL-B-v2', 5, 47.00, '2024-01-13'),
        ('PROTO-BB-830-TIE-PT', 20, 3.50, '2024-01-14'),
        ('WIRE-JMP-MM-65PCS-KIT', 10, 5.25, '2024-01-15'),
    ]

    columns = ['product_name', 'quantity', 'price', 'purchase_date']

    # Create SQLite database
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE purchases (
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            purchase_date TEXT NOT NULL
        )
    ''')

    cursor.executemany('INSERT INTO purchases VALUES (?, ?, ?, ?)', data)
    conn.commit()
    conn.close()

    # Create CSV (without header)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    # Create JSON (list of row dicts)
    json_data = [dict(zip(columns, row, strict=True)) for row in data]

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    return [sqlite_path, csv_path, json_path]
