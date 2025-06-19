# data/generate_data.py

import pandas as pd
import random
from datetime import datetime, timedelta

now = datetime.now()
rows = []

for i in range(100000):  # 100 bin satır veri üret
    ts = now - timedelta(seconds=i * 10)  # Her kayıt 10 saniye aralıklı
    cpu = round(random.uniform(40, 100), 2)  # CPU sıcaklığı
    bw = round(random.uniform(10, 1000), 2)  # Bant genişliği
    status = random.choices(["OK", "WARNING", "ERROR"], weights=[0.7, 0.2, 0.1])[0]
    
    rows.append([
        ts,
        f"device_{i % 20}",  # 20 farklı cihaz
        random.randint(100, 999),  # Hata kodu
        cpu,
        bw,
        status
    ])

# Veriyi DataFrame'e çevir
df = pd.DataFrame(rows, columns=[
    "timestamp", "device_id", "error_code", "cpu_temp", "bandwidth", "status"
])

# CSV olarak kaydet
df.to_csv("data/logs.csv", index=False)

print("✔️ Simülasyon verisi üretildi: data/logs.csv")
