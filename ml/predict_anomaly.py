import sys
import numpy as np
import pandas as pd
import joblib

# 1. Model ve scaler'ı yükle
model = joblib.load("models/anomaly_detector.pkl")
scaler = joblib.load("models/scaler.pkl")

# 2. Tekil tahmin fonksiyonu
def predict_anomaly(cpu_temp, bandwidth, error_code):
    X = np.array([[cpu_temp, bandwidth, error_code]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return "Anomaly" if pred == -1 else "Normal"

# 3. Tüm veri setini analiz eden fonksiyon
def analyze_entire_dataset():
    df = pd.read_csv("data/logs.csv")

    # Eğer anomaly sütunu zaten varsa, yeniden hesaplamadan çık (isteğe bağlı)
    if "anomaly" in df.columns:
        print("✅ Anomaly sütunu zaten mevcut. Güncelleme yapılmadı.")
        return

    X = df[["cpu_temp", "bandwidth", "error_code"]]
    X_scaled = scaler.transform(X)
    df["anomaly"] = model.predict(X_scaled)
    df["anomaly"] = df["anomaly"].map({1: "normal", -1: "anomaly"})

    # ✅ Anomaly sütununu CSV'ye kaydet
    df.to_csv("data/logs.csv", index=False)

    print("📊 First 10 anomalies found in the dataset:\n")
    print(df[df["anomaly"] == "anomaly"].head(10))

# 4. Ana çalışma bloğu
if __name__ == "__main__":
    if len(sys.argv) == 4:
        try:
            cpu_temp = float(sys.argv[1])
            bandwidth = float(sys.argv[2])
            error_code = int(sys.argv[3])
            result = predict_anomaly(cpu_temp, bandwidth, error_code)
            print(f"✅ Prediction for input: {result}")
        except ValueError:
            print("⚠️ Hatalı giriş. Örn: python predict_anomaly.py 65.5 200.0 500")
    else:
        analyze_entire_dataset()
