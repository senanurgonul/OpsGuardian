import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# 1. CSV verisini oku
df = pd.read_csv("data/logs.csv")

# 2. Özellik seçimi
X = df[["cpu_temp", "bandwidth", "error_code"]]

# 3. StandardScaler ile normalizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Anomali modelini oluştur ve eğit
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_scaled)

# 5. Eğitilen modelleri diske kaydet
joblib.dump(model, "models/anomaly_detector.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Anomali modeli ve scaler başarıyla kaydedildi.")
