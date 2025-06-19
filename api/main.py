from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# XGBoost modelini ve label haritasını yükle
xgb_model = joblib.load("models/xgb_model.pkl")
label_map = {0: "ERROR", 1: "OK", 2: "WARNING"}

# Anomali tespiti için gerekli model ve scaler
anomaly_model = joblib.load("models/anomaly_detector.pkl")
scaler = joblib.load("models/scaler.pkl")

# Ortak veri modeli
class InputData(BaseModel):
    cpu_temp: float
    bandwidth: float
    error_code: int = 500  # Anomali için kullanılır, predict için yok sayılır

# ✅ 1. XGBoost endpoint (hata tipi tahmini)
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.cpu_temp, data.bandwidth]])
    pred = xgb_model.predict(features)[0]
    return {"prediction": label_map.get(pred, "UNKNOWN")}

# ✅ 2. IsolationForest endpoint (anomaly tespiti)
@app.post("/anomaly")
def detect_anomaly(data: InputData):
    try:
        features = np.array([[data.cpu_temp, data.bandwidth, data.error_code]])
        scaled = scaler.transform(features)
        result = anomaly_model.predict(scaled)[0]
        return {"prediction": "Anomaly" if result == -1 else "Normal"}
    except Exception as e:
        return {"error": str(e)}
