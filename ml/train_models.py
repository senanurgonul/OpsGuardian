# ml/train_models.py
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

def retrain_all_models():
    df = pd.read_csv("data/logs.csv")

    # Anomaly Detection
    X_anomaly = df[["cpu_temp", "bandwidth", "error_code"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomaly)

    anomaly_model = IsolationForest(contamination=0.01, random_state=42)
    anomaly_model.fit(X_scaled)

    joblib.dump(anomaly_model, "models/anomaly_detector.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("OpsGuardian-Anomaly")

    with mlflow.start_run(run_name="anomaly_detector_run"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("features", ["cpu_temp", "bandwidth", "error_code"])
        mlflow.sklearn.log_model(anomaly_model, "anomaly_detector_model")

    # Status Classification
    df["status_encoded"] = LabelEncoder().fit_transform(df["status"])
    X_status = df[["cpu_temp", "bandwidth"]]
    y_status = df["status_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X_status, y_status, test_size=0.2, random_state=42)

    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, "models/xgb_model.pkl")

    mlflow.set_experiment("OpsGuardian-Status")
    with mlflow.start_run(run_name="xgb_status_run"):
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_metric("accuracy", xgb_model.score(X_test, y_test))
        mlflow.sklearn.log_model(xgb_model, "xgb_status_model")

    print("✅ Tüm modeller başarıyla yeniden eğitildi ve MLflow'a loglandı.")

if __name__ == "__main__":
    retrain_all_models()
