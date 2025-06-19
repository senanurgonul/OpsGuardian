from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.sklearn

def retrain_all_models():
    # Veri yükle
    df = pd.read_csv("data/logs.csv")

    # 1. ANOMALY DETECTION
    X_anomaly = df[["cpu_temp", "bandwidth", "error_code"]]
    scaler = StandardScaler()
    X_anomaly_scaled = scaler.fit_transform(X_anomaly)

    iso_model = IsolationForest(contamination=0.01, random_state=42)
    iso_model.fit(X_anomaly_scaled)

    joblib.dump(iso_model, "models/anomaly_detector.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    with mlflow.start_run(run_name="AnomalyModel"):
        mlflow.sklearn.log_model(iso_model, "isolation_forest")
        mlflow.log_params({"contamination": 0.01, "features": 3})
    print("✅ Anomaly model retrained.")

    # 2. STATUS CLASSIFICATION
    le = LabelEncoder()
    df["status_encoded"] = le.fit_transform(df["status"])
    X_status = df[["cpu_temp", "bandwidth"]]
    y_status = df["status_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X_status, y_status, test_size=0.2, random_state=42)

    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, "models/xgb_model.pkl")

    with mlflow.start_run(run_name="StatusModel"):
        mlflow.sklearn.log_model(xgb_model, "xgb_model")
        mlflow.log_params({"algorithm": "XGBoost", "features": 2})
    print("✅ Status model retrained.")

default_args = {
    "start_date": datetime(2025, 6, 18),
}

with DAG(
    dag_id="daily_retrain_all_models",
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["ml", "retrain"]
) as dag:
    retrain_task = PythonOperator(
        task_id="retrain_anomaly_and_status_models",
        python_callable=retrain_all_models
    )
