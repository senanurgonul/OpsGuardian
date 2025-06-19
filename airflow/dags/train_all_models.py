# dags/train_all_models.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.append("ml")  # ml klasörünü import edilebilir yap
from train_models import retrain_all_models

with DAG(
    dag_id="daily_retrain_all_models",
    start_date=datetime(2025, 6, 18),
    schedule_interval="@daily",
    catchup=False
) as dag:
    retrain_task = PythonOperator(
        task_id="retrain_anomaly_and_status_models",
        python_callable=retrain_all_models
    )
