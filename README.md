# OpsGuardian – AI-Powered Operational Risk Monitoring

**OpsGuardian** is an AI-based monitoring platform that analyzes log data from systems to detect operational risks, identify anomalies, and visualize the results.

---

## Project Goals

- Detect abnormal behavior by processing system logs  
- Classify and monitor critical conditions  
- Regularly update and log machine learning models  
- Visualize results using Power BI and MLflow  
- Create an isolated and reproducible infrastructure using Docker  

---

## Technologies Used

- Python (Pandas, Scikit-learn, XGBoost)  
- Apache Airflow (scheduling and automation)  
- MLflow (model tracking)  
- Power BI (data visualization)  
- Docker & Docker Compose (service isolation)  

---

## Implemented Steps

### 1. Data Simulation  
Simulated data including CPU temperature, bandwidth, error codes, and statuses were saved into `data/logs.csv`.

### 2. Machine Learning Models  
- **Anomaly Detection**: Isolation Forest  
- **Status Classification**: XGBoost  

Models were trained using defined metrics and saved for inference.

### 3. Automation with Airflow  
A daily Airflow DAG automates model retraining and versioning.

### 4. MLflow Integration  
- Training processes are automatically logged to MLflow  
- Each run includes metrics, parameters, and version info  

### 5. Power BI Dashboard  
CSV data was imported to create the following visualizations:
- Top anomaly-generating devices (bar chart)  
- Anomaly vs normal distribution (pie chart)  
- Time series of CPU temperature and bandwidth (line chart with anomaly tooltips)

### 6. FastAPI Service  
Trained models are served via a REST API built with FastAPI:

- `/predict/anomaly` → Returns anomaly prediction  
- `/predict/status` → Returns status classification  

### 7. Docker-Based Architecture  
Airflow, database, and supporting services run in isolated containers via Docker for a portable and reproducible setup.
