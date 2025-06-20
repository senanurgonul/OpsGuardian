# OpsGuardian – AI Destekli Operasyonel Risk İzleme

**OpsGuardian**, sistemlerden gelen log verilerini analiz ederek operasyonel riskleri tespit eden, anormallikleri saptayan ve sonuçları görselleştiren bir yapay zeka izleme platformudur.

---

## Proje Amacı

- Sistem loglarını işleyerek anormal davranışları tespit etmek
- Kritik durumları sınıflandırmak ve takip etmek
- Modelleri düzenli olarak güncellemek ve loglamak
- Sonuçları Power BI ve MLflow ile görselleştirmek
- Docker ile izole ve tekrarlanabilir bir altyapı kurmak

---

## Kullanılan Teknolojiler

- Python (Pandas, Scikit-learn, XGBoost)
- Apache Airflow (zamanlama ve otomasyon)
- MLflow (model izleme)
- Power BI (görselleştirme)
- Docker & Docker Compose (servis izolasyonu)

---

## Gerçekleştirilen Adımlar

### 1. Veri Simülasyonu
- CPU sıcaklığı, bant genişliği, hata kodu ve durum içeren veriler simüle edilerek `data/logs.csv` içine kaydedildi.

### 2. Makine Öğrenmesi Modelleri
- **Anomali Tespiti**: Isolation Forest
- **Durum Sınıflandırması**: XGBoost

Modeller belirli metriklerle eğitildi ve kaydedildi.

### 3. Airflow ile Otomasyon
Airflow DAG ile günlük çalışan bir görev ile modeller yeniden eğitiliyor ve versiyonlanıyor.

### 4. MLflow Entegrasyonu
- Model eğitim süreçleri MLflow ile otomatik olarak loglandı.
- Her model eğitimi MLflow’a otomatik olarak loglanıyor: skorlar, parametreler, versiyon bilgileri.

### 5. Power BI Dashboard
Veri CSV dosyasından içe aktarıldı ve aşağıdaki grafikler hazırlandı:
- En çok anomali üreten cihazlar (bar chart)
- Anomali vs normal oranı (pie chart)
- Zaman serisi CPU sıcaklığı ve bandwidth (line chart, anomaly tooltip’li)

### 6. FastAPI Servisi
Eğitilen modelleri bir REST API üzerinden sunmak için FastAPI kuruldu.

 **/predict/anomaly** → Anomali tahmini döner  
 **/predict/status** → Durum sınıfı tahmini döner


### 7. Docker ile Servis Yapılandırması
- Airflow, veritabanı ve yardımcı servisler Docker üzerinden ayağa kaldırıldı. Böylece taşınabilir ve izole bir ortam elde edildi.

