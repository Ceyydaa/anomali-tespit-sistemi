# Talep Tahmin Doğruluk Analizi ve Anomali Tespit Sistemi

FMCG/içecek sektörü için geliştirilen iki bölümlü analiz projesi.

## Proje Özeti

Mevcut tahmin modelinin (final_tahmin) doğruluğunu ölçmek, hata
türlerini sınıflandırmak ve daha iyi bir alternatif model geliştirmek
amacıyla oluşturulmuştur.

## Klasör Yapısı

## Kullanılan Yöntemler

**Makine Öğrenmesi:**
- Sınıflandırma: LightGBM, XGBoost, CatBoost, Random Forest, Logistic Regression
- Regresyon: ElasticNet, LightGBM (Tweedie), CatBoost, XGBoost
- Optimizasyon: Optuna (TPE sampler, 20 deneme, MedianPruner)
- Metrikler: Precision, Recall, F1, ROC-AUC, MAAPE, RMSE, Bias

**İstatistiksel Analiz:**
IQR, Z-Skor, MAD (Modifiye Z-Skor), UCL/LCL (SPC),
CUSUM, Benford Yasası, Mann-Kendall Trend Testi,
KS Testi, Shannon Entropy, Runs Testi

## Veri

Veri dosyaları gizlilik nedeniyle repository'de bulunmamaktadır.
- faz1_faz2_anonim.xlsx (711k satır, tahmin verisi)
- TblGecmisSatisVerileri_haftalik.csv (879k satır, gerçekleşen satış)

## Kurulum

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost \
            optuna scipy pymannkendall statsmodels openpyxl
```

## Çalıştırma Sırası

**1_ML_Model:**
```bash
python 1_ML_Model/01_eda.py
python 1_ML_Model/02_temizleme.py
python 1_ML_Model/03_feature_engineering.py
python 1_ML_Model/04_classification.py
python 1_ML_Model/05_regression.py
python 1_ML_Model/06_final_pipeline.py
python 1_ML_Model/07_evaluation.py
python 1_ML_Model/08_anomali.py
python 1_ML_Model/09_excel_rapor.py
python 1_ML_Model/10_gorseller.py
```

**2_Kural_Tabanli_Denetim:**
```bash
python 2_Kural_Tabanli_Denetim/denetim.py
```
