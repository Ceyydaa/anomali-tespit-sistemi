"""
1_ML_Model/06_final_pipeline.py — Nihai İki Aşamalı Model
===========================================================
Adım 4'teki sınıflandırıcı + Adım 5'teki regresörü birleştirir.
Pipeline:
  1. Sınıflandırıcı → olasılık >= optimal_esik ise "sipariş var"
  2. Sipariş var ise Regresör → log_miktar tahmini → expm1 ile geri çevir
Çıktı: ara_ciktilar/final_tahminler.parquet

Çalıştır: python 1_ML_Model/06_final_pipeline.py
"""

import pathlib, sys, warnings, pickle
warnings.filterwarnings("ignore")

KOK = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(KOK))

import numpy as np
import pandas as pd

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"

print("=" * 65)
print("NİHAİ İKİ AŞAMALI MODEL BAŞLADI")
print("=" * 65)

# ── Modelleri Yükle ─────────────────────────────────────────────────────────
with open(ARA_DIR / "best_classifier.pkl", "rb") as f:
    clf_paket   = pickle.load(f)
with open(ARA_DIR / "best_regressor.pkl", "rb") as f:
    reg_paket   = pickle.load(f)

clf_model      = clf_paket["model"]
clf_ozellikler = clf_paket["ozellikler"]
optimal_esik   = clf_paket["optimal_esik"]
reg_model      = reg_paket["model"]
reg_ozellikler = reg_paket["ozellikler"]
reg_scaler     = reg_paket["scaler"]  # SVR durumunda dolu, değilse None

print(f"  Sınıflandırıcı : {clf_paket['model_adi']} (eşik={optimal_esik:.2f})")
print(f"  Regresör       : {reg_paket['model_adi']}")

# ── Feature Matrix Yükle ───────────────────────────────────────────────────
panel = pd.read_parquet(ARA_DIR / "feature_matrix.parquet")
print(f"  feature_matrix : {panel.shape}")

# ── Sınıflandırma Aşaması ──────────────────────────────────────────────────
print("\n  Aşama 1: Sınıflandırma...")
# Özellik sütunlarını kontrol et
clf_mevcut = [f for f in clf_ozellikler if f in panel.columns]
panel_clf  = panel[clf_mevcut].fillna(0)

clf_proba         = clf_model.predict_proba(panel_clf.values)[:, 1]
panel["siparis_proba"]   = clf_proba
panel["yeni_siparis_var"] = (clf_proba >= optimal_esik).astype(int)

n_pos = panel["yeni_siparis_var"].sum()
print(f"  Tahmin edilen sipariş var: {n_pos:,} / {len(panel):,} satır ({n_pos/len(panel)*100:.1f}%)")

# ── Regresyon Aşaması ──────────────────────────────────────────────────────
print("\n  Aşama 2: Regresyon (sipariş var olanlara)...")
panel["yeni_tahmin_miktar"] = 0.0  # varsayılan

pos_idx = panel[panel["yeni_siparis_var"] == 1].index
if len(pos_idx) > 0:
    reg_mevcut = [f for f in reg_ozellikler if f in panel.columns]
    X_reg      = panel.loc[pos_idx, reg_mevcut].fillna(0).values
    if reg_scaler is not None:
        X_reg = reg_scaler.transform(X_reg)
    log_pred   = reg_model.predict(X_reg)
    # log1p'den geri çevir
    miktar_pred = np.clip(np.expm1(log_pred), 0, None)
    panel.loc[pos_idx, "yeni_tahmin_miktar"] = miktar_pred

print(f"  Regresyon uygulanan satır: {len(pos_idx):,}")
print(f"  yeni_tahmin_miktar istat.:\n{panel['yeni_tahmin_miktar'].describe().round(2).to_string()}")

# ── Tahmin verisini de dahil et ────────────────────────────────────────────
tahmin_temiz = pd.read_parquet(ARA_DIR / "tahmin_temiz.parquet")
# forecast_date haftasına ait satırları ara
forecast_date = tahmin_temiz["forecast_date"].max()

# Forecast haftası için özellikleri panel'den çek ya da yoksa tahmin_temiz ile birleştir
# Panel'de forecast_date haftası yoksa (gelecek) → final_tahmin mevcut
# Bu durumda final_tahmin bilgisini de kayıt altına al
print(f"\n  forecast_date: {forecast_date.date()}")
print(f"  Panel hafta aralığı: {panel['Hafta'].min().date()} – {panel['Hafta'].max().date()}")

# Panel'deki son haftanın tahminlerini çıktıya ekle
son_hafta_panel = panel[panel["Hafta"] >= panel["Hafta"].max()].copy()
son_hafta_panel = son_hafta_panel[[
    "CariKod","StockKod","Hafta","ToplamMiktar",
    "siparis_var","siparis_proba","yeni_siparis_var","yeni_tahmin_miktar"
]].copy()

# final_tahmin ile birleştir
birlesik = tahmin_temiz.merge(
    son_hafta_panel,
    left_on=["cari_kod","stok_kod"],
    right_on=["CariKod","StockKod"],
    how="left",
    suffixes=("_tahmin","_panel"),
)
# Eksik yeni_tahmin için final_tahmin'i kullan
birlesik["yeni_tahmin_miktar"]  = birlesik["yeni_tahmin_miktar"].fillna(0)
birlesik["yeni_siparis_var"]    = birlesik["yeni_siparis_var"].fillna(0).astype(int)
birlesik["siparis_proba"]       = birlesik["siparis_proba"].fillna(0)

# Panel tahminlerini de kaydet
panel.to_parquet(ARA_DIR / "panel_with_predictions.parquet", index=False)
birlesik.to_parquet(ARA_DIR / "final_tahminler.parquet", index=False)

print(f"\nKaydedildi:")
print(f"  {ARA_DIR / 'panel_with_predictions.parquet'} — tam panel ({panel.shape})")
print(f"  {ARA_DIR / 'final_tahminler.parquet'} — birleşik ({birlesik.shape})")
print("\nNİHAİ PIPELINE TAMAMLANDI ✓")
