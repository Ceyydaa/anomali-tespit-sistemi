"""
1_ML_Model/02_temizleme.py — Veri Temizleme
=============================================
- Eksik f1_tahmin → 0
- Birim fiyat anomalisi tespiti (IQR), veri_kalitesi_supheli listesi
- log1p dönüşümü (yalnızca model girdisi için)
- Çıktı: ara_ciktilar/temiz_veri.parquet

Çalıştır: python 1_ML_Model/02_temizleme.py
"""

import pathlib, sys, warnings
warnings.filterwarnings("ignore")

KOK = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(KOK))
from veri_yukleme import yukle_tahmin, yukle_satis

import numpy as np
import pandas as pd

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"
ARA_DIR.mkdir(parents=True, exist_ok=True)

# Birim fiyat IQR çarpanı (bu değerin üstü/altı anomali)
FIYAT_IQR_CARPAN = 3.0

print("=" * 60)
print("VERİ TEMİZLEME BAŞLADI")
print("=" * 60)

df_tahmin = yukle_tahmin()
df_satis  = yukle_satis()

forecast_date = df_tahmin["forecast_date"].max()
hist = df_satis[df_satis["Hafta"] < forecast_date].copy()

# ── 1. Eksik f1_tahmin → 0 ─────────────────────────────────────────────────
n_eksik_f1 = df_tahmin["f1_tahmin"].isna().sum()
df_tahmin["f1_tahmin"] = df_tahmin["f1_tahmin"].fillna(0)
print(f"\n[f1_tahmin] {n_eksik_f1:,} eksik → 0 ile dolduruldu")

# ── 2. Birim Fiyat Anomalisi ────────────────────────────────────────────────
print("\n── Birim Fiyat Anomalisi Tespiti (IQR yöntemi) ──")

hist_fiyat = hist[
    (hist["ToplamMiktar"] > 0) & (hist["ToplamTutar"] > 0)
].copy()
hist_fiyat["birim_fiyat"] = hist_fiyat["ToplamTutar"] / hist_fiyat["ToplamMiktar"]

# CariKod + StockKod bazlı IQR
def iqr_anomali(grup):
    q1, q3 = grup["birim_fiyat"].quantile([0.25, 0.75])
    iqr     = q3 - q1
    alt     = q1 - FIYAT_IQR_CARPAN * iqr
    ust     = q3 + FIYAT_IQR_CARPAN * iqr
    grup["alt_sinir"] = alt
    grup["ust_sinir"] = ust
    grup["fiyat_anomali"] = (
        (grup["birim_fiyat"] < alt) | (grup["birim_fiyat"] > ust)
    )
    return grup

hist_fiyat = (
    hist_fiyat.groupby(["CariKod","StockKod"], group_keys=False)
    .apply(iqr_anomali)
)

veri_kalitesi_supheli = hist_fiyat[hist_fiyat["fiyat_anomali"]].copy()
n_anomali = len(veri_kalitesi_supheli)
print(f"  Birim fiyat anomalisi tespit edildi: {n_anomali:,} satır")
print(f"  Etkilenen tekil stok_kod : {veri_kalitesi_supheli['StockKod'].nunique():,}")

# Mükerrer kayıt (aynı CariKod-StockKod-Hafta birden fazla satır)
mukerrer = (
    hist.groupby(["CariKod","StockKod","Hafta"]).size().reset_index(name="satir_sayisi")
)
mukerrer = mukerrer[mukerrer["satir_sayisi"] > 1].copy()
print(f"  Mükerrer kayıt tespiti   : {len(mukerrer):,} (cari-stok-hafta) grubu")

veri_kalitesi_supheli.to_parquet(ARA_DIR / "veri_kalitesi_supheli.parquet", index=False)
mukerrer.to_parquet(ARA_DIR / "mukerrer_kayitlar.parquet", index=False)
print(f"  Kaydedildi: veri_kalitesi_supheli.parquet, mukerrer_kayitlar.parquet")

# ── 3. log1p Dönüşümü ──────────────────────────────────────────────────────
print("\n── log1p Dönüşümü (model girdisi için) ──")

hist_temiz = hist.copy()
# Haftalık özet: duplicate haftalarda toplama yap
hist_temiz = (
    hist_temiz.groupby(["CariKod","StockKod","Hafta"])[["ToplamMiktar","ToplamTutar"]]
    .sum()
    .reset_index()
)
hist_temiz["log_miktar"] = np.log1p(hist_temiz["ToplamMiktar"])
hist_temiz["log_tutar"]  = np.log1p(hist_temiz["ToplamTutar"])

# Tahmin dosyası temizleme
df_tahmin_temiz = df_tahmin.copy()
for kol in ["f1_tahmin","f2_tahmin","final_tahmin"]:
    df_tahmin_temiz[f"log_{kol}"] = np.log1p(
        df_tahmin_temiz[kol].fillna(0).clip(lower=0)
    )

# Kaydet
hist_temiz.to_parquet(ARA_DIR / "hist_temiz.parquet", index=False)
df_tahmin_temiz.to_parquet(ARA_DIR / "tahmin_temiz.parquet", index=False)

print(f"  hist_temiz    : {hist_temiz.shape} → {ARA_DIR/'hist_temiz.parquet'}")
print(f"  tahmin_temiz  : {df_tahmin_temiz.shape} → {ARA_DIR/'tahmin_temiz.parquet'}")

# ── Doğrulama ──────────────────────────────────────────────────────────────
print("\n── Doğrulama ──")
print(f"  hist_temiz NaN log_miktar: {hist_temiz['log_miktar'].isna().sum()}")
print(f"  tahmin_temiz NaN log_final_tahmin: {df_tahmin_temiz['log_final_tahmin'].isna().sum()}")
print(f"  f1_tahmin NaN kaldı       : {df_tahmin_temiz['f1_tahmin'].isna().sum()}")

print("\nVERİ TEMİZLEME TAMAMLANDI ✓")
