"""
1_ML_Model/03_feature_engineering.py — Özellik Mühendisliği
=============================================================
- Lag özellikleri (t-1, t-2, t-4, t-8)
- Rolling istatistikler (4/8/12 hafta ortalama & std)
- Recency: son siparişten geçen hafta sayısı
- SBC talep tipi (eşik kuralı, kümeleme yok)
- Frequency / Target encoding
- Takvim özellikleri
- Çıktı: ara_ciktilar/feature_matrix.parquet

Çalıştır: python 1_ML_Model/03_feature_engineering.py
"""

import pathlib, sys, warnings
warnings.filterwarnings("ignore")

KOK = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(KOK))

import numpy as np
import pandas as pd

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"

# SBC eşikleri
ADI_ESIK = 1.32
CV2_ESIK = 0.49

print("=" * 65)
print("ÖZELLİK MÜHENDİSLİĞİ BAŞLADI")
print("=" * 65)

# ── Temiz verileri yükle ────────────────────────────────────────────────────
hist_temiz   = pd.read_parquet(ARA_DIR / "hist_temiz.parquet")
tahmin_temiz = pd.read_parquet(ARA_DIR / "tahmin_temiz.parquet")
adi_cv2      = pd.read_parquet(ARA_DIR / "adi_cv2.parquet")

print(f"hist_temiz   : {hist_temiz.shape}")
print(f"tahmin_temiz : {tahmin_temiz.shape}")
print(f"adi_cv2      : {adi_cv2.shape}")

forecast_date = tahmin_temiz["forecast_date"].max()

# ── Temel panel veri yapısı ─────────────────────────────────────────────────
# Her (CariKod, StockKod, Hafta) için tek satır (haftalık panel)
panel = hist_temiz.sort_values(["CariKod","StockKod","Hafta"]).copy()
panel["siparis_var"] = (panel["ToplamMiktar"] > 0).astype(int)

# ── 1. Lag Özellikleri ──────────────────────────────────────────────────────
print("\n── 1. Lag Özellikleri ──")
for lag in [1, 2, 4, 8]:
    panel[f"lag_{lag}"] = (
        panel.groupby(["CariKod","StockKod"])["log_miktar"]
        .shift(lag)
    )
    panel[f"lag_siparis_{lag}"] = (
        panel.groupby(["CariKod","StockKod"])["siparis_var"]
        .shift(lag)
    )
print("  lag_1, lag_2, lag_4, lag_8 oluşturuldu")

# ── 2. Rolling İstatistikler ────────────────────────────────────────────────
print("\n── 2. Rolling Ortalama/Std ──")
for pencere in [4, 8, 12]:
    roll_grp = panel.groupby(["CariKod","StockKod"])["log_miktar"]
    panel[f"roll_ort_{pencere}"] = (
        roll_grp.shift(1)
        .transform(lambda x: x.rolling(pencere, min_periods=1).mean())
    )
    panel[f"roll_std_{pencere}"] = (
        roll_grp.shift(1)
        .transform(lambda x: x.rolling(pencere, min_periods=2).std().fillna(0))
    )
    # Ham miktar rolling (sıklık)
    panel[f"roll_siklık_{pencere}"] = (
        panel.groupby(["CariKod","StockKod"])["siparis_var"]
        .shift(1)
        .transform(lambda x: x.rolling(pencere, min_periods=1).mean())
    )
print("  roll_ort/std/sıklık (4/8/12 hafta) oluşturuldu")

# ── 3. Recency ──────────────────────────────────────────────────────────────
print("\n── 3. Recency (son siparişten geçen hafta) ──")

def recency_hesapla(grup):
    """Her haftada, o haftaya kadar son siparişten kaçıncı haftada olduğumuzu döndür."""
    son_siparis = None
    recency = []
    for idx, row in grup.iterrows():
        if son_siparis is None:
            recency.append(np.nan)
        else:
            hafta_fark = (row["Hafta"] - son_siparis).days / 7
            recency.append(hafta_fark)
        if row["siparis_var"] == 1:
            son_siparis = row["Hafta"]
    return pd.Series(recency, index=grup.index)

panel["recency_hafta"] = (
    panel.groupby(["CariKod","StockKod"], group_keys=False)
    .apply(recency_hesapla)
)
# NaN → o üründe ilk defa sipariş (ya da hiç yok) → büyük bir değer
max_recency = panel["recency_hafta"].quantile(0.99)
panel["recency_hafta"] = panel["recency_hafta"].fillna(max_recency)
print(f"  recency_hafta medyanı: {panel['recency_hafta'].median():.1f} hafta")

# ── 4. ADI / CV² ve SBC Talep Tipi ─────────────────────────────────────────
print("\n── 4. SBC Talep Tipi (eşik kuralı) ──")

panel = panel.merge(
    adi_cv2[["CariKod","StockKod","ADI","CV2","TalepTipi"]],
    on=["CariKod","StockKod"], how="left"
)
panel["ADI"]       = panel["ADI"].fillna(panel["ADI"].median())
panel["CV2"]       = panel["CV2"].fillna(panel["CV2"].median())
panel["TalepTipi"] = panel["TalepTipi"].fillna("Bilinmiyor")

tip_kodlari = {"Düzgün": 0, "Düzensiz": 1, "Aralıklı": 2, "Yumrulu": 3, "Bilinmiyor": 4}
panel["TalepTipiKod"] = panel["TalepTipi"].map(tip_kodlari)
print(panel["TalepTipi"].value_counts().to_string())

# ── 5. Encoding ─────────────────────────────────────────────────────────────
print("\n── 5. Encoding ──")

# Frequency encoding: CariKod, StockKod
cari_freq  = panel["CariKod"].value_counts(normalize=True).rename("cari_freq")
stok_freq  = panel["StockKod"].value_counts(normalize=True).rename("stok_freq")
panel["cari_freq"] = panel["CariKod"].map(cari_freq)
panel["stok_freq"] = panel["StockKod"].map(stok_freq)

# Target encoding (log_miktar hedefi üzerinden, leakage önlemek için train setinde yapılacak)
# Burada global mean kullanıyoruz (proxy, gerçek TE adım 1.4'te yapılacak)
cari_target  = panel.groupby("CariKod")["log_miktar"].mean().rename("cari_target_enc")
stok_target  = panel.groupby("StockKod")["log_miktar"].mean().rename("stok_target_enc")
panel["cari_target_enc"] = panel["CariKod"].map(cari_target)
panel["stok_target_enc"] = panel["StockKod"].map(stok_target)
print("  Frequency encoding: cari_freq, stok_freq")
print("  Target encoding (global proxy): cari_target_enc, stok_target_enc")

# ── 6. Takvim Özellikleri ───────────────────────────────────────────────────
print("\n── 6. Takvim Özellikleri ──")
panel["ay"]            = panel["Hafta"].dt.month
panel["ceyrek"]        = panel["Hafta"].dt.quarter
panel["hafta_no"]      = panel["Hafta"].dt.isocalendar().week.astype(int)
panel["yil"]           = panel["Hafta"].dt.year
panel["yil_basi"]      = (panel["hafta_no"] <= 4).astype(int)
panel["yil_sonu"]      = (panel["hafta_no"] >= 49).astype(int)
panel["ay_basi"]       = (panel["Hafta"].dt.day <= 7).astype(int)
print("  ay, ceyrek, hafta_no, yil, yil_basi, yil_sonu, ay_basi oluşturuldu")

# ── Kaydet ─────────────────────────────────────────────────────────────────
panel.to_parquet(ARA_DIR / "feature_matrix.parquet", index=False)
print(f"\nfeature_matrix: {panel.shape}")
print(f"  NaN oranı özet:\n{(panel.isna().mean() * 100).round(2).to_string()}")
print(f"\nKaydedildi: {ARA_DIR / 'feature_matrix.parquet'}")
print("\nÖZELLİK MÜHENDİSLİĞİ TAMAMLANDI ✓")
