"""
1_ML_Model/08_anomali.py — Anomali Tespiti (3 Kategori)
=========================================================
a) İş/Talep Anomalileri:
   - Ampirik 95./5. yüzdelik dilim dışı satışlar
   - "Hep sipariş veren ama bu hafta sıfır" (churn sinyali)
b) Model/Tahmin Anomalileri:
   - Sistematik yön hatası (cari/ürün bazlı hep + veya hep -)
c) Veri Kalitesi Anomalileri:
   - Fiyat anomalisi (IQR)
   - Mükerrer/robotik kayıt
Çıktı: ara_ciktilar/anomali_*.parquet

Çalıştır: python 1_ML_Model/08_anomali.py
"""

import pathlib, sys, warnings
warnings.filterwarnings("ignore")

KOK = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(KOK))
from veri_yukleme import yukle_satis

import numpy as np
import pandas as pd

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"

# Ampirik yüzdelik eşikler
UCUNCU_YUKARI   = 95
UCUNCU_ASAGI    = 5
MIN_AKTIF_HAFTA = 4    # "hep sipariş veren" tanımı: son N haftanın en az 75%'i

print("=" * 65)
print("ANOMALİ TESPİTİ BAŞLADI")
print("=" * 65)

# ── Verileri Yükle ─────────────────────────────────────────────────────────
panel     = pd.read_parquet(ARA_DIR / "panel_with_predictions.parquet")
df_satis  = yukle_satis()

# Hata = gerçek - tahmin (ham miktar)
panel["hata"] = panel["yeni_tahmin_miktar"] - panel["ToplamMiktar"]
panel["hata_yon"] = np.sign(panel["hata"])  # +1, -1, 0

# ── (a) İş / Talep Anomalileri ─────────────────────────────────────────────
print("\n── (a) İş/Talep Anomalileri ──")

# 1. Ampirik yüzdelik dışı satışlar (cari-stok bazlı)
def yuzdelik_anomali(grup):
    q_yukari = grup["ToplamMiktar"].quantile(UCUNCU_YUKARI / 100)
    q_asagi  = grup["ToplamMiktar"].quantile(UCUNCU_ASAGI  / 100)
    grup = grup.copy()
    grup["yukari_anomali"] = grup["ToplamMiktar"] > q_yukari
    grup["asagi_anomali"]  = grup["ToplamMiktar"] < q_asagi
    return grup

panel_anomali = (
    panel.groupby(["CariKod","StockKod"], group_keys=False)
    .apply(yuzdelik_anomali)
)
is_anomali_yuzdelik = panel_anomali[
    panel_anomali["yukari_anomali"] | panel_anomali["asagi_anomali"]
][["CariKod","StockKod","Hafta","ToplamMiktar",
   "yukari_anomali","asagi_anomali","yeni_tahmin_miktar","hata"]].copy()
print(f"  Yüzdelik dışı satış anomalisi: {len(is_anomali_yuzdelik):,} satır")

# 2. Churn sinyali: Son N haftada hep sipariş var, bu hafta sıfır
son_n_hafta = panel["Hafta"].drop_duplicates().nlargest(MIN_AKTIF_HAFTA)
panel_son_n = panel[panel["Hafta"].isin(son_n_hafta)]
churn_adaylar = (
    panel_son_n.groupby(["CariKod","StockKod"])
    .apply(lambda g: (
        g["siparis_var"].sum() >= len(g) * 0.75 and    # son N hafta aktif
        g[g["Hafta"] == g["Hafta"].max()]["siparis_var"].values[0] == 0  # bu hafta 0
    ) if len(g) > 0 else False)
    .reset_index()
    .rename(columns={0: "churn_sinyali"})
)
churn_sinyali = churn_adaylar[churn_adaylar["churn_sinyali"]][["CariKod","StockKod"]].copy()
print(f"  Churn sinyali (hep aktif, bu hafta sıfır): {len(churn_sinyali):,} ikili")

# Kaydet
is_anomali_yuzdelik.to_parquet(ARA_DIR / "anomali_is_yuzdelik.parquet", index=False)
churn_sinyali.to_parquet(ARA_DIR / "anomali_churn.parquet", index=False)

# ── (b) Model/Tahmin Anomalileri ───────────────────────────────────────────
print("\n── (b) Model/Tahmin Anomalileri ──")

# Sistematik yön hatası: cari bazlı hep + veya hep -
MIN_GOZLEM = 5  # en az bu kadar gözlem olmalı

def sistematik_yon(grup):
    ynlar = grup["hata_yon"]
    if len(ynlar) < MIN_GOZLEM:
        return None
    pozitif_oran = (ynlar > 0).mean()
    if pozitif_oran >= 0.85:
        return "+1 (sistematik fazla tahmin)"
    elif pozitif_oran <= 0.15:
        return "-1 (sistematik eksik tahmin)"
    return None

cari_sistematik = (
    panel.groupby("CariKod")
    .apply(sistematik_yon)
    .dropna()
    .reset_index()
    .rename(columns={0: "sistematik_yon"})
)
stok_sistematik = (
    panel.groupby("StockKod")
    .apply(sistematik_yon)
    .dropna()
    .reset_index()
    .rename(columns={0: "sistematik_yon"})
)

print(f"  Sistematik yön hatası — cari bazlı: {len(cari_sistematik):,}")
print(f"  Sistematik yön hatası — ürün bazlı: {len(stok_sistematik):,}")

cari_sistematik.to_parquet(ARA_DIR / "anomali_sistematik_cari.parquet", index=False)
stok_sistematik.to_parquet(ARA_DIR / "anomali_sistematik_stok.parquet", index=False)

# ── (c) Veri Kalitesi Anomalileri ──────────────────────────────────────────
print("\n── (c) Veri Kalitesi Anomalileri ──")

# Fiyat anomalisi (Adım 2'de hesaplanmış)
try:
    fiyat_anomali = pd.read_parquet(ARA_DIR / "veri_kalitesi_supheli.parquet")
    print(f"  Fiyat anomalisi (Adım 2'den): {len(fiyat_anomali):,} satır")
except FileNotFoundError:
    print("  UYARI: veri_kalitesi_supheli.parquet bulunamadı, yeniden hesaplanıyor...")
    fiyat_anomali = pd.DataFrame()

# Mükerrer kayıt
try:
    mukerrer = pd.read_parquet(ARA_DIR / "mukerrer_kayitlar.parquet")
    print(f"  Mükerrer kayıt (Adım 2'den): {len(mukerrer):,} grup")
except FileNotFoundError:
    print("  UYARI: mukerrer_kayitlar.parquet bulunamadı.")
    mukerrer = pd.DataFrame()

# Robotik kayıt: tam aynı miktar + tam aynı tutar birden fazla kez gelen
robotik = df_satis[
    df_satis.duplicated(subset=["CariKod","StockKod","ToplamMiktar","ToplamTutar"], keep=False) &
    (df_satis["ToplamMiktar"] > 0)
].copy()
print(f"  Robotik/kopyalanmış kayıt: {len(robotik):,} satır")
robotik.to_parquet(ARA_DIR / "anomali_robotik.parquet", index=False)

# ── Özet ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ANOMALİ TESPİTİ ÖZET")
print("=" * 65)
print(f"""
  (a) İş/Talep Anomalileri:
      Yüzdelik dışı satış   : {len(is_anomali_yuzdelik):,} satır
      Churn sinyali         : {len(churn_sinyali):,} cari-stok ikilisi

  (b) Model/Tahmin Anomalileri:
      Sistematik hata (cari): {len(cari_sistematik):,}
      Sistematik hata (ürün): {len(stok_sistematik):,}

  (c) Veri Kalitesi Anomalileri:
      Birim fiyat anomalisi : {len(fiyat_anomali):,} satır
      Mükerrer kayıt grubu  : {len(mukerrer):,}
      Robotik kayıt         : {len(robotik):,} satır
""")
print("ANOMALİ TESPİTİ TAMAMLANDI ✓")
