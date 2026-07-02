"""
1_ML_Model/01_eda.py — Veri Keşfi (EDA)
========================================
- Temel istatistikler
- ADI / CV² hesabı (cari-stok bazlı, Syntetos-Boylan)
- Haftalık toplam satış zaman serisi grafiği
- ADI vs CV² scatter (4 bölge)
- Sipariş sıklığı histogramı
- Türkçe özet metin

Çalıştır: python 1_ML_Model/01_eda.py
"""

import pathlib, sys, warnings
warnings.filterwarnings("ignore")

KOK = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(KOK))
from veri_yukleme import yukle_tahmin, yukle_satis

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Dizinler ──────────────────────────────────────────────────────────────
CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
GRAFIK_DIR  = CALISMA_DIR / "grafikler" / "eda"
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
ARA_DIR.mkdir(parents=True, exist_ok=True)

YLGNBU = plt.cm.YlGnBu

# SBC (Syntetos-Boylan Classification) eşikleri
ADI_ESIK = 1.32
CV2_ESIK = 0.49

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

print("=" * 65)
print("EDA — VERİ KEŞFİ BAŞLADI")
print("=" * 65)

# ── Veri Yükle ─────────────────────────────────────────────────────────────
df_tahmin = yukle_tahmin()
df_satis  = yukle_satis()

forecast_date = df_tahmin["forecast_date"].max()
print(f"\nReferans forecast_date: {forecast_date.date()}")

# ── 1. Temel İstatistikler ─────────────────────────────────────────────────
print("\n── 1. Temel İstatistikler ──")
print("\n[Tahmin Verisi]")
print(df_tahmin[["f1_tahmin","f2_tahmin","final_tahmin"]].describe().round(2).to_string())
print(f"  Tekil cari_kod  : {df_tahmin['cari_kod'].nunique():,}")
print(f"  Tekil stok_kod  : {df_tahmin['stok_kod'].nunique():,}")
print(f"  Toplam satır    : {len(df_tahmin):,}")
print(f"  NaN final_tahmin: {df_tahmin['final_tahmin'].isna().sum():,}")

print("\n[Satış Verisi]")
print(df_satis[["ToplamMiktar","ToplamTutar","SiparisSatirSayisi"]].describe().round(2).to_string())
print(f"  Tekil CariKod   : {df_satis['CariKod'].nunique():,}")
print(f"  Tekil StockKod  : {df_satis['StockKod'].nunique():,}")
print(f"  Toplam satır    : {len(df_satis):,}")
print(f"  Hafta aralığı   : {df_satis['Hafta'].min().date()} – {df_satis['Hafta'].max().date()}")

# ── 2. ADI / CV² Hesabı ────────────────────────────────────────────────────
print("\n── 2. ADI / CV² Hesabı (cari-stok bazlı) ──")

# Sadece forecast_date öncesi (tarihi olan) satışlar
hist = df_satis[df_satis["Hafta"] < forecast_date].copy()

# Hafta aralığını belirle
hafta_min = hist["Hafta"].min()
hafta_max = hist["Hafta"].max()
n_hafta   = int((hafta_max - hafta_min).days / 7) + 1

# Her (CariKod, StockKod) için haftalık pivot oluştur
pivot = (
    hist.groupby(["CariKod", "StockKod", "Hafta"])["ToplamMiktar"]
    .sum()
    .unstack("Hafta", fill_value=0)
)

# ADI: Ortalama talep aralığı (demand interval)
def adi_hesapla(serisi: np.ndarray) -> float:
    """Non-zero elemanlar arasındaki ortalama aralık (hafta)."""
    nz_idx = np.where(serisi > 0)[0]
    if len(nz_idx) < 2:
        return n_hafta   # çok seyrek → büyük ADI
    aralıklar = np.diff(nz_idx)
    return float(np.mean(aralıklar))

def cv2_hesapla(serisi: np.ndarray) -> float:
    """Non-zero talepler arasındaki CV² (varyasyon katsayısı karesi)."""
    nz = serisi[serisi > 0]
    if len(nz) < 2:
        return 0.0
    return float((np.std(nz, ddof=1) / np.mean(nz)) ** 2)

vals     = pivot.values
adi_list = [adi_hesapla(row)  for row in vals]
cv2_list = [cv2_hesapla(row) for row in vals]

df_adi = pd.DataFrame({
    "CariKod" : pivot.index.get_level_values("CariKod"),
    "StockKod": pivot.index.get_level_values("StockKod"),
    "ADI"     : adi_list,
    "CV2"     : cv2_list,
})

# Talep tipi sınıflandırması (SBC eşikleri)
def talep_tipi(adi: float, cv2: float) -> str:
    if adi <= ADI_ESIK and cv2 <= CV2_ESIK:
        return "Düzgün"
    elif adi <= ADI_ESIK and cv2 > CV2_ESIK:
        return "Düzensiz"
    elif adi > ADI_ESIK and cv2 <= CV2_ESIK:
        return "Aralıklı"
    else:
        return "Yumrulu"

df_adi["TalepTipi"] = df_adi.apply(lambda r: talep_tipi(r["ADI"], r["CV2"]), axis=1)

print(df_adi["TalepTipi"].value_counts().to_string())
print(f"\nADI istatistikleri:\n{df_adi['ADI'].describe().round(3).to_string()}")
print(f"\nCV² istatistikleri:\n{df_adi['CV2'].describe().round(3).to_string()}")

# Ara çıktı: ADI tablosu
df_adi.to_parquet(ARA_DIR / "adi_cv2.parquet", index=False)
print(f"\nAra çıktı kaydedildi: {ARA_DIR / 'adi_cv2.parquet'}")

# ── Grafik 1: Haftalık Toplam Satış Zaman Serisi ───────────────────────────
print("\n── Grafik 1: Haftalık Toplam Satış Zaman Serisi ──")
haftalik = hist.groupby("Hafta")["ToplamMiktar"].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(14, 5))
renk_grad = YLGNBU(0.7)
ax1.fill_between(haftalik["Hafta"], haftalik["ToplamMiktar"],
                 alpha=0.3, color=renk_grad)
ax1.plot(haftalik["Hafta"], haftalik["ToplamMiktar"],
         color=YLGNBU(0.85), linewidth=1.8, label="Haftalık Toplam Miktar")
ax1.axvline(forecast_date, color="tomato", linestyle="--", linewidth=1.5,
            label=f"Forecast Tarihi: {forecast_date.date()}")
ax1.set_xlabel("Hafta", fontsize=12)
ax1.set_ylabel("Toplam Satış Miktarı", fontsize=12)
ax1.set_title("Haftalık Toplam Satış Miktarı", fontsize=14, fontweight="bold", pad=12)
ax1.legend(fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines[["top","right"]].set_visible(False)
fig1.tight_layout()
yol1 = GRAFIK_DIR / "01_haftalik_satis.png"
fig1.savefig(yol1); plt.close(fig1)
print(f"  Kaydedildi: {yol1}")

# ── Grafik 2: ADI vs CV² Scatter (4 bölge) ─────────────────────────────────
print("\n── Grafik 2: ADI vs CV² Scatter (Syntetos-Boylan) ──")

tip_renkler = {
    "Düzgün"  : YLGNBU(0.85),
    "Düzensiz": YLGNBU(0.65),
    "Aralıklı": YLGNBU(0.45),
    "Yumrulu" : YLGNBU(0.25),
}

# Görselleştirme için ADI/CV² değerlerini kırp (aşırı uç değerler grafiği bozmasın)
adi_p95 = df_adi["ADI"].quantile(0.95)
cv2_p95 = df_adi["CV2"].quantile(0.95)
df_goster = df_adi[
    (df_adi["ADI"] <= adi_p95 * 1.2) &
    (df_adi["CV2"] <= cv2_p95 * 1.2)
].copy()

fig2, ax2 = plt.subplots(figsize=(10, 7))
for tip, grup in df_goster.groupby("TalepTipi"):
    ax2.scatter(grup["ADI"], grup["CV2"],
                s=12, alpha=0.4, c=[tip_renkler[tip]], label=tip, rasterized=True)

ax2.axvline(ADI_ESIK, color="gray", linestyle="--", linewidth=1.3, alpha=0.8)
ax2.axhline(CV2_ESIK, color="gray", linestyle="--", linewidth=1.3, alpha=0.8)

# Bölge etiketleri
for (adi_pos, cv2_pos, metin) in [
    (ADI_ESIK * 0.45, CV2_ESIK * 0.25, "Düzgün"),
    (ADI_ESIK * 0.45, CV2_ESIK * 1.6,  "Düzensiz"),
    (ADI_ESIK * 1.5,  CV2_ESIK * 0.25, "Aralıklı"),
    (ADI_ESIK * 1.5,  CV2_ESIK * 1.6,  "Yumrulu"),
]:
    ax2.text(adi_pos, cv2_pos, metin, fontsize=11, alpha=0.6,
             ha="center", fontweight="bold",
             color=tip_renkler.get(metin, "gray"))

ax2.set_xlabel(f"ADI (Ortalama Talep Aralığı)  [Eşik: {ADI_ESIK}]", fontsize=12)
ax2.set_ylabel(f"CV²  [Eşik: {CV2_ESIK}]", fontsize=12)
ax2.set_title("ADI vs CV² — Syntetos-Boylan Talep Tipi Sınıflandırması\n"
              f"({len(df_goster):,} ürün-müşteri ikilisi, 95. yüzdelik sınırda kırpıldı)",
              fontsize=13, fontweight="bold", pad=12)
ax2.legend(title="Talep Tipi", fontsize=10)
ax2.grid(linestyle="--", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)
fig2.tight_layout()
yol2 = GRAFIK_DIR / "02_adi_cv2_scatter.png"
fig2.savefig(yol2); plt.close(fig2)
print(f"  Kaydedildi: {yol2}")

# ── Grafik 3: Sipariş Sıklığı Histogramı ───────────────────────────────────
print("\n── Grafik 3: Sipariş Sıklığı Histogramı ──")

# Sipariş sıklığı = (cari, stok) başına siparişin olduğu hafta sayısı / n_hafta
siparis_sikligi = (
    hist[hist["ToplamMiktar"] > 0]
    .groupby(["CariKod","StockKod"])["Hafta"]
    .nunique()
    .reset_index()
    .rename(columns={"Hafta": "siparis_haftasi"})
)
siparis_sikligi["sıklık"] = siparis_sikligi["siparis_haftasi"] / n_hafta

fig3, ax3 = plt.subplots(figsize=(10, 5))
n_bins = 40
counts, edges, patches = ax3.hist(
    siparis_sikligi["sıklık"], bins=n_bins, edgecolor="white", linewidth=0.5
)
# YlGnBu renk degrade
cmap = YLGNBU
norm_vals = (edges[:-1] - edges[:-1].min()) / (edges[:-1].max() - edges[:-1].min() + 1e-9)
for patch, nv in zip(patches, norm_vals):
    patch.set_facecolor(cmap(0.3 + nv * 0.6))

ax3.set_xlabel("Sipariş Sıklığı (Sipariş Olan Haftalar / Toplam Hafta)", fontsize=12)
ax3.set_ylabel("Ürün-Müşteri İkilisi Sayısı", fontsize=12)
ax3.set_title("Sipariş Sıklığı Dağılımı", fontsize=14, fontweight="bold", pad=12)
ax3.grid(axis="y", linestyle="--", alpha=0.4)
ax3.spines[["top","right"]].set_visible(False)
fig3.tight_layout()
yol3 = GRAFIK_DIR / "03_siparis_sikligi_hist.png"
fig3.savefig(yol3); plt.close(fig3)
print(f"  Kaydedildi: {yol3}")

# ── Bulgular Özet ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EDA — BULGULAR ÖZETİ")
print("=" * 65)
tip_sayilari = df_adi["TalepTipi"].value_counts()
en_fazla = tip_sayilari.idxmax()
print(f"""
Toplam analiz edilen ürün-müşteri ikilisi : {len(df_adi):,}
Veri seti hafta aralığı                   : {n_hafta} hafta

Talep Tipi Dağılımı (SBC eşikleri: ADI>{ADI_ESIK}, CV²>{CV2_ESIK}):
{tip_sayilari.to_string()}

  ▸ En baskın talep tipi: '{en_fazla}' ({tip_sayilari[en_fazla]/len(df_adi)*100:.1f}%)

  ▸ Tahmin verisinde {df_tahmin['final_tahmin'].isna().sum():,} satırda final_tahmin boştur.

  ▸ Sipariş sıklığı analizi:
    - Medyan sıklık  : {siparis_sikligi['sıklık'].median():.2f} (haftaların {siparis_sikligi['sıklık'].median()*100:.1f}%'inde sipariş var)
    - Hiç sipariş yok (sıklık=0): {(siparis_sikligi['sıklık']==0).sum():,} ikili

  ▸ Kaydedilen grafikler:
    - {yol1.name}
    - {yol2.name}
    - {yol3.name}
""")
print("EDA TAMAMLANDI ✓")
