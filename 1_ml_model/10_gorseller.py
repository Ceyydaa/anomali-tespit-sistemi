"""
1_ML_Model/10_gorseller.py — Görselleştirmeler (Toplantı Seti)
===============================================================
YlGnBu renk paleti, Türkçe etiketler
Grafikler:
  1. ADI/CV² haritası (Syntetos-Boylan 4 bölge)
  2. Confusion matrix — final_tahmin vs yeni_tahmin yan yana
  3. Tahmin-vs-gerçek scatter (log-log) — iki model yan yana
  4. Zaman serisi + anomali noktaları işaretli
  5. Anomali kategori bar chart
  6. 7 metrik eski-yeni karşılaştırma bar chart

Çalıştır: python 1_ML_Model/10_gorseller.py
"""

import pathlib, sys, warnings
warnings.filterwarnings("ignore")

KOK = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(KOK))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"
GRAFIK_DIR  = CALISMA_DIR / "grafikler"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)

YLGNBU = plt.cm.YlGnBu
ADI_ESIK = 1.32
CV2_ESIK = 0.49

plt.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi"     : 150,
    "savefig.dpi"    : 150,
    "savefig.bbox"   : "tight",
})

print("=" * 65)
print("GÖRSELLEŞTİRMELER OLUŞTURULUYOR")
print("=" * 65)

# ── Verileri Yükle ─────────────────────────────────────────────────────────
panel      = pd.read_parquet(ARA_DIR / "panel_with_predictions.parquet")
adi_cv2    = pd.read_parquet(ARA_DIR / "adi_cv2.parquet")
metrikler  = pd.read_parquet(ARA_DIR / "metrik_karsilastirma.parquet")

try:
    anomali_yuzd  = pd.read_parquet(ARA_DIR / "anomali_is_yuzdelik.parquet")
except: anomali_yuzd = pd.DataFrame()
try:
    churn         = pd.read_parquet(ARA_DIR / "anomali_churn.parquet")
except: churn = pd.DataFrame()
try:
    sistematik    = pd.read_parquet(ARA_DIR / "anomali_sistematik_cari.parquet")
except: sistematik = pd.DataFrame()
try:
    robotik       = pd.read_parquet(ARA_DIR / "anomali_robotik.parquet")
except: robotik = pd.DataFrame()
try:
    vk_supheli    = pd.read_parquet(ARA_DIR / "veri_kalitesi_supheli.parquet")
except: vk_supheli = pd.DataFrame()

y_gercek  = panel["siparis_var"].values
y_yeni    = panel["yeni_siparis_var"].values
y_eski    = panel["final_siparis"].fillna(0).astype(int).values

# ── Grafik 1: ADI / CV² Haritası ───────────────────────────────────────────
print("\n  Grafik 1: ADI/CV² haritası...")
tip_renkler = {
    "Düzgün"  : YLGNBU(0.85),
    "Düzensiz": YLGNBU(0.65),
    "Aralıklı": YLGNBU(0.45),
    "Yumrulu" : YLGNBU(0.25),
}
adi_p95 = adi_cv2["ADI"].quantile(0.95)
cv2_p95 = adi_cv2["CV2"].quantile(0.95)
df_g = adi_cv2[(adi_cv2["ADI"] <= adi_p95*1.2) & (adi_cv2["CV2"] <= cv2_p95*1.2)]

fig1, ax1 = plt.subplots(figsize=(10, 7))
for tip, grup in df_g.groupby("TalepTipi"):
    ax1.scatter(grup["ADI"], grup["CV2"], s=12, alpha=0.4,
                c=[tip_renkler.get(tip, "gray")], label=tip, rasterized=True)
ax1.axvline(ADI_ESIK, color="dimgray", linestyle="--", lw=1.3)
ax1.axhline(CV2_ESIK, color="dimgray", linestyle="--", lw=1.3)
for (adi_pos, cv2_pos, metin) in [
    (ADI_ESIK*0.45, CV2_ESIK*0.2, "Düzgün"),
    (ADI_ESIK*0.45, CV2_ESIK*1.7, "Düzensiz"),
    (ADI_ESIK*1.5,  CV2_ESIK*0.2, "Aralıklı"),
    (ADI_ESIK*1.5,  CV2_ESIK*1.7, "Yumrulu"),
]:
    ax1.text(adi_pos, cv2_pos, metin, fontsize=12, alpha=0.55,
             ha="center", fontweight="bold", color=tip_renkler.get(metin,"gray"))
ax1.set_xlabel(f"ADI (Ort. Talep Aralığı)  [Eşik: {ADI_ESIK}]", fontsize=12)
ax1.set_ylabel(f"CV²  [Eşik: {CV2_ESIK}]", fontsize=12)
ax1.set_title("ADI / CV² — Syntetos-Boylan Talep Tipi Haritası", fontsize=14, fontweight="bold", pad=12)
ax1.legend(title="Talep Tipi", fontsize=10)
ax1.grid(linestyle="--", alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)
fig1.tight_layout()
yol1 = GRAFIK_DIR / "10_01_adi_cv2_harita.png"
fig1.savefig(yol1); plt.close(fig1)
print(f"    Kaydedildi: {yol1.name}")

# ── Grafik 2: Confusion Matrix (yan yana) ──────────────────────────────────
print("\n  Grafik 2: Confusion matrix...")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, baslik in [
    (axes2[0], y_eski, "Final Tahmin (Eski)"),
    (axes2[1], y_yeni, "Yeni ML Tahmini"),
]:
    cm = confusion_matrix(y_gercek, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Sipariş Yok","Sipariş Var"])
    disp.plot(ax=ax, colorbar=False,
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  "ylgnbu_light", [YLGNBU(0.1), YLGNBU(0.9)]))
    ax.set_title(baslik, fontsize=13, fontweight="bold", pad=10)
fig2.suptitle("Confusion Matrix Karşılaştırması", fontsize=14, fontweight="bold")
fig2.tight_layout()
yol2 = GRAFIK_DIR / "10_02_confusion_matrix.png"
fig2.savefig(yol2); plt.close(fig2)
print(f"    Kaydedildi: {yol2.name}")

# ── Grafik 3: Tahmin-vs-Gerçek Scatter (log-log) ────────────────────────────
print("\n  Grafik 3: Tahmin-vs-gerçek scatter...")
gercek_pos  = panel[panel["siparis_var"] == 1]["ToplamMiktar"].values
yeni_pos    = panel[panel["siparis_var"] == 1]["yeni_tahmin_miktar"].values
eski_pos_val= panel[panel["siparis_var"] == 1].get(
    "final_tahmin", pd.Series(0, index=panel[panel["siparis_var"]==1].index)
).values

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
for ax, tahmin, baslik in [
    (axes3[0], eski_pos_val, "Final Tahmin (Eski)"),
    (axes3[1], yeni_pos,     "Yeni ML Tahmini"),
]:
    gercek_log = np.log1p(gercek_pos)
    tahmin_log = np.log1p(tahmin)
    ax.scatter(gercek_log, tahmin_log, s=5, alpha=0.15,
               c=[YLGNBU(0.7)], rasterized=True)
    lim = max(gercek_log.max(), tahmin_log.max()) + 0.1
    ax.plot([0, lim], [0, lim], "r--", lw=1.3, alpha=0.7, label="Mükemmel Tahmin")
    ax.set_xlabel("Gerçek Miktar (log1p)", fontsize=11)
    ax.set_ylabel("Tahmin Miktarı (log1p)", fontsize=11)
    ax.set_title(baslik, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
fig3.suptitle("Tahmin vs Gerçek Miktar (log-log ölçek)", fontsize=14, fontweight="bold")
fig3.tight_layout()
yol3 = GRAFIK_DIR / "10_03_tahmin_vs_gercek.png"
fig3.savefig(yol3); plt.close(fig3)
print(f"    Kaydedildi: {yol3.name}")

# ── Grafik 4: Zaman Serisi + Anomali Noktaları ─────────────────────────────
print("\n  Grafik 4: Zaman serisi + anomali...")
haftalik = panel.groupby("Hafta")["ToplamMiktar"].sum().reset_index()

fig4, ax4 = plt.subplots(figsize=(15, 5))
ax4.fill_between(haftalik["Hafta"], haftalik["ToplamMiktar"],
                 alpha=0.2, color=YLGNBU(0.7))
ax4.plot(haftalik["Hafta"], haftalik["ToplamMiktar"],
         color=YLGNBU(0.85), lw=1.8, label="Haftalık Toplam Miktar")

# Anomali noktaları
if not anomali_yuzd.empty and "Hafta" in anomali_yuzd.columns:
    anomali_haftalik = anomali_yuzd.groupby("Hafta")["ToplamMiktar"].sum().reset_index()
    ax4.scatter(anomali_haftalik["Hafta"], anomali_haftalik["ToplamMiktar"],
                color="tomato", s=40, zorder=5, label="Anomali (Yüzdelik Dışı)", alpha=0.8)

ax4.set_xlabel("Hafta", fontsize=12)
ax4.set_ylabel("Toplam Satış Miktarı", fontsize=12)
ax4.set_title("Haftalık Satış Zaman Serisi — Anomali Noktaları İşaretli",
              fontsize=13, fontweight="bold", pad=12)
ax4.legend(fontsize=10)
ax4.grid(axis="y", linestyle="--", alpha=0.4)
ax4.spines[["top","right"]].set_visible(False)
fig4.tight_layout()
yol4 = GRAFIK_DIR / "10_04_zaman_serisi_anomali.png"
fig4.savefig(yol4); plt.close(fig4)
print(f"    Kaydedildi: {yol4.name}")

# ── Grafik 5: Anomali Kategori Bar Chart ────────────────────────────────────
print("\n  Grafik 5: Anomali kategori bar chart...")
anomali_kategoriler = {
    "Yüzdelik Dışı\nSatış"   : len(anomali_yuzd),
    "Churn\nSinyali"          : len(churn),
    "Sistematik\nYön Hatası"  : len(sistematik),
    "Veri Kalitesi\nŞüpheli"  : len(vk_supheli),
    "Robotik\nKayıt"          : len(robotik),
}
fig5, ax5 = plt.subplots(figsize=(11, 5))
renkler5  = [YLGNBU(v) for v in np.linspace(0.35, 0.85, len(anomali_kategoriler))]
bars5     = ax5.bar(list(anomali_kategoriler.keys()), list(anomali_kategoriler.values()),
                    color=renkler5, edgecolor="white", width=0.55)
for bar in bars5:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{int(bar.get_height()):,}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
ax5.set_ylabel("Satır / Kayıt Sayısı", fontsize=12)
ax5.set_title("Anomali Kategorileri — Tespit Edilen Kayıt Sayıları",
              fontsize=13, fontweight="bold", pad=12)
ax5.grid(axis="y", linestyle="--", alpha=0.4)
ax5.spines[["top","right"]].set_visible(False)
fig5.tight_layout()
yol5 = GRAFIK_DIR / "10_05_anomali_kategori.png"
fig5.savefig(yol5); plt.close(fig5)
print(f"    Kaydedildi: {yol5.name}")

# ── Grafik 6: 7 Metrik Eski-Yeni Karşılaştırma ─────────────────────────────
print("\n  Grafik 6: 7 metrik karşılaştırma...")
metrikleri_listesi = ["Precision","Recall","F1","ROC_AUC","MAAPE","RMSE","Bias"]
x = np.arange(len(metrikleri_listesi))
genislik = 0.35

fig6, axes6 = plt.subplots(1, 2, figsize=(18, 6))
for ax, kapsam_etiket in [
    (axes6[0], "Tüm"),
    (axes6[1], "Aksiyon Alınabilir"),
]:
    eski_vals = [metrikler.loc[metrikler["Metrik"] == m,
                               f"Final Tahmin — {kapsam_etiket}"].values[0]
                 if m in metrikler["Metrik"].values else 0
                 for m in metrikleri_listesi]
    yeni_vals = [metrikler.loc[metrikler["Metrik"] == m,
                               f"Yeni Tahmin — {kapsam_etiket}"].values[0]
                 if m in metrikler["Metrik"].values else 0
                 for m in metrikleri_listesi]

    ax.bar(x - genislik/2, eski_vals, genislik,
           label="Final Tahmin (Eski)", color=YLGNBU(0.4), edgecolor="white")
    ax.bar(x + genislik/2, yeni_vals, genislik,
           label="Yeni ML Tahmini",    color=YLGNBU(0.85), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metrikleri_listesi, rotation=25, ha="right", fontsize=9)
    ax.set_title(f"Metrik Karşılaştırması — {kapsam_etiket} Kapsam",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Değer", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

fig6.suptitle("7 Metrik: Final Tahmin vs Yeni ML Tahmini",
              fontsize=14, fontweight="bold")
fig6.tight_layout()
yol6 = GRAFIK_DIR / "10_06_metrik_karsilastirma.png"
fig6.savefig(yol6); plt.close(fig6)
print(f"    Kaydedildi: {yol6.name}")

print("\n" + "=" * 65)
print("GÖRSELLEŞTİRMELER TAMAMLANDI ✓")
print(f"Tüm grafikler: {GRAFIK_DIR}")
print("=" * 65)
