"""
1_ML_Model/07_evaluation.py — Tahmin Doğruluğu Değerlendirmesi (7 Metrik)
===========================================================================
final_tahmin (mevcut) vs yeni_tahmin (ML) yan yana karşılaştırma
Kapsam: "Tüm" ve "Sadece Aktif" (sipariş olan) satırlar
Metrikler: Precision, Recall, F1, ROC-AUC, MAAPE, RMSE, Bias

Çalıştır: python 1_ML_Model/07_evaluation.py
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
)

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"
GRAFIK_DIR  = CALISMA_DIR / "grafikler"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
YLGNBU      = plt.cm.YlGnBu

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                      "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})

print("=" * 65)
print("TAHMİN DOĞRULUĞU DEĞERLENDİRMESİ BAŞLADI")
print("=" * 65)

# ── Veri ──────────────────────────────────────────────────────────────────
panel = pd.read_parquet(ARA_DIR / "panel_with_predictions.parquet")

# Gerçek değerler
y_gercek        = panel["siparis_var"].values
y_gercek_miktar = panel["ToplamMiktar"].values  # ham miktar

# Gerçek final_tahmin'i tahmin_temiz.parquet'ten al
tahmin_temiz = pd.read_parquet(ARA_DIR / "tahmin_temiz.parquet")
final_tahmin_ref = (
    tahmin_temiz.groupby(["cari_kod", "stok_kod"])["final_tahmin"]
    .mean()
    .reset_index()
    .rename(columns={"final_tahmin": "final_tahmin_ref"})
)
panel = panel.merge(
    final_tahmin_ref,
    left_on=["CariKod", "StockKod"],
    right_on=["cari_kod", "stok_kod"],
    how="left"
)
panel["final_tahmin"] = panel["final_tahmin_ref"].fillna(0)
panel = panel.drop(columns=["cari_kod", "stok_kod", "final_tahmin_ref"],
                   errors="ignore")

panel["final_tahmin"] = pd.to_numeric(panel["final_tahmin"], errors="coerce").fillna(0)
panel["final_siparis"] = (panel["final_tahmin"] > 0).astype(int)

# ── Yardımcı Metrik Fonksiyonları ──────────────────────────────────────────
def maape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.arctan(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9)))))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.array(y_true), np.array(y_pred))))

def bias(y_true, y_pred):
    return float(np.mean(np.array(y_pred) - np.array(y_true)))

def metrik_hesapla(y_true_cls, y_pred_cls, y_prob, y_true_reg, y_pred_reg, etiket):
    return {
        "Kapsam"      : etiket,
        "Precision"   : precision_score(y_true_cls, y_pred_cls, zero_division=0),
        "Recall"      : recall_score(y_true_cls, y_pred_cls, zero_division=0),
        "F1"          : f1_score(y_true_cls, y_pred_cls, zero_division=0),
        "ROC_AUC"     : roc_auc_score(y_true_cls, y_prob) if len(np.unique(y_true_cls)) > 1 else 0.5,
        "MAAPE"       : maape(y_true_reg, y_pred_reg),
        "RMSE"        : rmse(y_true_reg, y_pred_reg),
        "Bias"        : bias(y_true_reg, y_pred_reg),
    }

# ── Final Tahmin (eski model) ───────────────────────────────────────────────
y_eski_cls   = panel["final_siparis"].values
y_eski_proba = panel["final_tahmin"].clip(0, panel["final_tahmin"].max()).values
if y_eski_proba.max() > 0:
    y_eski_proba = y_eski_proba / y_eski_proba.max()
y_eski_reg   = panel["final_tahmin"].values

# ── Yeni Tahmin (ML pipeline) ──────────────────────────────────────────────
y_yeni_cls   = panel["yeni_siparis_var"].values
y_yeni_proba = panel["siparis_proba"].values
y_yeni_reg   = panel["yeni_tahmin_miktar"].values

# ── Tüm Kapsam ─────────────────────────────────────────────────────────────
print("\n  Tüm kapsam metrikleri hesaplanıyor...")
eski_tum  = metrik_hesapla(y_gercek, y_eski_cls, y_eski_proba,
                           y_gercek_miktar, y_eski_reg, "Tüm Kapsam")
yeni_tum  = metrik_hesapla(y_gercek, y_yeni_cls, y_yeni_proba,
                           y_gercek_miktar, y_yeni_reg, "Tüm Kapsam")

# ── Sadece Aktif — model bazında ayrı maske (Accuracy Paradox Önleme) ─────
# TN satırları (her iki taraf da 0) metrikten hariç tutulur.
aktif_mask_eski = (y_gercek == 1) | (y_eski_cls == 1)
aktif_mask_yeni = (y_gercek == 1) | (y_yeni_cls == 1)

print(f"  Eski model aksiyon alınabilir satır: {aktif_mask_eski.sum():,} / {len(y_gercek):,}")
print(f"  Yeni model aksiyon alınabilir satır: {aktif_mask_yeni.sum():,} / {len(y_gercek):,}")

eski_aktif = metrik_hesapla(
    y_gercek[aktif_mask_eski], y_eski_cls[aktif_mask_eski],
    y_eski_proba[aktif_mask_eski],
    y_gercek_miktar[aktif_mask_eski], y_eski_reg[aktif_mask_eski],
    "Sadece Aktif"
)
yeni_aktif = metrik_hesapla(
    y_gercek[aktif_mask_yeni], y_yeni_cls[aktif_mask_yeni],
    y_yeni_proba[aktif_mask_yeni],
    y_gercek_miktar[aktif_mask_yeni], y_yeni_reg[aktif_mask_yeni],
    "Sadece Aktif"
)

# ── Karşılaştırma Tablosu ──────────────────────────────────────────────────
metrikleri_listesi = ["Precision","Recall","F1","ROC_AUC","MAAPE","RMSE","Bias"]

df_karsilastirma = pd.DataFrame({
    "Metrik"                             : metrikleri_listesi,
    "Final Tahmin — Tüm"                : [eski_tum[m]  for m in metrikleri_listesi],
    "Yeni Tahmin — Tüm"                 : [yeni_tum[m]  for m in metrikleri_listesi],
    "Final Tahmin — Aksiyon Alınabilir" : [eski_aktif[m] for m in metrikleri_listesi],
    "Yeni Tahmin — Aksiyon Alınabilir"  : [yeni_aktif[m] for m in metrikleri_listesi],
})

print("\n" + "=" * 65)
print("7 METRİK KARŞILAŞTIRMA TABLOSU")
print("=" * 65)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df_karsilastirma.to_string(index=False))

df_karsilastirma.to_parquet(ARA_DIR / "metrik_karsilastirma.parquet", index=False)

# ── Bar Chart ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax, kapsam, eski_soz, yeni_soz in [
    (axes[0], "Tüm Kapsam", eski_tum, yeni_tum),
    (axes[1], "Aksiyon Alınabilir (TN Hariç)", eski_aktif, yeni_aktif),
]:
    x        = np.arange(len(metrikleri_listesi))
    genislik = 0.35
    ax.bar(x - genislik/2, [eski_soz[m] for m in metrikleri_listesi],
           genislik, label="Final Tahmin (Eski)", color=YLGNBU(0.5), edgecolor="white")
    ax.bar(x + genislik/2, [yeni_soz[m] for m in metrikleri_listesi],
           genislik, label="Yeni Tahmin (ML)",    color=YLGNBU(0.85), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metrikleri_listesi, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"Metrik Karşılaştırması — {kapsam}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Değer", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Final Tahmin vs Yeni ML Tahmini — 7 Metrik Karşılaştırması",
             fontsize=14, fontweight="bold")
fig.tight_layout()
yol = GRAFIK_DIR / "07_metrik_karsilastirma.png"
fig.savefig(yol); plt.close(fig)
print(f"\nGrafik kaydedildi: {yol}")

# Güncellenmiş panel'i kaydet (final_siparis kolonu dahil)
panel.to_parquet(ARA_DIR / "panel_with_predictions.parquet", index=False)
print("Panel güncellendi: panel_with_predictions.parquet ✓")

print("\nDEĞERLENDİRME TAMAMLANDI ✓")
