"""
1_ML_Model/04_classification.py — Sınıflandırma Model Seçimi (Hız Optimize)
==============================================================================
MacBook Pro M3, CPU-only optimizasyonları:
  - Ön-eleme turu: tüm adaylar varsayılan paramlarla hızlıca test edilir
  - Optuna: sadece en iyi 2 model, 40k alt küme, 20 deneme, timeout=600s
  - 3-katlı CV, MedianPruner, n_jobs=4 model / n_jobs=2 Optuna

Çalıştır: python 1_ML_Model/04_classification.py
"""

import pathlib, sys, warnings, pickle
warnings.filterwarnings("ignore")

KOK = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(KOK))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ── Ayarlar ───────────────────────────────────────────────────────────────
N_FOLD          = 3       # 5 yerine 3
N_OPTUNA        = 20      # 100 yerine 20
OPTUNA_TIMEOUT  = 600     # saniye (model başına maks)
N_JOBS_MODEL    = 4       # model n_jobs
N_JOBS_OPTUNA   = 2       # study.optimize n_jobs
ONEK_ELEME_K    = 2       # Optuna'ya gidecek en iyi model sayısı
ALT_KUME_BOYUTU = 40_000  # Optuna arama alt küme

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"
GRAFIK_DIR  = CALISMA_DIR / "grafikler"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
YLGNBU      = plt.cm.YlGnBu

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                      "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})

print("=" * 65)
print("SINIFLANDIRMA MODEL SEÇİMİ — HIZ OPTİMİZE")
print("=" * 65)

# ── Veri Hazırla ───────────────────────────────────────────────────────────
panel = pd.read_parquet(ARA_DIR / "feature_matrix.parquet")
print(f"feature_matrix: {panel.shape}")

OZELLIKLER = [
    "lag_1","lag_2","lag_4","lag_8",
    "lag_siparis_1","lag_siparis_2","lag_siparis_4","lag_siparis_8",
    "roll_ort_4","roll_ort_8","roll_ort_12",
    "roll_std_4","roll_std_8","roll_std_12",
    "roll_siklık_4","roll_siklık_8","roll_siklık_12",
    "recency_hafta","ADI","CV2","TalepTipiKod",
    "cari_freq","stok_freq","cari_target_enc","stok_target_enc",
    "ay","ceyrek","hafta_no","yil_basi","yil_sonu","ay_basi",
]
OZELLIKLER = [f for f in OZELLIKLER if f in panel.columns]
HEDEF      = "siparis_var"

veri = panel[OZELLIKLER + [HEDEF, "Hafta"]].dropna(subset=OZELLIKLER).copy()
veri = veri.sort_values("Hafta").reset_index(drop=True)
print(f"Model verisi (NaN temizlendi): {veri.shape}")
print(f"Sipariş oranı: {veri[HEDEF].mean()*100:.1f}%")

X_tam = veri[OZELLIKLER].values
y_tam = veri[HEDEF].values

# ── Stratified 40k Alt Küme (Optuna için) ──────────────────────────────────
np.random.seed(42)
if len(veri) > ALT_KUME_BOYUTU:
    # Sınıf dengesini koru
    pos_idx = np.where(y_tam == 1)[0]
    neg_idx = np.where(y_tam == 0)[0]
    n_pos   = int(ALT_KUME_BOYUTU * y_tam.mean())
    n_neg   = ALT_KUME_BOYUTU - n_pos
    sec_pos = np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False)
    sec_neg = np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)
    alt_idx = np.sort(np.concatenate([sec_pos, sec_neg]))
    X_alt   = X_tam[alt_idx]
    y_alt   = y_tam[alt_idx]
    print(f"Optuna alt küme: {len(alt_idx):,} satır (sipariş: {y_alt.mean()*100:.1f}%)")
else:
    X_alt, y_alt = X_tam, y_tam
    print("Optuna: tam veri kullanılıyor (40k altında)")

tscv = TimeSeriesSplit(n_splits=N_FOLD)

def cv_roc_auc(model_sinifi, params, X, y):
    """3-katlı zaman bazlı CV ile ortalama ROC-AUC döndür."""
    auc_list = []
    for train_idx, val_idx in tscv.split(X):
        m = model_sinifi(**params)
        m.fit(X[train_idx], y[train_idx])
        y_prob = m.predict_proba(X[val_idx])[:, 1]
        try:
            auc_list.append(roc_auc_score(y[val_idx], y_prob))
        except Exception:
            auc_list.append(0.5)
    return float(np.mean(auc_list))

def cv_tum_metrik(model_sinifi, params, X, y):
    f1_list, pr_list, rc_list, auc_list = [], [], [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = model_sinifi(**params)
        m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        f1_list.append(f1_score(y_val, y_pred, zero_division=0))
        pr_list.append(precision_score(y_val, y_pred, zero_division=0))
        rc_list.append(recall_score(y_val, y_pred, zero_division=0))
        try:
            auc_list.append(roc_auc_score(y_val, y_prob))
        except Exception:
            auc_list.append(0.5)
    return {
        "F1"     : float(np.mean(f1_list)),
        "Precision": float(np.mean(pr_list)),
        "Recall" : float(np.mean(rc_list)),
        "ROC_AUC": float(np.mean(auc_list)),
    }

# ══════════════════════════════════════════════════════════════════════════
# AŞAMA 1 — ÖN ELEME TURU (varsayılan parametreler, tam veri)
# ══════════════════════════════════════════════════════════════════════════
print("\n── AŞAMA 1: Ön Eleme Turu (varsayılan parametreler) ──")

VARSAYILAN_MODELLER = [
    ("LightGBM",      LGBMClassifier,
     dict(n_estimators=200, n_jobs=N_JOBS_MODEL, random_state=42, verbose=-1)),
    ("XGBoost",       XGBClassifier,
     dict(n_estimators=200, n_jobs=N_JOBS_MODEL, random_state=42,
          eval_metric="logloss", verbosity=0)),
    ("CatBoost",      CatBoostClassifier,
     dict(iterations=200, thread_count=N_JOBS_MODEL, random_seed=42, verbose=0)),
    ("Random Forest", RandomForestClassifier,
     dict(n_estimators=200, n_jobs=N_JOBS_MODEL, random_state=42)),
    ("Logistic Reg.", LogisticRegression,
     dict(max_iter=500, n_jobs=N_JOBS_MODEL, random_state=42)),
]

onek_sonuclar = []
for adi, sinif, params in VARSAYILAN_MODELLER:
    print(f"  [{adi}] varsayılan CV...", end=" ", flush=True)
    auc = cv_roc_auc(sinif, params, X_tam, y_tam)
    print(f"ROC-AUC={auc:.4f}")
    onek_sonuclar.append({"Model": adi, "Sinif": sinif,
                          "VarsayilanParams": params, "OncekROC_AUC": auc})

onek_df = pd.DataFrame(onek_sonuclar).sort_values("OncekROC_AUC", ascending=False)
print("\n  Ön eleme sıralaması:")
print(onek_df[["Model","OncekROC_AUC"]].to_string(index=False))

# En iyi 2'yi Optuna'ya gönder
optuna_adaylar = onek_df.head(ONEK_ELEME_K).to_dict("records")
elenen_adaylar = onek_df.tail(len(onek_df) - ONEK_ELEME_K).to_dict("records")
print(f"\n  Optuna'ya gidecek: {[a['Model'] for a in optuna_adaylar]}")
print(f"  Elenenler       : {[a['Model'] for a in elenen_adaylar]}")

# ══════════════════════════════════════════════════════════════════════════
# AŞAMA 2 — OPTUNA (sadece en iyi 2 model, 40k alt küme)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n── AŞAMA 2: Optuna ({N_OPTUNA} deneme, timeout={OPTUNA_TIMEOUT}s) ──")

def lgbm_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500),
        learning_rate    = trial.suggest_float("lr", 0.02, 0.3, log=True),
        num_leaves       = trial.suggest_int("num_leaves", 16, 96),
        max_depth        = trial.suggest_int("max_depth", 3, 9),
        min_child_samples= trial.suggest_int("min_child_samples", 10, 80),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample", 0.6, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        n_jobs=N_JOBS_MODEL, random_state=42, verbose=-1,
    )
    score = cv_roc_auc(LGBMClassifier, params, X_alt, y_alt)
    return score

def xgb_objective(trial):
    params = dict(
        n_estimators    = trial.suggest_int("n_estimators", 100, 500),
        learning_rate   = trial.suggest_float("lr", 0.02, 0.3, log=True),
        max_depth       = trial.suggest_int("max_depth", 3, 8),
        min_child_weight= trial.suggest_int("min_child_weight", 1, 15),
        subsample       = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree= trial.suggest_float("colsample", 0.6, 1.0),
        gamma           = trial.suggest_float("gamma", 0.0, 3.0),
        n_jobs=N_JOBS_MODEL, random_state=42,
        eval_metric="logloss", verbosity=0,
    )
    return cv_roc_auc(XGBClassifier, params, X_alt, y_alt)

def cat_objective(trial):
    params = dict(
        iterations    = trial.suggest_int("iterations", 100, 500),
        learning_rate = trial.suggest_float("lr", 0.02, 0.3, log=True),
        depth         = trial.suggest_int("depth", 3, 8),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 0.5, 15.0),
        thread_count=N_JOBS_MODEL, random_seed=42, verbose=0,
    )
    return cv_roc_auc(CatBoostClassifier, params, X_alt, y_alt)

def rf_objective(trial):
    params = dict(
        n_estimators   = trial.suggest_int("n_estimators", 100, 400),
        max_depth      = trial.suggest_int("max_depth", 3, 15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
        min_samples_leaf =trial.suggest_int("min_samples_leaf", 1, 15),
        max_features   = trial.suggest_categorical("max_features", ["sqrt","log2"]),
        n_jobs=N_JOBS_MODEL, random_state=42,
    )
    return cv_roc_auc(RandomForestClassifier, params, X_alt, y_alt)

MODEL_OBJECTIVE_MAP = {
    "LightGBM"     : (LGBMClassifier,       lgbm_objective),
    "XGBoost"      : (XGBClassifier,        xgb_objective),
    "CatBoost"     : (CatBoostClassifier,   cat_objective),
    "Random Forest": (RandomForestClassifier, rf_objective),
    "Logistic Reg.": (None, None),  # Optuna gerekmez
}

pruner   = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
sonuclar = []
en_iyi_f1         = -1
en_iyi_model_adi  = None
en_iyi_model_sinifi = None
en_iyi_params     = None

# Optuna uygulanan modeller
for aday in optuna_adaylar:
    adi   = aday["Model"]
    sinif, obj_fn = MODEL_OBJECTIVE_MAP.get(adi, (None, None))

    if obj_fn is None:
        # Logistic Regression için Optuna yok, varsayılan paramlar kullan
        metrikleri = cv_tum_metrik(
            aday["Sinif"], aday["VarsayilanParams"], X_tam, y_tam
        )
        en_iyi_prm = aday["VarsayilanParams"]
        sinif      = aday["Sinif"]
        print(f"  [{adi}] Optuna atlandı — F1={metrikleri['F1']:.4f}")
    else:
        print(f"  [{adi}] Optuna başlıyor ({N_OPTUNA} deneme, {OPTUNA_TIMEOUT}s)...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner,
        )
        study.optimize(obj_fn, n_trials=N_OPTUNA,
                       timeout=OPTUNA_TIMEOUT, n_jobs=N_JOBS_OPTUNA,
                       show_progress_bar=False)
        en_iyi_prm = study.best_params
        # n_jobs/thread_count ekle
        if "LightGBM" in adi:
            en_iyi_prm.update({"n_jobs": N_JOBS_MODEL, "random_state": 42, "verbose": -1})
        elif "XGBoost" in adi:
            en_iyi_prm.update({"n_jobs": N_JOBS_MODEL, "random_state": 42,
                                "eval_metric": "logloss", "verbosity": 0})
        elif "CatBoost" in adi:
            en_iyi_prm.update({"thread_count": N_JOBS_MODEL, "random_seed": 42, "verbose": 0})
        elif "Random Forest" in adi:
            en_iyi_prm.update({"n_jobs": N_JOBS_MODEL, "random_state": 42})
        metrikleri = cv_tum_metrik(sinif, en_iyi_prm, X_tam, y_tam)
        print(f"    Optuna en iyi ROC-AUC={study.best_value:.4f} → tam veri CV F1={metrikleri['F1']:.4f}")

    sonuclar.append({
        "Model": adi, **metrikleri,
        "Durum": "Optuna", "EnIyiParams": en_iyi_prm
    })
    if metrikleri["F1"] > en_iyi_f1:
        en_iyi_f1, en_iyi_model_adi = metrikleri["F1"], adi
        en_iyi_model_sinifi, en_iyi_params = sinif, en_iyi_prm

# Elenen modeller — varsayılan metrikler
for aday in elenen_adaylar:
    adi = aday["Model"]
    metrikleri = cv_tum_metrik(
        aday["Sinif"], aday["VarsayilanParams"], X_tam, y_tam
    )
    sonuclar.append({
        "Model": adi, **metrikleri,
        "Durum": "Ön Elmede Elendi", "EnIyiParams": aday["VarsayilanParams"]
    })
    print(f"  [{adi}] Ön elmede elendi — F1={metrikleri['F1']:.4f}")

# ── Sonuç Tablosu ──────────────────────────────────────────────────────────
df_sonuc = pd.DataFrame(sonuclar).sort_values("F1", ascending=False)
print("\n" + "=" * 65)
print("SINIFLANDIRMA SONUÇ TABLOSU")
print("=" * 65)
print(df_sonuc[["Model","Durum","F1","Precision","Recall","ROC_AUC"]].to_string(index=False))

# ── Optimal Karar Eşiği ────────────────────────────────────────────────────
print(f"\n  En iyi model: {en_iyi_model_adi} (CV F1={en_iyi_f1:.4f})")
son_fold  = list(tscv.split(X_tam))[-1]
X_tr, X_te= X_tam[son_fold[0]], X_tam[son_fold[1]]
y_tr, y_te= y_tam[son_fold[0]], y_tam[son_fold[1]]
son_model = en_iyi_model_sinifi(**en_iyi_params)
son_model.fit(X_tr, y_tr)
y_prob_te = son_model.predict_proba(X_te)[:, 1]
esikler   = np.arange(0.1, 0.9, 0.01)
f1_esikler= [f1_score(y_te, (y_prob_te >= e).astype(int), zero_division=0) for e in esikler]
optimal_esik = esikler[np.argmax(f1_esikler)]
print(f"  Optimal karar eşiği: {optimal_esik:.2f} (F1={max(f1_esikler):.4f})")

# ── Final Model — Tam Veriyle Eğit ────────────────────────────────────────
print(f"\n  Final model tam veriyle eğitiliyor...")
final_clf = en_iyi_model_sinifi(**en_iyi_params)
final_clf.fit(X_tam, y_tam)

with open(ARA_DIR / "best_classifier.pkl", "wb") as f:
    pickle.dump({
        "model"       : final_clf,
        "ozellikler"  : OZELLIKLER,
        "optimal_esik": optimal_esik,
        "model_adi"   : en_iyi_model_adi,
        "cv_sonuclari": df_sonuc.drop(columns=["EnIyiParams"]).to_dict("records"),
    }, f)
print(f"  Kaydedildi: {ARA_DIR / 'best_classifier.pkl'}")

df_sonuc.drop(columns=["EnIyiParams"]).to_parquet(
    ARA_DIR / "siniflandirma_sonuclari.parquet", index=False
)

# ── Bar Chart ──────────────────────────────────────────────────────────────
metrikler_listesi = ["F1","Precision","Recall","ROC_AUC"]
model_isimleri    = df_sonuc["Model"].tolist()
x, w              = np.arange(len(model_isimleri)), 0.18

fig, ax = plt.subplots(figsize=(14, 6))
for i, (metrik, renk) in enumerate(zip(metrikler_listesi,
                                        [YLGNBU(v) for v in [0.4,0.55,0.7,0.85]])):
    bars = ax.bar(x + i*w, df_sonuc[metrik], w, label=metrik,
                  color=renk, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7, rotation=90)

ax.set_xticks(x + w*1.5)
ax.set_xticklabels(model_isimleri, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Skor", fontsize=12)
ax.set_title(
    f"Sınıflandırma Modeli Karşılaştırması\n"
    f"(Ön eleme + Optuna {N_OPTUNA} deneme, {N_FOLD}-katlı CV, 40k alt küme)",
    fontsize=12, fontweight="bold", pad=12
)
ax.legend(title="Metrik", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
yol = GRAFIK_DIR / "04_siniflandirma_karsilastirma.png"
fig.savefig(yol); plt.close(fig)
print(f"\nGrafik kaydedildi: {yol}")
print("\nSINIFLANDIRMA TAMAMLANDI ✓")
