"""
1_ML_Model/05_regression.py — Regresyon Model Seçimi (Hız Optimize)
======================================================================
MacBook Pro M3, CPU-only optimizasyonları:
  - SVR ÇIKARILDI (büyük veri yavaş)
  - Adaylar: LightGBM(Tweedie), XGBoost, CatBoost, ElasticNet
  - Ön-eleme turu → en iyi 2'ye Optuna
  - 40k alt küme, 20 deneme, timeout=600s, 3-katlı CV, MedianPruner

Çalıştır: python 1_ML_Model/05_regression.py
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ── Ayarlar ───────────────────────────────────────────────────────────────
N_FOLD          = 3
N_OPTUNA        = 20
OPTUNA_TIMEOUT  = 600
N_JOBS_MODEL    = 4
N_JOBS_OPTUNA   = 2
ONEK_ELEME_K    = 2
ALT_KUME_BOYUTU = 40_000

CALISMA_DIR = pathlib.Path(__file__).parent.resolve()
ARA_DIR     = CALISMA_DIR / "ara_ciktilar"
GRAFIK_DIR  = CALISMA_DIR / "grafikler"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
YLGNBU      = plt.cm.YlGnBu

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                      "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})

print("=" * 65)
print("REGRESYON MODEL SEÇİMİ — HIZ OPTİMİZE")
print("=" * 65)

# ── Veri Hazırla ───────────────────────────────────────────────────────────
panel    = pd.read_parquet(ARA_DIR / "feature_matrix.parquet")
reg_veri = panel[panel["siparis_var"] == 1].copy()
print(f"Regresyon veri (siparis_var=1): {reg_veri.shape}")

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
OZELLIKLER = [f for f in OZELLIKLER if f in reg_veri.columns]
HEDEF      = "log_miktar"

veri = reg_veri[OZELLIKLER + [HEDEF, "Hafta"]].dropna(
    subset=OZELLIKLER + [HEDEF]
).sort_values("Hafta").reset_index(drop=True)
print(f"NaN temizlendi: {veri.shape}")

X_tam = veri[OZELLIKLER].values
y_tam = veri[HEDEF].values

# ElasticNet için scaler (en başta fit et, alt küme hariç)
scaler = StandardScaler()
X_tam_sc = scaler.fit_transform(X_tam)

# ── 40k Alt Küme (zaman sırası korunarak son 40k satır) ────────────────────
if len(veri) > ALT_KUME_BOYUTU:
    alt_idx  = np.arange(len(veri) - ALT_KUME_BOYUTU, len(veri))
    X_alt    = X_tam[alt_idx]
    y_alt    = y_tam[alt_idx]
    X_alt_sc = X_tam_sc[alt_idx]
    print(f"Optuna alt küme: {len(alt_idx):,} satır (son {ALT_KUME_BOYUTU:,})")
else:
    X_alt, y_alt, X_alt_sc = X_tam, y_tam, X_tam_sc

tscv = TimeSeriesSplit(n_splits=N_FOLD)

# ── Metrik Fonksiyonları ───────────────────────────────────────────────────
def maape(y_true, y_pred):
    return float(np.mean(np.arctan(np.abs((y_true-y_pred)/(np.abs(y_true)+1e-9)))))
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))

def cv_maape(model_sinifi, params, X, y):
    vals = []
    for tr, val in tscv.split(X):
        m = model_sinifi(**params)
        m.fit(X[tr], y[tr])
        vals.append(maape(y[val], m.predict(X[val])))
    return float(np.mean(vals))

def cv_tum_metrik(model_sinifi, params, X, y):
    m_list, r_list, b_list = [], [], []
    for tr, val in tscv.split(X):
        m = model_sinifi(**params)
        m.fit(X[tr], y[tr])
        yp = m.predict(X[val])
        m_list.append(maape(y[val], yp))
        r_list.append(rmse(y[val], yp))
        b_list.append(bias(y[val], yp))
    return {"MAAPE": np.mean(m_list), "RMSE": np.mean(r_list), "Bias": np.mean(b_list)}

# ══════════════════════════════════════════════════════════════════════════
# AŞAMA 1 — ÖN ELEME (varsayılan parametreler, tam veri)
# ══════════════════════════════════════════════════════════════════════════
print("\n── AŞAMA 1: Ön Eleme Turu ──")

VARSAYILAN = [
    ("LightGBM (Tweedie)", LGBMRegressor,
     dict(n_estimators=200, objective="tweedie", tweedie_variance_power=1.5,
          n_jobs=N_JOBS_MODEL, random_state=42, verbose=-1), X_tam),
    ("XGBoost",            XGBRegressor,
     dict(n_estimators=200, n_jobs=N_JOBS_MODEL, random_state=42, verbosity=0), X_tam),
    ("CatBoost",           CatBoostRegressor,
     dict(iterations=200, thread_count=N_JOBS_MODEL, random_seed=42, verbose=0), X_tam),
    ("ElasticNet",         ElasticNet,
     dict(alpha=0.01, l1_ratio=0.5, max_iter=2000), X_tam_sc),  # scaled
]

onek_sonuclar = []
for adi, sinif, params, X_kullan in VARSAYILAN:
    print(f"  [{adi}]...", end=" ", flush=True)
    m_val = cv_maape(sinif, params, X_kullan, y_tam)
    print(f"MAAPE={m_val:.4f}")
    onek_sonuclar.append({
        "Model": adi, "Sinif": sinif, "VarsayilanParams": params,
        "OncekMAAPE": m_val, "KullanX": "scaled" if adi == "ElasticNet" else "raw"
    })

onek_df = pd.DataFrame(onek_sonuclar).sort_values("OncekMAAPE")
print("\n  Ön eleme sıralaması (küçük MAAPE iyi):")
print(onek_df[["Model","OncekMAAPE"]].to_string(index=False))

optuna_adaylar = onek_df.head(ONEK_ELEME_K).to_dict("records")
elenen_adaylar = onek_df.tail(len(onek_df) - ONEK_ELEME_K).to_dict("records")
print(f"\n  Optuna'ya gidecek: {[a['Model'] for a in optuna_adaylar]}")
print(f"  Elenenler       : {[a['Model'] for a in elenen_adaylar]}")

# ══════════════════════════════════════════════════════════════════════════
# AŞAMA 2 — OPTUNA
# ══════════════════════════════════════════════════════════════════════════
print(f"\n── AŞAMA 2: Optuna ({N_OPTUNA} deneme, timeout={OPTUNA_TIMEOUT}s) ──")

def lgbm_obj(trial):
    params = dict(
        n_estimators =trial.suggest_int("n_estimators", 100, 500),
        learning_rate=trial.suggest_float("lr", 0.02, 0.25, log=True),
        num_leaves   =trial.suggest_int("num_leaves", 16, 80),
        max_depth    =trial.suggest_int("max_depth", 3, 9),
        min_child_samples=trial.suggest_int("min_cs", 10, 80),
        subsample    =trial.suggest_float("sub", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("col", 0.6, 1.0),
        objective    ="tweedie",
        tweedie_variance_power=trial.suggest_float("tvp", 1.0, 1.9),
        n_jobs=N_JOBS_MODEL, random_state=42, verbose=-1,
    )
    return -cv_maape(LGBMRegressor, params, X_alt, y_alt)

def xgb_obj(trial):
    params = dict(
        n_estimators    =trial.suggest_int("n_estimators", 100, 500),
        learning_rate   =trial.suggest_float("lr", 0.02, 0.25, log=True),
        max_depth       =trial.suggest_int("max_depth", 3, 8),
        min_child_weight=trial.suggest_int("mcw", 1, 15),
        subsample       =trial.suggest_float("sub", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("col", 0.6, 1.0),
        n_jobs=N_JOBS_MODEL, random_state=42, verbosity=0,
    )
    return -cv_maape(XGBRegressor, params, X_alt, y_alt)

def cat_obj(trial):
    params = dict(
        iterations   =trial.suggest_int("iterations", 100, 500),
        learning_rate=trial.suggest_float("lr", 0.02, 0.25, log=True),
        depth        =trial.suggest_int("depth", 3, 8),
        l2_leaf_reg  =trial.suggest_float("l2", 0.5, 15.0),
        thread_count=N_JOBS_MODEL, random_seed=42, verbose=0,
    )
    return -cv_maape(CatBoostRegressor, params, X_alt, y_alt)

def en_obj(trial):
    params = dict(
        alpha    =trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        l1_ratio =trial.suggest_float("l1_ratio", 0.0, 1.0),
        max_iter =2000,
    )
    return -cv_maape(ElasticNet, params, X_alt_sc, y_alt)

OBJ_MAP = {
    "LightGBM (Tweedie)": (LGBMRegressor, lgbm_obj, "raw"),
    "XGBoost"            : (XGBRegressor,  xgb_obj,  "raw"),
    "CatBoost"           : (CatBoostRegressor, cat_obj, "raw"),
    "ElasticNet"         : (ElasticNet,    en_obj,   "scaled"),
}

pruner   = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
sonuclar = []
en_iyi_maape        = float("inf")
en_iyi_model_adi    = None
en_iyi_model_sinifi = None
en_iyi_params       = None
en_iyi_x_tipi       = "raw"

for aday in optuna_adaylar:
    adi = aday["Model"]
    sinif, obj_fn, x_tipi = OBJ_MAP[adi]
    X_opt = X_alt_sc if x_tipi == "scaled" else X_alt
    X_full = X_tam_sc if x_tipi == "scaled" else X_tam

    print(f"  [{adi}] Optuna ({N_OPTUNA} deneme)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )
    study.optimize(obj_fn, n_trials=N_OPTUNA,
                   timeout=OPTUNA_TIMEOUT, n_jobs=N_JOBS_OPTUNA,
                   show_progress_bar=False)

    en_iyi_prm = study.best_params
    # n_jobs ekle
    if "LightGBM" in adi:
        en_iyi_prm.update({"objective":"tweedie","n_jobs":N_JOBS_MODEL,
                           "random_state":42,"verbose":-1})
        if "tvp" in en_iyi_prm:
            en_iyi_prm["tweedie_variance_power"] = en_iyi_prm.pop("tvp")
    elif "XGBoost" in adi:
        en_iyi_prm.update({"n_jobs":N_JOBS_MODEL,"random_state":42,"verbosity":0})
    elif "CatBoost" in adi:
        en_iyi_prm.update({"thread_count":N_JOBS_MODEL,"random_seed":42,"verbose":0})

    met = cv_tum_metrik(sinif, en_iyi_prm, X_full, y_tam)
    print(f"    MAAPE={met['MAAPE']:.4f} RMSE={met['RMSE']:.4f} Bias={met['Bias']:.4f}")

    sonuclar.append({"Model": adi, **met, "Durum": "Optuna", "EnIyiParams": en_iyi_prm})
    if met["MAAPE"] < en_iyi_maape:
        en_iyi_maape, en_iyi_model_adi = met["MAAPE"], adi
        en_iyi_model_sinifi, en_iyi_params, en_iyi_x_tipi = sinif, en_iyi_prm, x_tipi

# Elenenler
for aday in elenen_adaylar:
    adi = aday["Model"]
    _, _, x_tipi = OBJ_MAP.get(adi, (None, None, "raw"))
    X_full = X_tam_sc if x_tipi == "scaled" else X_tam
    met = cv_tum_metrik(aday["Sinif"], aday["VarsayilanParams"], X_full, y_tam)
    sonuclar.append({"Model": adi, **met, "Durum":"Ön Elmede Elendi",
                     "EnIyiParams": aday["VarsayilanParams"]})
    print(f"  [{adi}] Ön elmede elendi — MAAPE={met['MAAPE']:.4f}")

# ── Sonuç Tablosu ──────────────────────────────────────────────────────────
df_sonuc = pd.DataFrame(sonuclar).sort_values("MAAPE")
print("\n" + "=" * 65)
print("REGRESYON SONUÇ TABLOSU")
print("=" * 65)
print(df_sonuc[["Model","Durum","MAAPE","RMSE","Bias"]].to_string(index=False))

# ── Final Model — Tam Veriyle ──────────────────────────────────────────────
X_fit = X_tam_sc if en_iyi_x_tipi == "scaled" else X_tam
final_reg = en_iyi_model_sinifi(**en_iyi_params)
final_reg.fit(X_fit, y_tam)
print(f"\n  En iyi: {en_iyi_model_adi} — tam veriyle eğitildi")

with open(ARA_DIR / "best_regressor.pkl", "wb") as f:
    pickle.dump({
        "model"      : final_reg,
        "ozellikler" : OZELLIKLER,
        "model_adi"  : en_iyi_model_adi,
        "scaler"     : scaler if en_iyi_x_tipi == "scaled" else None,
        "cv_sonuclari": df_sonuc.drop(columns=["EnIyiParams"]).to_dict("records"),
    }, f)
print(f"  Kaydedildi: {ARA_DIR / 'best_regressor.pkl'}")

df_sonuc.drop(columns=["EnIyiParams"]).to_parquet(
    ARA_DIR / "regresyon_sonuclari.parquet", index=False
)

# ── Bar Chart ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metrik, renk in zip(axes, ["MAAPE","RMSE","Bias"],
                             [YLGNBU(0.6), YLGNBU(0.75), YLGNBU(0.45)]):
    bars = ax.bar(df_sonuc["Model"], df_sonuc[metrik], color=renk, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(metrik, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle(
    f"Regresyon Modeli Karşılaştırması\n"
    f"(Ön eleme + Optuna {N_OPTUNA} deneme, {N_FOLD}-katlı CV, 40k alt küme)",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
yol = GRAFIK_DIR / "05_regresyon_karsilastirma.png"
fig.savefig(yol); plt.close(fig)
print(f"Grafik: {yol}")
print("\nREGRESYON TAMAMLANDI ✓")
