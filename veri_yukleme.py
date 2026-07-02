"""
veri_yukleme.py — Ortak Veri Yükleme Yardımcısı
=================================================
Her iki çalışma klasörü (1_ML_Model ve 2_Kural_Tabanli_Denetim) bu
modülü import ederek veri yükler. İş mantığı ve çıktılar ayrı kalır.

Kullanım:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from veri_yukleme import yukle_tahmin, yukle_satis
"""

import pathlib
import sys

import numpy as np
import pandas as pd

# ── Kök dizin (bu dosyanın bulunduğu yer) ──────────────────────────────────
KOK_DIR = pathlib.Path(__file__).parent.resolve()

TAHMIN_DOSYA = KOK_DIR / "faz1_faz2_anonim.xlsx"
SATIS_DOSYA  = KOK_DIR / "TblGecmisSatisVerileri_haftalik(Sheet1).csv"

# ── Sabitler ───────────────────────────────────────────────────────────────
SATIS_ENCODING  = "cp1254"
SATIS_SEP       = ";"
SATIS_DECIMAL   = ","
CHUNK_SIZE      = 100_000   # bellek verimliliği için


# ─────────────────────────────────────────────────────────────────────────────
def _iso_hafta_to_pazartesi(hafta_str: str) -> pd.Timestamp:
    """
    ISO hafta kodunu (örn. '2026-24') Pazartesi başlangıçlı gerçek tarihe çevirir.
    Farklı formatları (YYYY-WXX, YYYY-XX, YYYY/XX) destekler.
    """
    s = str(hafta_str).strip()
    # 'W' harfi varsa temizle: '2026-W24' → '2026-24'
    s = s.replace("W", "").replace("/", "-")
    bolumler = s.split("-")
    if len(bolumler) == 2:
        yil, hafta = int(bolumler[0]), int(bolumler[1])
    else:
        raise ValueError(f"Bilinmeyen oneri_hafta formatı: {hafta_str!r}")
    # ISO hafta → Pazartesi tarihi
    return pd.Timestamp.fromisocalendar(yil, hafta, 1)


# ─────────────────────────────────────────────────────────────────────────────
def yukle_tahmin(dosya: pathlib.Path = TAHMIN_DOSYA) -> pd.DataFrame:
    """
    faz1_faz2_anonim.xlsx dosyasını yükler.

    İşlemler:
    - İlk sayfa okunur.
    - cari_kod, stok_kod → str (strip ile).
    - oneri_hafta → forecast_date (Pazartesi, datetime64).
    - f1_tahmin eksiklerini 0 ile doldurur (Adım 1.2 öncesi basit düzeltme).

    Döndürür: pd.DataFrame
    """
    print(f"[yukle_tahmin] Yükleniyor: {dosya}")
    df = pd.read_excel(dosya, sheet_name=0, engine="openpyxl", dtype=str)

    # Sütun isimlerini küçük harfe çevir ve boşlukları temizle
    df.columns = df.columns.str.strip().str.lower()

    # STRING kolonlar
    str_kolonlar = ["cari_kod", "stok_kod", "urun_adi", "birim", "oneri_hafta"]
    for kol in str_kolonlar:
        if kol in df.columns:
            df[kol] = df[kol].astype(str).str.strip()

    # Sayısal kolonlar
    sayi_kolonlar = ["f1_tahmin", "f2_tahmin", "final_tahmin"]
    for kol in sayi_kolonlar:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce")

    # tahmin_tarihi
    if "tahmin_tarihi" in df.columns:
        df["tahmin_tarihi"] = pd.to_datetime(df["tahmin_tarihi"], errors="coerce")

    # forecast_date: oneri_hafta → Pazartesi
    df["forecast_date"] = df["oneri_hafta"].apply(_iso_hafta_to_pazartesi)

    print(f"  → {df.shape[0]:,} satır, {df.shape[1]} sütun")
    print(f"  → forecast_date aralığı: {df['forecast_date'].min().date()} – {df['forecast_date'].max().date()}")
    print(f"  → final_tahmin NaN sayısı: {df['final_tahmin'].isna().sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def yukle_satis(
    dosya: pathlib.Path = SATIS_DOSYA,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    TblGecmisSatisVerileri_haftalik(Sheet1).csv dosyasını chunk'lı okur.

    İşlemler:
    - encoding=cp1254, sep=';', decimal=','.
    - CariKod, StockKod → str (strip ile).
    - Hafta → datetime64.
    - ToplamMiktar, ToplamTutar, SiparisSatirSayisi → float.
    - CariId, StockId → int (mümkünse).

    Döndürür: pd.DataFrame
    """
    print(f"[yukle_satis] Yükleniyor: {dosya}")
    parcalar = []
    toplam_satir = 0

    for parca in pd.read_csv(
        dosya,
        encoding=SATIS_ENCODING,
        sep=SATIS_SEP,
        decimal=SATIS_DECIMAL,
        dtype=str,          # önce hepsini str oku, sonra cast et
        chunksize=chunk_size,
        low_memory=False,
    ):
        # Sütun isimlerini temizle
        parca.columns = parca.columns.str.strip()

        # STRING kolonlar
        for kol in ["CariKod", "StockKod", "StockAd"]:
            if kol in parca.columns:
                parca[kol] = parca[kol].astype(str).str.strip()

        # Hafta → datetime
        if "Hafta" in parca.columns:
            parca["Hafta"] = pd.to_datetime(parca["Hafta"], errors="coerce", dayfirst=True)

        # Sayısal kolonlar
        for kol in ["ToplamMiktar", "ToplamTutar", "SiparisSatirSayisi"]:
            if kol in parca.columns:
                parca[kol] = pd.to_numeric(parca[kol], errors="coerce")

        # ID kolonları
        for kol in ["CariId", "StockId"]:
            if kol in parca.columns:
                parca[kol] = pd.to_numeric(parca[kol], errors="coerce")

        parcalar.append(parca)
        toplam_satir += len(parca)
        print(f"  ... {toplam_satir:,} satır okundu", end="\r")

    df = pd.concat(parcalar, ignore_index=True)
    print(f"\n  → Toplam: {df.shape[0]:,} satır, {df.shape[1]} sütun")
    print(f"  → Hafta aralığı: {df['Hafta'].min().date()} – {df['Hafta'].max().date()}")
    null_counts = df[["CariKod","StockKod","ToplamMiktar","ToplamTutar"]].isna().sum()
    print(f"  → Null sayıları:\n{null_counts.to_string()}")
    return df
