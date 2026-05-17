"""
Lag tabanlı kısa-vadeli tahmin modeli.

Bu modül energy_agent_model.py'yi tamamlar, onun yerine geçmez:
- energy_agent_model.py : sadece takvim özellikleri kullanır; herhangi bir uzak
  gelecek tarihi için iklimsel (climatological) tahmin verir.
- lag_forecast.py       : geçmiş değer (lag) özellikleri kullanır; son gerçek
  veriden başlayarak özyinelemeli (recursive) tahmin yapar. Kısa vadede çok daha
  isabetlidir; uzak horizonda doğal olarak mevsimsel ortalamaya yakınsar.

Neden lag? Gerçek veride lag-1 otokorelasyonu yüksek (enerji ~0.84,
renewable_share ~0.91). "Dünkü değer" bugünü güçlü tahmin eder; takvim modeli
bu sinyali hiç kullanmaz.

Neden LSTM değil? ~1800 günlük / ~260 haftalık / ~60 aylık satırda LSTM aşırı
öğrenir. Bu veri boyutunda lag özellikli düzenlileştirilmiş lineer model (Ridge)
hem daha kararlı hem genelde daha isabetlidir; özyinelemede de kararlıdır.

Değerlendirme dürüstlüğü: lag modelinin hatası horizona bağlıdır.
- 1 adım ileri  (gün/hafta/ay öncesi): en iyi durum, lag gerçek değer.
- Tüm yıl özyinelemeli: en kötü durum, lag'lar tahminden besleniyor.
Gerçek kullanım bu ikisinin arasındadır. Her ikisi de raporlanır.

Eğitim:   python lag_forecast.py --train --csv energy_data_real.csv
Tahmin:   python lag_forecast.py            (demo: sonraki dönemleri tahmin eder)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_agent_model import (
    TARGET_ENERGY,
    TARGET_RENEWABLE,
    TARGETS,
    LEVELS,
    HOLDOUT_START,
    load_and_prepare_data,
    aggregate_data,
)

# Her seviye için: pandas frekansı, lag gecikmeleri, hareketli ortalama pencereleri.
LAG_CONFIG = {
    "daily": {"freq": "D", "lags": [1, 2, 3, 7, 14], "rolls": [7, 28]},
    "weekly": {"freq": "W-MON", "lags": [1, 2, 3, 4, 52], "rolls": [4]},
    "monthly": {"freq": "MS", "lags": [1, 2, 3, 12], "rolls": [3]},
}


def _calendar_block(index: pd.DatetimeIndex, level: str) -> pd.DataFrame:
    """Seviyeye uygun, kompakt takvim özellikleri (lag özelliklerine ek olarak)."""
    idx = pd.DatetimeIndex(index)
    days = (idx - pd.Timestamp("2020-01-01")).days.astype(float)
    out = pd.DataFrame(index=idx)
    out["trend"] = days / 365.25  # lineer trend (yıllık büyümeyi yakalar)

    if level == "daily":
        out["dayofweek"] = idx.dayofweek
        out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        out["month"] = idx.month
        for k in (1, 2, 3):
            out[f"yr_sin{k}"] = np.sin(2 * np.pi * k * days / 365.25)
            out[f"yr_cos{k}"] = np.cos(2 * np.pi * k * days / 365.25)
        for k in (1, 2):
            out[f"wk_sin{k}"] = np.sin(2 * np.pi * k * days / 7)
            out[f"wk_cos{k}"] = np.cos(2 * np.pi * k * days / 7)
    else:
        out["month"] = idx.month
        harmonics = (1, 2, 3) if level == "weekly" else (1, 2)
        for k in harmonics:
            out[f"yr_sin{k}"] = np.sin(2 * np.pi * k * days / 365.25)
            out[f"yr_cos{k}"] = np.cos(2 * np.pi * k * days / 365.25)
    return out


def build_features(series: pd.Series, level: str) -> pd.DataFrame:
    """series.index'in tamamı için lag + hareketli ortalama + takvim özellikleri.

    Bir tarihin kendi değeri özellik olarak kullanılmaz (sadece shift'lenmiş geçmiş),
    bu yüzden tablo hem eğitim hem özyinelemeli tahmin için tutarlıdır.
    """
    cfg = LAG_CONFIG[level]
    X = _calendar_block(series.index, level)
    for lag in cfg["lags"]:
        X[f"lag_{lag}"] = series.shift(lag).to_numpy()
    for window in cfg["rolls"]:
        X[f"roll_{window}"] = series.shift(1).rolling(window).mean().to_numpy()
    return X


def _make_model() -> Pipeline:
    """Ölçekleme + RidgeCV (alpha çapraz doğrulamayla seçilir). Lineer = özyinelemede kararlı."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-1, 3, 12))),
    ])


def _clip(values: np.ndarray, target: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target == TARGET_RENEWABLE:
        return np.clip(values, 0.0, 100.0)
    return np.maximum(values, 0.0)


def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _fit(series: pd.Series, level: str, target: str, upto: pd.Timestamp | None = None) -> Pipeline:
    """Lag modelini eğitir. upto verilirse sadece o tarihe kadarki veriyle eğitir."""
    X = build_features(series, level)
    valid = X.dropna()
    y = series.loc[valid.index]
    if upto is not None:
        mask = valid.index < upto
        valid, y = valid.loc[mask], y.loc[mask]
    model = _make_model()
    model.fit(valid, y)
    return model


def recursive_forecast(model: Pipeline, history: pd.Series, level: str,
                       target: str, n_periods: int) -> pd.Series:
    """Son gerçek veriden başlayarak n_periods dönemi özyinelemeli tahmin eder."""
    freq = LAG_CONFIG[level]["freq"]
    series = history.astype(float).copy()
    future = pd.date_range(series.index[-1], periods=n_periods + 1, freq=freq)[1:]
    preds = {}
    for date in future:
        series.loc[date] = np.nan
        row = build_features(series, level).loc[[date]]
        yhat = _clip(model.predict(row), target)[0]
        series.loc[date] = yhat
        preds[date] = yhat
    return pd.Series(preds, name=target)


def _backtest(series: pd.Series, level: str, target: str,
              holdout_start: pd.Timestamp) -> Dict[str, float]:
    """İki dürüst hata ölçümü: 1 adım ileri ve tüm holdout boyunca özyinelemeli."""
    model = _fit(series, level, target, upto=holdout_start)
    holdout_dates = series.index[series.index >= holdout_start]

    # --- 1 adım ileri: her tahmin gerçek geçmiş değerleri kullanır ---
    X_true = build_features(series, level)
    one_step_idx = [d for d in holdout_dates if d in X_true.dropna().index]
    one_step_pred = _clip(model.predict(X_true.loc[one_step_idx]), target)
    one_step = _mape(series.loc[one_step_idx], one_step_pred)

    # --- Tüm holdout boyunca özyinelemeli: lag'lar tahminden beslenir (en kötü durum) ---
    known = series[series.index < holdout_start].astype(float).copy()
    rec_pred = []
    for date in holdout_dates:
        known.loc[date] = np.nan
        row = build_features(known, level).loc[[date]]
        yhat = _clip(model.predict(row), target)[0]
        known.loc[date] = yhat
        rec_pred.append(yhat)
    recursive = _mape(series.loc[holdout_dates], rec_pred)
    r2_rec = float(r2_score(series.loc[holdout_dates], rec_pred))

    return {
        "mape_1step": one_step,
        "mape_recursive": recursive,
        "r2_recursive": r2_rec,
        "n_holdout": int(len(holdout_dates)),
    }


def train_and_save(csv_path: str | Path = "energy_data_real.csv",
                   model_dir: str | Path = "models") -> Dict[str, Any]:
    """Tüm seviye/hedefler için lag modellerini eğitir, backtest yapar ve kaydeder."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare_data(csv_path)

    rows = []
    for level in LEVELS:
        level_data = aggregate_data(df, level)
        for target in TARGETS:
            series = level_data[target].astype(float)
            metrics = _backtest(series, level, target, HOLDOUT_START)

            # Dağıtım modeli: tüm veriyle eğitilir.
            final_model = _fit(series, level, target)
            bundle = {
                "level": level,
                "target": target,
                "model": final_model,
                "history": series,
                "lag_config": LAG_CONFIG[level],
                "metrics": metrics,
                "last_date": series.index[-1],
            }
            joblib.dump(bundle, model_dir / f"lag_{level}_{target}_model.pkl")
            rows.append({"level": level, "target": target, **metrics})
            print(f"{level:8s} {target:18s}  1-adım MAPE=%{metrics['mape_1step']:.2f}   "
                  f"özyinelemeli MAPE=%{metrics['mape_recursive']:.2f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(model_dir / "lag_model_metrics.csv", index=False)
    return {"metrics": metrics_df}


class LagForecaster:
    """Kaydedilmiş lag modellerini yükler ve sonraki dönemleri tahmin eder."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.bundles: Dict[str, Dict[str, Any]] = {}
        for level in LEVELS:
            for target in TARGETS:
                path = self.model_dir / f"lag_{level}_{target}_model.pkl"
                self.bundles[f"{level}_{target}"] = joblib.load(path)

    def forecast(self, level: str, n_periods: int) -> pd.DataFrame:
        """Son gerçek veriden sonraki n_periods dönem için enerji + yenilenebilir tahmini."""
        eb = self.bundles[f"{level}_{TARGET_ENERGY}"]
        rb = self.bundles[f"{level}_{TARGET_RENEWABLE}"]
        energy = recursive_forecast(eb["model"], eb["history"], level, TARGET_ENERGY, n_periods)
        share = recursive_forecast(rb["model"], rb["history"], level, TARGET_RENEWABLE, n_periods)
        out = pd.DataFrame({
            "date": energy.index,
            "predicted_energy_consumption": energy.to_numpy(),
            "predicted_renewable_share_percent": share.to_numpy(),
        })
        out["predicted_renewable_energy"] = (
            out["predicted_energy_consumption"] * out["predicted_renewable_share_percent"] / 100.0
        )
        out["horizon"] = np.arange(1, n_periods + 1)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lag tabanlı kısa-vadeli enerji tahmini")
    parser.add_argument("--csv", default="energy_data_real.csv", help="Eğitim CSV yolu")
    parser.add_argument("--model-dir", default="models", help="Model klasörü")
    parser.add_argument("--train", action="store_true", help="Lag modellerini eğit ve kaydet")
    args = parser.parse_args()

    if args.train:
        print("Lag modelleri eğitiliyor...\n")
        train_and_save(args.csv, args.model_dir)
        print(f"\nModeller ve metrikler '{args.model_dir}' klasörüne kaydedildi.")
    else:
        agent = LagForecaster(args.model_dir)
        for level, n in [("daily", 7), ("weekly", 4), ("monthly", 3)]:
            print(f"\n=== {level.upper()} — sonraki {n} dönem ===")
            fc = agent.forecast(level, n)
            print(fc[["date", "predicted_energy_consumption",
                      "predicted_renewable_share_percent"]].to_string(index=False))
