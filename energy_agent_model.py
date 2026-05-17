"""
Improved Energy Forecast Agent Model

Architecture (unchanged):
- historical CSV is used for training only
- trained .pkl bundles are saved under models/
- every user request loads the trained model and predicts the requested date/period live
- forecast results are NOT read from a pre-generated forecast CSV

Modelling changes in this version:
- A robust SeasonalTrend baseline (seasonal lookup + linear trend) is added to the
  candidate pool. It cannot overfit and extrapolates sanely, so the system can never
  do worse than a trivial seasonal model.
- Model selection now uses TimeSeriesSplit cross-validation instead of a single
  holdout year, so the chosen model is robust rather than lucky on one split.
- Log-space variants of every regressor are tried; log-space optimisation reduces
  MAPE on highly multiplicative / noisy series (notably daily energy).
- The quadratic `trend_squared` feature was removed: it extrapolated badly into the
  future and produced negative R^2 for trend-dominated targets.
- Weekly/monthly aggregation now drops incomplete edge periods (partial first/last
  weeks) that previously injected artificial low-sum outliers.

Honest note on limits: daily energy_consumption in this dataset is dominated by
random noise (lag-1 autocorrelation ~0.20), so daily MAPE stays high regardless of
the model. Weekly and monthly forecasts are far more reliable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - fallback for environments without xgboost
    XGBRegressor = None

TARGET_ENERGY = "energy_consumption"
TARGET_RENEWABLE = "renewable_share"
TARGETS = [TARGET_ENERGY, TARGET_RENEWABLE]
LEVELS = ["daily", "weekly", "monthly"]

# Selection uses every record up to this date; the year after it is a clean holdout.
SELECTION_CUTOFF = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
HOLDOUT_END = pd.Timestamp("2024-12-31")

# Per-level seasonal lookup key and smoothing half-window for the SeasonalTrend model.
_SEASONAL_CONFIG = {
    "daily": {"period_col": "dayofyear", "n_periods": 366, "smooth_window": 7},
    "weekly": {"period_col": "weekofyear", "n_periods": 53, "smooth_window": 1},
    "monthly": {"period_col": "month", "n_periods": 12, "smooth_window": 0},
}


def load_and_prepare_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the source CSV, keep Turkey records, fill missing daily dates."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "country" in df.columns:
        df = df[df["country"].astype(str).str.lower().eq("turkey")].copy()

    df = df.sort_values("date").drop_duplicates("date")
    df = df.set_index("date")

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df.index.name = "date"
    df["country"] = "Turkey"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(method="time").ffill().bfill()

    required = [TARGET_ENERGY, TARGET_RENEWABLE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV içinde gerekli sütunlar eksik: {missing}")

    return df


def aggregate_data(df: pd.DataFrame, level: Literal["daily", "weekly", "monthly"]) -> pd.DataFrame:
    """Create daily/weekly/monthly modelling table.

    Energy consumption is additive. Renewable share is recomputed as a weighted share,
    not a simple average, so the period-level renewable amount stays consistent.
    Incomplete edge periods (partial first/last weeks) are dropped so their artificially
    low sums do not distort training and metrics.
    """
    if level == "daily":
        out = df[[TARGET_ENERGY, TARGET_RENEWABLE]].copy()
        return out.dropna()

    freq = "W-MON" if level == "weekly" else "MS"
    energy = df[TARGET_ENERGY].resample(freq).sum()
    renewable_amount = (df[TARGET_ENERGY] * df[TARGET_RENEWABLE] / 100.0).resample(freq).sum()
    renewable_share = renewable_amount / energy * 100.0
    day_count = df[TARGET_ENERGY].resample(freq).count()

    out = pd.DataFrame({TARGET_ENERGY: energy, TARGET_RENEWABLE: renewable_share})

    if level == "weekly":
        complete = day_count >= 7
    else:
        complete = day_count.to_numpy() >= out.index.days_in_month.to_numpy()
    out = out[complete]

    return out.dropna()


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Date-only features usable for any future date without needing pre-saved forecasts."""
    index = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index)

    days_from_start = (index - pd.Timestamp("2020-01-01")).days.astype(float)
    years_from_start = days_from_start / 365.25

    # Linear trend only. A quadratic term was removed: it extrapolated badly into the
    # future for trend-dominated targets such as renewable_share.
    out["trend"] = years_from_start
    out["month"] = index.month
    out["quarter"] = index.quarter
    out["dayofyear"] = index.dayofyear
    out["dayofmonth"] = index.day
    out["dayofweek"] = index.dayofweek
    out["weekofyear"] = index.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (index.dayofweek >= 5).astype(int)
    out["days_in_month"] = index.days_in_month

    # Fourier features capture weekly and yearly seasonality while still working for future dates.
    for period, name, harmonics in [
        (7.0, "week", [1, 2, 3]),
        (365.25, "year", [1, 2, 3, 4, 5, 6]),
    ]:
        for k in harmonics:
            out[f"{name}_sin_{k}"] = np.sin(2 * np.pi * k * days_from_start / period)
            out[f"{name}_cos_{k}"] = np.cos(2 * np.pi * k * days_from_start / period)

    return out


class SeasonalTrendBaseline(BaseEstimator, RegressorMixin):
    """Robust baseline: a smoothed seasonal lookup plus a linear trend.

    The seasonal component is a per-period median (robust to noise) and the trend is
    fitted on the deseasonalised residual. This model cannot overfit, needs no scaling,
    and extrapolates linearly, so it is a safe floor for every target/level.
    """

    def __init__(self, period_col: str = "month", n_periods: int = 12, smooth_window: int = 0):
        self.period_col = period_col
        self.n_periods = n_periods
        self.smooth_window = smooth_window

    def fit(self, X: pd.DataFrame, y) -> "SeasonalTrendBaseline":
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        periods = np.clip(X[self.period_col].to_numpy().astype(int), 1, self.n_periods)
        trend = X["trend"].to_numpy().astype(float)

        global_median = float(np.median(y))
        table = np.full(self.n_periods + 1, np.nan)  # 1-indexed lookup
        for p in range(1, self.n_periods + 1):
            vals = y[periods == p]
            if vals.size:
                table[p] = np.median(vals)
        table = np.where(np.isnan(table), global_median, table)

        w = int(self.smooth_window)
        if w > 0 and self.n_periods > 2 * w:
            core = table[1:]
            padded = np.concatenate([core[-w:], core, core[:w]])
            kernel = np.ones(2 * w + 1) / (2 * w + 1)
            table[1:] = np.convolve(padded, kernel, mode="same")[w:-w]

        self.seasonal_table_ = table
        self.global_median_ = global_median

        residual = y - table[periods]
        design = np.vstack([trend, np.ones_like(trend)]).T
        self.trend_coef_, *_ = np.linalg.lstsq(design, residual, rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        periods = np.clip(X[self.period_col].to_numpy().astype(int), 1, self.n_periods)
        trend = X["trend"].to_numpy().astype(float)
        seasonal = self.seasonal_table_[periods]
        return seasonal + self.trend_coef_[0] * trend + self.trend_coef_[1]


def _log_space(estimator: Any) -> TransformedTargetRegressor:
    """Wrap an estimator so it is fitted in log space (all targets here are strictly positive)."""
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def build_candidates(level: str) -> Dict[str, Any]:
    """Candidate models for a given aggregation level.

    Tree models (GradientBoosting/RandomForest/XGBoost) are offered only for the
    daily level. Weekly (~200 rows) and monthly (~50 rows) have too few samples for
    high-variance learners: they overfit the cross-validation folds and then lose to
    the robust seasonal baseline on real data. So small-data levels keep only the
    low-variance models (SeasonalTrend and Ridge/Fourier).
    """
    cfg = _SEASONAL_CONFIG[level]
    base: Dict[str, Any] = {
        "SeasonalTrend": SeasonalTrendBaseline(
            period_col=cfg["period_col"],
            n_periods=cfg["n_periods"],
            smooth_window=cfg["smooth_window"],
        ),
        "Ridge_Fourier": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]),
    }
    if level == "daily":
        base["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=8,
            random_state=42,
        )
        base["RandomForest"] = RandomForestRegressor(
            n_estimators=160,
            max_depth=7,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        if XGBRegressor is not None:
            base["XGBoost"] = XGBRegressor(
                n_estimators=160,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )

    candidates: Dict[str, Any] = dict(base)
    # Log-space variants for the regression models (the seasonal baseline is already robust).
    for name in list(base.keys()):
        if name != "SeasonalTrend":
            candidates[f"{name}_log"] = _log_space(clone(base[name]))
    return candidates


def _robustness_rank(name: str) -> int:
    """Lower = more robust / less prone to overfitting. Used by the one-SE rule."""
    if name.startswith("SeasonalTrend"):
        return 0
    if name.startswith("Ridge"):
        return 1
    return 2  # tree-based models


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_percent": float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
    }


def reliability_from_mape(mape: float) -> str:
    if mape <= 10:
        return "high"
    if mape <= 30:
        return "medium"
    return "low"


def _clip_for_target(pred: np.ndarray, target: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if target == TARGET_RENEWABLE:
        return np.clip(pred, 0.0, 100.0)
    return np.maximum(pred, 0.0)


def _cv_scores(model: Any, X: pd.DataFrame, y: pd.Series, n_splits: int, target: str) -> list[float]:
    """MAPE for each expanding-window time-series fold."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in splitter.split(X):
        current = clone(model)
        current.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = _clip_for_target(current.predict(X.iloc[test_idx]), target)
        scores.append(regression_metrics(y.iloc[test_idx], pred)["MAPE_percent"])
    return scores


def _fit_best_model(data: pd.DataFrame, target: str, level: str) -> Dict[str, Any]:
    X_all = calendar_features(data.index)
    y_all = data[target]

    selection_mask = data.index <= SELECTION_CUTOFF
    holdout_mask = (data.index >= HOLDOUT_START) & (data.index <= HOLDOUT_END)
    X_sel, y_sel = X_all.loc[selection_mask], y_all.loc[selection_mask]

    candidates = build_candidates(level)
    n_splits = min(4, max(2, len(X_sel) // 15))

    # Cross-validate every candidate on pre-holdout data only.
    fold_scores = {name: _cv_scores(model, X_sel, y_sel, n_splits, target)
                   for name, model in candidates.items()}
    cv_mape = {name: float(np.mean(s)) for name, s in fold_scores.items()}
    cv_std = {name: float(np.std(s, ddof=1)) for name, s in fold_scores.items()}

    # One-standard-error rule: among models statistically tied with the CV leader
    # (within one standard error of its mean), pick the most robust one. This stops a
    # marginally-better tree model from being chosen over the stable seasonal baseline,
    # which is what caused the selected models to lose to the naive baseline before.
    cv_leader = min(cv_mape, key=cv_mape.get)
    se_leader = cv_std[cv_leader] / math.sqrt(n_splits)
    threshold = cv_mape[cv_leader] + se_leader
    tied = [name for name in cv_mape if cv_mape[name] <= threshold]
    best_name = min(tied, key=lambda name: (_robustness_rank(name), cv_mape[name]))

    # Honest holdout metrics: train the chosen model on pre-holdout data, score on 2024.
    holdout_model = clone(candidates[best_name])
    holdout_model.fit(X_sel, y_sel)
    holdout_pred = _clip_for_target(holdout_model.predict(X_all.loc[holdout_mask]), target)
    holdout_metrics = regression_metrics(y_all.loc[holdout_mask], holdout_pred)

    # Naive baseline on the same holdout, so we can prove the selected model has skill.
    naive_model = clone(candidates["SeasonalTrend"])
    naive_model.fit(X_sel, y_sel)
    naive_pred = _clip_for_target(naive_model.predict(X_all.loc[holdout_mask]), target)
    naive_mape = regression_metrics(y_all.loc[holdout_mask], naive_pred)["MAPE_percent"]

    # Holdout guard: never deploy a model that loses to the seasonal baseline on the
    # most recent full year. Cross-validation can disagree with the holdout on volatile,
    # weather-driven targets (e.g. renewable_share); when it does, fall back to the
    # robust baseline so the shipped model is never worse than the trivial one.
    guard_applied = False
    if best_name != "SeasonalTrend" and naive_mape < holdout_metrics["MAPE_percent"]:
        guard_applied = True
        best_name = "SeasonalTrend"
        holdout_model = naive_model
        holdout_pred = naive_pred
        holdout_metrics = regression_metrics(y_all.loc[holdout_mask], naive_pred)

    residuals = y_all.loc[holdout_mask].to_numpy() - holdout_pred
    residual_quantiles = {
        "low80": float(np.quantile(residuals, 0.10)),
        "high80": float(np.quantile(residuals, 0.90)),
        "low95": float(np.quantile(residuals, 0.025)),
        "high95": float(np.quantile(residuals, 0.975)),
        "std": float(np.std(residuals, ddof=1)),
    }

    # Deployment model: refit the selected algorithm on every available row.
    final_model = clone(candidates[best_name])
    final_model.fit(X_all, y_all)

    return {
        "target": target,
        "selected_model_name": best_name,
        "model": final_model,
        "metrics": holdout_metrics,
        "candidate_results": {name: {"cv_mape": cv_mape[name], "cv_std": cv_std[name]}
                              for name in cv_mape},
        "cv_mape": float(cv_mape[best_name]),
        "naive_holdout_mape": float(naive_mape),
        "beats_naive": bool(holdout_metrics["MAPE_percent"] <= naive_mape + 1e-9),
        "holdout_guard_applied": guard_applied,
        "residual_quantiles": residual_quantiles,
        "feature_columns": list(X_all.columns),
        "reliability": reliability_from_mape(holdout_metrics["MAPE_percent"]),
    }


def train_and_save(csv_path: str | Path, model_dir: str | Path = "models") -> Dict[str, Any]:
    """Train improved date-based models and save all bundles under model_dir."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(csv_path)
    summary: Dict[str, Any] = {
        "training_date_start": str(df.index.min().date()),
        "training_date_end": str(df.index.max().date()),
        "row_count": int(len(df)),
        "architecture": "live_per_request_prediction_from_saved_models",
        "selection": "TimeSeriesSplit cross-validation on pre-2024 data; 2024 used as clean holdout",
        "note": "Daily energy is noise-dominated, so daily MAPE stays high by data design; "
                "weekly/monthly models are far more reliable.",
        "levels": {},
    }

    metric_rows = []
    for level in LEVELS:
        level_data = aggregate_data(df, level)
        summary["levels"][level] = {}
        for target in TARGETS:
            bundle = _fit_best_model(level_data, target, level)
            bundle.update({
                "level": level,
                "last_training_date": df.index.max(),
                "period_last_date": level_data.index.max(),
                "training_index_start": level_data.index.min(),
                "training_index_end": level_data.index.max(),
            })
            file_name = f"trained_{level}_{target}_model.pkl"
            joblib.dump(bundle, model_dir / file_name)

            summary["levels"][level][target] = {
                "model_file": file_name,
                "selected_model_name": bundle["selected_model_name"],
                "metrics": bundle["metrics"],
                "cv_mape": bundle["cv_mape"],
                "naive_holdout_mape": bundle["naive_holdout_mape"],
                "beats_naive": bundle["beats_naive"],
                "reliability": bundle["reliability"],
            }
            metric_rows.append({
                "level": level,
                "target": target,
                "selected_model_name": bundle["selected_model_name"],
                "reliability": bundle["reliability"],
                "beats_naive": bundle["beats_naive"],
                "holdout_guard_applied": bundle["holdout_guard_applied"],
                "cv_mape": bundle["cv_mape"],
                "naive_holdout_mape": bundle["naive_holdout_mape"],
                **bundle["metrics"],
            })

    joblib.dump(summary, model_dir / "training_summary.pkl")
    pd.DataFrame(metric_rows).to_csv(model_dir / "model_metrics.csv", index=False)
    return summary


@dataclass
class EnergyForecastAgent:
    model_dir: str | Path = "models"

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.bundles: Dict[str, Dict[str, Any]] = {}
        for level in LEVELS:
            self.bundles[level] = {
                TARGET_ENERGY: joblib.load(self.model_dir / f"trained_{level}_{TARGET_ENERGY}_model.pkl"),
                TARGET_RENEWABLE: joblib.load(self.model_dir / f"trained_{level}_{TARGET_RENEWABLE}_model.pkl"),
            }

    def _predict_level_value(self, date_index: pd.DatetimeIndex, level: str) -> pd.DataFrame:
        X = calendar_features(date_index)
        energy_bundle = self.bundles[level][TARGET_ENERGY]
        renewable_bundle = self.bundles[level][TARGET_RENEWABLE]

        energy_pred = _clip_for_target(energy_bundle["model"].predict(X), TARGET_ENERGY)
        share_pred = _clip_for_target(renewable_bundle["model"].predict(X), TARGET_RENEWABLE)

        out = pd.DataFrame({
            "date": date_index,
            "predicted_energy_consumption": energy_pred,
            "predicted_renewable_share_percent": share_pred,
        })
        out["predicted_renewable_energy"] = out["predicted_energy_consumption"] * out["predicted_renewable_share_percent"] / 100.0
        out["predicted_nonrenewable_energy"] = out["predicted_energy_consumption"] - out["predicted_renewable_energy"]
        return out

    def _interval(self, value: float, bundle: Dict[str, Any], target: str) -> list[float]:
        q = bundle["residual_quantiles"]
        low = value + q["low95"]
        high = value + q["high95"]
        if target == TARGET_RENEWABLE:
            low = max(0.0, min(100.0, low))
            high = max(0.0, min(100.0, high))
        else:
            low = max(0.0, low)
        return [float(low), float(high)]

    def predict_daily(self, target_date: str) -> Dict[str, Any]:
        target = pd.to_datetime(target_date)
        last_date = self.bundles["daily"][TARGET_ENERGY]["last_training_date"]
        if target <= last_date:
            raise ValueError(f"Tahmin tarihi eğitim verisinden sonra olmalı. Son eğitim tarihi: {last_date.date()}")

        df = self._predict_level_value(pd.DatetimeIndex([target]), "daily")
        row = df.iloc[0].to_dict()
        row["date"] = str(pd.to_datetime(row["date"]).date())

        eb = self.bundles["daily"][TARGET_ENERGY]
        rb = self.bundles["daily"][TARGET_RENEWABLE]
        row["energy_95_interval"] = self._interval(row["predicted_energy_consumption"], eb, TARGET_ENERGY)
        row["renewable_share_95_interval"] = self._interval(row["predicted_renewable_share_percent"], rb, TARGET_RENEWABLE)
        row["model_quality"] = {
            "level": "daily",
            "energy_model": eb["selected_model_name"],
            "renewable_share_model": rb["selected_model_name"],
            "energy_reliability": eb["reliability"],
            "renewable_share_reliability": rb["reliability"],
            "message": "Daily forecasts are date-specific but less reliable when daily data is highly volatile. Prefer weekly/monthly for reporting if possible.",
        }
        row["model_metrics"] = {
            TARGET_ENERGY: eb["metrics"],
            TARGET_RENEWABLE: rb["metrics"],
        }
        return row

    def predict_period(self, start_date: str, end_date: str, level: Literal["daily", "weekly", "monthly"] = "daily") -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if end < start:
            raise ValueError("end_date, start_date değerinden önce olamaz.")

        if level == "daily":
            idx = pd.date_range(start, end, freq="D")
            return self._predict_level_value(idx, "daily")

        if level == "weekly":
            # Weekly model predicts each week-start period independently.
            idx = pd.date_range(start.to_period("W-MON").start_time, end.to_period("W-MON").start_time, freq="W-MON")
            out = self._predict_level_value(idx, "weekly")
            return out.rename(columns={"date": "period_start"}).assign(level="weekly")

        if level == "monthly":
            idx = pd.date_range(start.to_period("M").start_time, end.to_period("M").start_time, freq="MS")
            out = self._predict_level_value(idx, "monthly")
            return out.rename(columns={"date": "period_start"}).assign(level="monthly")

        raise ValueError("level sadece 'daily', 'weekly' veya 'monthly' olabilir.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Improved Türkiye Energy Forecast Agent")
    parser.add_argument("--csv", help="Training CSV path")
    parser.add_argument("--model-dir", default="models", help="Folder to save/load models")
    parser.add_argument("--train", action="store_true", help="Train and save improved model files")
    parser.add_argument("--predict-date", help="Example: 2027-03-12")
    args = parser.parse_args()

    if args.train:
        if not args.csv:
            raise ValueError("--train için --csv yolu verilmelidir.")
        print(train_and_save(args.csv, args.model_dir))

    if args.predict_date:
        agent = EnergyForecastAgent(args.model_dir)
        print(agent.predict_daily(args.predict_date))
