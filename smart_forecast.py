"""
Akıllı yönlendirici (smart router).

Takvim modeli (energy_agent_model.EnergyForecastAgent) ile lag modelini
(lag_forecast.LagForecaster) tek bir arayüzde birleştirir. Kullanıcı yalnızca
tarih/dönem verir; yönlendirici seviye, hedef ve tahmin ufkuna göre HER HEDEF
için en isabetli modeli otomatik seçer. Mevcut iki model dosyasına dokunmaz;
yalnızca onları çağırır.

Seçim mantığı (README "Hangi modeli seçmeli?" tablosundan):
  - Günlük (her iki hedef) ............... lag modeli
  - Haftalık/aylık yenilenebilir pay ..... lag modeli
  - Haftalık/aylık tüketim ............... takvim modeli (SeasonalTrend)
  - Ufuk > ~1 yıl (her seviye/hedef) ..... takvim modeli

Neden hedef bazında seçim? Haftalık/aylık bir dönem için tüketim ile
yenilenebilir pay farklı modellere gider: tüketim takvim modelinden, yenilenebilir
pay lag modelinden alınır ve tek tabloda birleştirilir.

Neden ufuk eşiği? Lag modeli özyinelemelidir; uzak horizonda lag değerleri
tahminden beslendiği için doğal olarak mevsimsel ortalamaya yakınsar ve takvim
modeline üstünlüğü kalmaz. Eşikler sezgiseldir: lag metrikleri 2024 holdout
yılı (~1 yıl) üzerinde ölçüldü, ötesi doğrulanmadı.

Not: Tahmin güven aralıkları yalnızca takvim modelinde vardır. Aralık gerekiyorsa
EnergyForecastAgent doğrudan kullanılmalıdır; bu yönlendirici tek tip (uniform)
bir çıktı şeması sunar.

Kullanım:
  python smart_forecast.py --date 2025-03-12
  python smart_forecast.py --start 2025-03-01 --end 2025-04-15 --level weekly
  python smart_forecast.py                      (demo: birkaç örnek)

  from smart_forecast import SmartForecaster
  sf = SmartForecaster(model_dir="models")
  sf.predict_daily("2025-03-12")
  sf.predict_period("2025-03-01", "2025-08-31", level="monthly")
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from energy_agent_model import (
    EnergyForecastAgent,
    LEVELS,
    TARGET_ENERGY,
    TARGET_RENEWABLE,
    reliability_from_mape,
)
from lag_forecast import LagForecaster

# Ufuk eşikleri (dönem cinsinden, ~1 yıl). Bu ufkun ötesinde lag modeli
# özyinelemeli olarak mevsimsel ortalamaya yakınsadığından takvim modeline
# üstünlüğü kalmaz; eşik sezgiseldir (lag metrikleri 2024 holdout = ~1 yıl).
LONG_HORIZON = {"daily": 365, "weekly": 52, "monthly": 12}


class SmartForecaster:
    """İki modeli sarar ve her istek için hedef bazında en iyi modeli seçer."""

    def __init__(self, model_dir: str | Path = "models"):
        self.calendar = EnergyForecastAgent(model_dir=model_dir)
        self.lag = LagForecaster(model_dir=model_dir)

    # ----------------------------------------------------------------- yardımcılar
    def _last_real(self, level: str) -> pd.Timestamp:
        """O seviyedeki son gerçek gözlem tarihi (lag modelinin özyineleme başlangıcı)."""
        return pd.Timestamp(self.lag.bundles[f"{level}_{TARGET_ENERGY}"]["last_date"]).normalize()

    def _horizon(self, date: Any, level: str) -> int:
        """Son gerçek veriden verilen tarihe kadarki dönem (gün/hafta/ay) sayısı."""
        last = self._last_real(level)
        d = pd.Timestamp(date).normalize()
        if level == "daily":
            return (d - last).days
        if level == "weekly":
            return round((d - last).days / 7)
        return (d.year - last.year) * 12 + (d.month - last.month)

    def _choose(self, level: str, target: str, horizon: int) -> Tuple[str, str]:
        """Bir (seviye, hedef, ufuk) için model adını ve gerekçesini döndürür."""
        if horizon > LONG_HORIZON[level]:
            return "calendar", (
                f"ufuk {horizon} dönem, {LONG_HORIZON[level]} eşiğini aşıyor: lag bu "
                "mesafede mevsimsel ortalamaya yakınsar, takvim modeli kullanılır"
            )
        if level == "daily":
            return "lag", (
                "günlük seviye: ardışık günler güçlü ilişkili (lag-1 otokorelasyon "
                "~0.84/0.91), lag modeli kısa vadede daha isabetli"
            )
        if target == TARGET_RENEWABLE:
            return "lag", (
                f"{level} yenilenebilir pay: lag modeli takvim modelinden daha isabetli"
            )
        return "calendar", (
            f"{level} tüketim: seri neredeyse saf mevsimsel, SeasonalTrend daha isabetli"
        )

    def _reliability(self, level: str, target: str, model: str) -> str:
        """Seçilen modelin o hedefteki güvenilirlik etiketi (high/medium/low)."""
        if model == "calendar":
            return self.calendar.bundles[level][target]["reliability"]
        mape = self.lag.bundles[f"{level}_{target}"]["metrics"]["mape_recursive"]
        return reliability_from_mape(mape)

    def _calendar_frame(self, level: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Takvim modelinden normalize edilmiş tahmin tablosu (date + iki hedef)."""
        df = self.calendar.predict_period(str(start.date()), str(end.date()), level=level)
        if "period_start" in df.columns:
            df = df.rename(columns={"period_start": "date"})
        return df[["date", "predicted_energy_consumption", "predicted_renewable_share_percent"]]

    def _lag_frame(self, level: str, end: pd.Timestamp) -> pd.DataFrame:
        """Lag modelinden, end tarihini kapsayacak kadar dönem tahmin eder."""
        n_periods = max(self._horizon(end, level) + 1, 1)
        df = self.lag.forecast(level, n_periods)
        return df[["date", "predicted_energy_consumption", "predicted_renewable_share_percent"]]

    # ----------------------------------------------------------------- genel API
    def route(self, level: str, horizon: int) -> Dict[str, Any]:
        """Tahmin yapmadan, verilen ufuk için yönlendirme kararını döndürür."""
        e_model, e_reason = self._choose(level, TARGET_ENERGY, horizon)
        r_model, r_reason = self._choose(level, TARGET_RENEWABLE, horizon)
        return {
            "level": level,
            "horizon_periods": horizon,
            "energy_consumption": {
                "model": e_model,
                "reason": e_reason,
                "reliability": self._reliability(level, TARGET_ENERGY, e_model),
            },
            "renewable_share": {
                "model": r_model,
                "reason": r_reason,
                "reliability": self._reliability(level, TARGET_RENEWABLE, r_model),
            },
        }

    def predict_period(self, start: Any, end: Any,
                       level: str = "daily") -> pd.DataFrame:
        """[start, end] aralığı için, her hedefi en iyi modelden alıp birleştirir."""
        if level not in LEVELS:
            raise ValueError(f"level sadece {LEVELS} olabilir.")
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end, start değerinden önce olamaz.")

        last_real = self._last_real(level)
        if self._horizon(start_ts, level) < 1:
            raise ValueError(
                f"start ({start_ts.date()}) son gerçek veriden ({last_real.date()}) "
                "sonra olmalı."
            )

        horizon = self._horizon(end_ts, level)
        e_model, _ = self._choose(level, TARGET_ENERGY, horizon)
        r_model, _ = self._choose(level, TARGET_RENEWABLE, horizon)

        frames: Dict[str, pd.DataFrame] = {}
        if "calendar" in (e_model, r_model):
            frames["calendar"] = self._calendar_frame(level, start_ts, end_ts)
        if "lag" in (e_model, r_model):
            frames["lag"] = self._lag_frame(level, end_ts)

        for name, frame in frames.items():
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frames[name] = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]

        energy = frames[e_model][["date", "predicted_energy_consumption"]]
        share = frames[r_model][["date", "predicted_renewable_share_percent"]]
        out = energy.merge(share, on="date", how="inner").sort_values("date").reset_index(drop=True)
        if out.empty:
            raise ValueError("Verilen aralık için tahmin üretilemedi (aralığı kontrol edin).")

        out["predicted_renewable_energy"] = (
            out["predicted_energy_consumption"] * out["predicted_renewable_share_percent"] / 100.0
        )
        out["predicted_nonrenewable_energy"] = (
            out["predicted_energy_consumption"] - out["predicted_renewable_energy"]
        )
        out["horizon"] = [self._horizon(d, level) for d in out["date"]]
        out["energy_model"] = e_model
        out["renewable_model"] = r_model
        return out

    def predict_daily(self, date: Any) -> Dict[str, Any]:
        """Tek bir gün için tahmin + hangi modelin neden seçildiğini döndürür."""
        ts = pd.Timestamp(date).normalize()
        horizon = self._horizon(ts, "daily")
        if horizon < 1:
            raise ValueError(
                f"Tahmin tarihi son gerçek veriden ({self._last_real('daily').date()}) "
                "sonra olmalı."
            )

        row = self.predict_period(ts, ts, level="daily").iloc[0]
        result = {
            "date": str(ts.date()),
            "level": "daily",
            "predicted_energy_consumption": float(row["predicted_energy_consumption"]),
            "predicted_renewable_share_percent": float(row["predicted_renewable_share_percent"]),
            "predicted_renewable_energy": float(row["predicted_renewable_energy"]),
            "predicted_nonrenewable_energy": float(row["predicted_nonrenewable_energy"]),
            "routing": self.route("daily", horizon),
        }
        return result


def _print_routing(routing: Dict[str, Any]) -> None:
    print(f"  ufuk: {routing['horizon_periods']} dönem")
    for target in (TARGET_ENERGY, TARGET_RENEWABLE):
        info = routing[target]
        print(f"  {target:18s} -> {info['model']:8s} "
              f"(güvenilirlik: {info['reliability']}) | {info['reason']}")


if __name__ == "__main__":
    import sys

    # Windows konsolunda Türkçe karakterlerin bozulmaması için.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description="Akıllı yönlendirici: en isabetli modeli otomatik seçer")
    parser.add_argument("--model-dir", default="models", help="Model klasörü")
    parser.add_argument("--date", help="Tek gün tahmini, örn. 2025-03-12")
    parser.add_argument("--start", help="Dönem başlangıcı")
    parser.add_argument("--end", help="Dönem bitişi")
    parser.add_argument("--level", default="daily", choices=LEVELS, help="Dönem seviyesi")
    args = parser.parse_args()

    sf = SmartForecaster(args.model_dir)

    if args.date:
        res = sf.predict_daily(args.date)
        print(f"\n{res['date']} tahmini")
        print(f"  enerji tüketimi      : {res['predicted_energy_consumption']:.2f}")
        print(f"  yenilenebilir pay    : %{res['predicted_renewable_share_percent']:.2f}")
        print(f"  yenilenebilir enerji : {res['predicted_renewable_energy']:.2f}")
        print(f"  yenilenebilir olmayan: {res['predicted_nonrenewable_energy']:.2f}")
        print("yönlendirme:")
        _print_routing(res["routing"])
    elif args.start and args.end:
        fc = sf.predict_period(args.start, args.end, level=args.level)
        print(fc.to_string(index=False))
    else:
        print("=== Tek gün — yakın tarih (2025-03-12) ===")
        _print_routing(sf.predict_daily("2025-03-12")["routing"])
        print("\n=== Tek gün — uzak tarih, >1 yıl (2027-03-12) ===")
        _print_routing(sf.predict_daily("2027-03-12")["routing"])
        print("\n=== Haftalık dönem (2025-03-01 .. 2025-04-15) ===")
        print(sf.predict_period("2025-03-01", "2025-04-15", level="weekly").to_string(index=False))
        print("\n=== Aylık dönem (2025-03-01 .. 2025-08-31) ===")
        print(sf.predict_period("2025-03-01", "2025-08-31", level="monthly").to_string(index=False))
