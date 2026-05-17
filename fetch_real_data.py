"""
EPİAŞ Şeffaflık Platformu'ndan GERÇEK Türkiye elektrik verisi indirir.

Ne yapar:
  1. EPİAŞ giriş servisinden TGT (Ticket Granting Ticket) alır.
  2. Gerçek zamanlı tüketim (realtime-consumption) verisini ay ay çeker.
  3. Gerçek zamanlı üretim (realtime-generation) verisini ay ay çeker.
  4. Saatlik veriyi günlüğe toplar (eksik saatli günler atılır).
  5. renewable_share'i ağırlıklı pay olarak hesaplar (yenilenebilir üretim / toplam üretim).
  6. Mevcut modelin beklediği şemada CSV yazar:  date, country, energy_consumption, renewable_share

DAYANIKLILIK:
  - Her ay indirildikçe .epias_cache/ klasörüne yazılır.
  - İndirme yarıda kalırsa script'i tekrar çalıştırman yeterli; kaldığı yerden devam eder.
  - Her istek zaman aşımı + yeniden denemeyle korunur (asılı kalmaz).

Kimlik bilgileri ortam değişkenlerinden okunur (koda ASLA yazma):
    Windows PowerShell:
        $env:EPIAS_USERNAME = "mail@ornek.com"
        $env:EPIAS_PASSWORD = "sifreniz"
        python fetch_real_data.py
    Linux/Mac:
        EPIAS_USERNAME="mail@ornek.com" EPIAS_PASSWORD="sifreniz" python fetch_real_data.py

Tarih aralığı / çıktı dosyası:
    python fetch_real_data.py --start 2020-01-01 --end 2024-12-31 --out energy_data_real.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
BASE = "https://seffaflik.epias.com.tr/electricity-service/v1"
CONSUMPTION_URL = f"{BASE}/consumption/data/realtime-consumption"
GENERATION_URL = f"{BASE}/generation/data/realtime-generation"

# EPİAŞ gerçek zamanlı üretim cevabındaki yenilenebilir kaynak alanları (camelCase).
# Cevapta hangileri varsa o kullanılır; eksikler atlanır ve ekranda raporlanır.
RENEWABLE_FIELDS = ["wind", "sun", "geothermal", "biomass", "dammedHydro", "river"]

CACHE_DIR = ".epias_cache"
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 90
MAX_RETRIES = 5


def log(msg: str) -> None:
    """Anlık (tamponlanmamış) ekran çıktısı."""
    print(msg, flush=True)


def get_tgt(username: str, password: str) -> str:
    """EPİAŞ giriş servisinden TGT alır."""
    resp = requests.post(
        LOGIN_URL,
        data={"username": username, "password": password},
        headers={"Accept": "text/plain"},
        timeout=(CONNECT_TIMEOUT, 30),
    )
    if resp.status_code not in (200, 201) or not resp.text.startswith("TGT-"):
        raise SystemExit(
            f"Giriş başarısız (HTTP {resp.status_code}). "
            f"Kullanıcı adı/şifreyi kontrol et.\nCevap: {resp.text[:300]}"
        )
    log("Giriş başarılı, TGT alındı.")
    return resp.text.strip()


class Client:
    """TGT yönetimi, zaman aşımı ve yeniden deneme yapan API istemcisi."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.tgt = get_tgt(username, password)

    def post(self, url: str, body: dict) -> dict:
        """API'ye POST atar; ağ hatası / zaman aşımı / hız sınırına karşı yeniden dener."""
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.post(
                    url,
                    json=body,
                    headers={"Content-Type": "application/json", "TGT": self.tgt},
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )
            except requests.exceptions.RequestException as e:
                last_err = e
                wait = 5 * attempt
                log(f"  ağ hatası ({type(e).__name__}); {wait}s sonra tekrar ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if resp.status_code in (200, 201):
                return resp.json()
            if resp.status_code in (401, 403):
                log("  TGT süresi dolmuş; yeniden giriş yapılıyor")
                self.tgt = get_tgt(self.username, self.password)
                continue
            if resp.status_code == 429:
                wait = 15 * attempt
                log(f"  hız sınırı (HTTP 429); {wait}s bekleniyor ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            wait = 5 * attempt
            log(f"  sunucu hatası ({last_err}); {wait}s sonra tekrar ({attempt}/{MAX_RETRIES})")
            time.sleep(wait)

        raise SystemExit(f"İstek {MAX_RETRIES} denemede başarısız oldu: {url}\nSon hata: {last_err}")


def fetch_range(client: Client, url: str, dataset: str, start: str, end: str) -> pd.DataFrame:
    """Tarih aralığını ay ay çeker. Her ay önbelleğe yazılır; varsa API'ye gidilmez."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    last_end = pd.Timestamp(end) + pd.Timedelta(days=1)  # son günü tam kapsamak için
    frames = []
    for i, month_start in enumerate(months):
        chunk_start = max(month_start, pd.Timestamp(start))
        chunk_end = months[i + 1] if i + 1 < len(months) else last_end
        cache_file = os.path.join(CACHE_DIR, f"{dataset}_{chunk_start:%Y-%m}.json")

        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 2:
            with open(cache_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            log(f"  {chunk_start:%Y-%m} (önbellek): {len(items)} kayıt")
        else:
            body = {
                "startDate": chunk_start.strftime("%Y-%m-%dT00:00:00+03:00"),
                "endDate": chunk_end.strftime("%Y-%m-%dT00:00:00+03:00"),
            }
            res = client.post(url, body)
            items = res.get("items") or res.get("body", {}).get("content") or []
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            log(f"  {chunk_start:%Y-%m}: {len(items)} kayıt indirildi")
            time.sleep(0.2)  # platforma nazik davran

        if items:
            frames.append(pd.DataFrame(items))

    if not frames:
        raise SystemExit(f"Hiç veri dönmedi: {url}")
    return pd.concat(frames, ignore_index=True)


def to_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Cevabı saatlik (Türkiye yerel saati) DatetimeIndex'li tabloya çevirir."""
    dt_col = next((c for c in ("date", "datetime", "tarih") if c in df.columns), None)
    if dt_col is None:
        raise SystemExit(f"Tarih sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}")

    dt = pd.to_datetime(df[dt_col], errors="coerce")
    # EPİAŞ tarihleri +03:00 ofsetli gelir. Saat dilimini Türkiye yerel saatine
    # sabitleyip tz bilgisini kaldırıyoruz; gün sınırları doğru kalsın.
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)

    df = df.copy()
    df["_dt"] = dt
    df = df.dropna(subset=["_dt"]).drop_duplicates("_dt").sort_values("_dt").set_index("_dt")
    return df


def build_daily_table(consumption_raw: pd.DataFrame, generation_raw: pd.DataFrame) -> pd.DataFrame:
    """Saatlik tüketim + üretim verisinden günlük modelleme tablosu üretir."""
    cons = to_hourly_index(consumption_raw)
    gen = to_hourly_index(generation_raw)

    # --- Tüketim: değer sütununu tespit et ---
    cons_candidates = [c for c in ("consumption", "consumptionQuantity", "tuketim") if c in cons.columns]
    if cons_candidates:
        cons_col = cons_candidates[0]
    else:
        numeric = cons.select_dtypes("number").columns.tolist()
        if not numeric:
            raise SystemExit(f"Tüketim değeri bulunamadı. Sütunlar: {list(cons.columns)}")
        cons_col = numeric[0]
    log(f"\nTüketim sütunu: '{cons_col}'")
    hourly_consumption = pd.to_numeric(cons[cons_col], errors="coerce")

    # --- Üretim: toplam ve yenilenebilir ---
    log(f"Üretim cevabındaki sütunlar: {list(gen.columns)}")
    if "total" in gen.columns:
        hourly_total = pd.to_numeric(gen["total"], errors="coerce")
    else:
        hourly_total = gen.select_dtypes("number").sum(axis=1)
        log("UYARI: 'total' sütunu yok; tüm sayısal üretim sütunları toplandı.")

    found = [f for f in RENEWABLE_FIELDS if f in gen.columns]
    missing = [f for f in RENEWABLE_FIELDS if f not in gen.columns]
    if not found:
        raise SystemExit(f"Yenilenebilir kaynak sütunu bulunamadı. Sütunlar: {list(gen.columns)}")
    log(f"Yenilenebilir kaynaklar (kullanılan): {found}")
    if missing:
        log(f"Yenilenebilir kaynaklar (cevapta yok, atlandı): {missing}")
    hourly_renewable = gen[found].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # --- Saatlikten günlüğe topla; eksik saatli günleri ele ---
    daily = pd.DataFrame({
        "energy_consumption": hourly_consumption.resample("D").sum(min_count=1),
        "cons_hours": hourly_consumption.resample("D").count(),
        "total_gen": hourly_total.resample("D").sum(min_count=1),
        "renewable_gen": hourly_renewable.resample("D").sum(min_count=1),
        "gen_hours": hourly_total.resample("D").count(),
    })
    incomplete = (daily["cons_hours"] < 24) | (daily["gen_hours"] < 24)
    if incomplete.any():
        log(f"\n{int(incomplete.sum())} gün 24 saatten az veri içeriyor ve atlandı "
            "(model bu günleri zaten enterpolasyonla doldurur).")
    daily = daily[~incomplete]

    daily["renewable_share"] = daily["renewable_gen"] / daily["total_gen"] * 100.0
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["energy_consumption", "renewable_share"]
    )

    out = daily[["energy_consumption", "renewable_share"]].copy()
    out.index.name = "date"
    out = out.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.insert(1, "country", "Turkey")
    return out


def main():
    parser = argparse.ArgumentParser(description="EPİAŞ gerçek veri indirme")
    parser.add_argument("--start", default="2020-01-01", help="Başlangıç tarihi (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Bitiş tarihi (YYYY-MM-DD)")
    parser.add_argument("--out", default="energy_data_real.csv", help="Çıktı CSV dosyası")
    args = parser.parse_args()

    username = os.environ.get("EPIAS_USERNAME")
    password = os.environ.get("EPIAS_PASSWORD")
    if not username or not password:
        sys.exit(
            "EPIAS_USERNAME ve EPIAS_PASSWORD ortam değişkenleri tanımlı değil.\n"
            "PowerShell:  $env:EPIAS_USERNAME='mail'; $env:EPIAS_PASSWORD='sifre'; python fetch_real_data.py"
        )

    client = Client(username, password)

    log(f"\nTüketim verisi indiriliyor ({args.start} -> {args.end})...")
    consumption_raw = fetch_range(client, CONSUMPTION_URL, "consumption", args.start, args.end)

    log(f"\nÜretim verisi indiriliyor ({args.start} -> {args.end})...")
    generation_raw = fetch_range(client, GENERATION_URL, "generation", args.start, args.end)

    daily = build_daily_table(consumption_raw, generation_raw)
    # Aralık dışına taşan günleri kırp
    daily = daily[(daily["date"] >= args.start) & (daily["date"] <= args.end)].reset_index(drop=True)

    idx = pd.to_datetime(daily["date"])
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    missing_days = len(full) - len(idx)

    daily.to_csv(args.out, index=False)

    log("\n" + "=" * 55)
    log(f"Yazıldı: {args.out}")
    log(f"Satır sayısı        : {len(daily)}")
    log(f"Tarih aralığı       : {daily['date'].iloc[0]} -> {daily['date'].iloc[-1]}")
    log(f"Eksik gün           : {missing_days}")
    log(f"Ort. enerji tüketimi: {daily['energy_consumption'].mean():,.0f} MWh/gün")
    log(f"Ort. yenilenebilir  : %{daily['renewable_share'].mean():.2f}")
    log("=" * 55)
    log("\nİlk 3 satır:")
    log(daily.head(3).to_string(index=False))
    log(f"\n('{CACHE_DIR}' klasörünü silebilirsin; sadece indirme önbelleğidir.)")
    log("Sonraki adım: train_models.py içinde csv_path'i bu dosyaya çevir ve yeniden eğit.")


if __name__ == "__main__":
    main()
