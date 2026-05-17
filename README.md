# Türkiye Elektrik Tüketimi ve Yenilenebilir Enerji Payı Tahmini

Bu depo, Türkiye'nin **elektrik tüketimini** (`energy_consumption`) ve **yenilenebilir
enerji payını** (`renewable_share`) günlük, haftalık ve aylık çözünürlükte tahmin eden
bir makine öğrenmesi sistemidir. Modeller, EPİAŞ Şeffaflık Platformu'ndan alınan gerçek
ölçüm verileriyle eğitilir. Sistem, farklı tahmin ihtiyaçları için iki tamamlayıcı
modelleme yaklaşımı sunar.

---

## Genel Bakış

Tahmin probleminin doğası, sorulan tarihin ne kadar uzakta olduğuna göre değişir. Yakın
gelecek için son gözlemler güçlü bilgi taşır; uzak gelecek içinse yalnızca mevsimsel
örüntüye dayanılabilir. Bu nedenle proje iki ayrı modülden oluşur:

| Modül | Yaklaşım | Uygun kullanım |
|---|---|---|
| `energy_agent_model.py` | Takvim tabanlı (iklimsel) | Herhangi bir uzak gelecek tarihi |
| `lag_forecast.py` | Lag tabanlı (özyinelemeli) | Kısa vadeli (yakın gelecek) tahmin |

İki modül birbirini tamamlar; biri diğerinin yerine geçmez.

---

## Proje Yapısı

```
.
├── fetch_real_data.py      # EPİAŞ Şeffaflık Platformu'ndan gerçek veri indirir
├── energy_agent_model.py   # Takvim tabanlı model + tahmin ajanı (EnergyForecastAgent)
├── lag_forecast.py         # Lag tabanlı kısa-vadeli model (LagForecaster)
├── train_models.py         # Takvim modelini eğitme betiği
├── predict.py              # Örnek tahmin betiği
├── ask_agent.py            # Etkileşimli (kullanıcıdan tarih alan) tahmin
├── energy_data_real.csv    # Gerçek veri (fetch_real_data.py tarafından üretilir)
├── requirements.txt        # Python bağımlılıkları
└── models/                 # Eğitilmiş modeller (.pkl) ve metrik dosyaları
```

---

## Veri

### Kaynak

Veriler **EPİAŞ Şeffaflık Platformu**'ndan (<https://seffaflik.epias.com.tr>) alınır.
Bu platform, Türkiye elektrik piyasasının resmî ve halka açık veri kaynağıdır. Kullanılan
iki servis:

- **Gerçek zamanlı tüketim** (`realtime-consumption`) — saatlik elektrik tüketimi.
- **Gerçek zamanlı üretim** (`realtime-generation`) — kaynak bazlı saatlik üretim.

`renewable_share`, üretim verisinden ağırlıklı pay olarak hesaplanır:

```
renewable_share = (Σ yenilenebilir üretim / Σ toplam üretim) × 100
```

Yenilenebilir kaynaklar: rüzgâr, güneş, jeotermal, biyokütle, barajlı hidroelektrik ve
akarsu (nehir tipi) santraller.

### Veri çekme

1. <https://seffaflik.epias.com.tr> adresinden ücretsiz bir hesap oluşturun.
2. Kimlik bilgilerini ortam değişkeni olarak tanımlayın :

   **Windows PowerShell**
   ```powershell
   $env:EPIAS_USERNAME = "kullanici@ornek.com"
   $env:EPIAS_PASSWORD = "parolaniz"
   ```
   **Linux / macOS**
   ```bash
   export EPIAS_USERNAME="kullanici@ornek.com"
   export EPIAS_PASSWORD="parolaniz"
   ```

3. Veri çekme betiğini çalıştırın:
   ```bash
   python fetch_real_data.py --start 2020-01-01 --end 2024-12-31 --out energy_data_real.csv
   ```

`fetch_real_data.py` şu işlemleri yapar:

- EPİAŞ giriş servisinden bir oturum bileti (TGT) alır.
- Tüketim ve üretim verisini ay ay indirir.
- Saatlik veriyi günlüğe toplar; 24 saatten az veri içeren günleri eler.
- Ağırlıklı `renewable_share` değerini hesaplar.
- Modelin beklediği şemada CSV üretir: `date, country, energy_consumption, renewable_share`.

Betik dayanıklıdır: indirilen her ay `.epias_cache/` klasörüne yazılır. İndirme yarıda
kalırsa betiğin yeniden çalıştırılması yeterlidir; kaldığı yerden devam eder. Her istek
zaman aşımı ve yeniden deneme ile korunur.

---

## Yöntem

### 1. Takvim tabanlı model — `energy_agent_model.py`

Bu model yalnızca **tarihten türetilen özellikler** kullanır: lineer trend, ay, çeyrek,
haftanın günü, hafta sonu göstergesi ve haftalık/yıllık mevsimselliği temsil eden Fourier
terimleri. Geçmiş gözlem değeri kullanılmadığı için model, herhangi bir uzak gelecek
tarihi için doğrudan tahmin üretebilir (iklimsel tahmin).

Günlük, haftalık ve aylık seviyeler için ayrı modeller eğitilir. Model seçimi şu üç
mekanizmayla sağlamlaştırılmıştır:

- **Zaman serisi çapraz doğrulaması** (`TimeSeriesSplit`): adaylar 2023 sonuna kadarki
  veriyle değerlendirilir; 2024 yılı temiz holdout olarak ölçüm için ayrılır.
- **Bir-standart-hata kuralı**: çapraz doğrulama lideriyle istatistiksel olarak "berabere"
  olan (bir standart hata içindeki) adaylar arasından en sağlam, aşırı öğrenmeye en az
  yatkın model seçilir.
- **Holdout güvenlik kapısı**: seçilen model son tam yılda naive mevsimsel taban
  çizgisini geçemezse, otomatik olarak taban çizgisine düşülür. Böylece dağıtılan model
  hiçbir zaman basit modelden kötü olamaz.

Az veri içeren seviyelerde (haftalık ~200, aylık ~50 gözlem) ağaç tabanlı modeller aday
havuzundan çıkarılmıştır; bu modeller bu boyutta aşırı öğrenmektedir.

### 2. Lag tabanlı model — `lag_forecast.py`

Bu model **geçmiş değer (lag)** özellikleri kullanır: önceki dönemlerin değerleri ve
hareketli ortalamaları, takvim özellikleriyle birlikte. Tahmin **özyinelemelidir**: son
gerçek gözlemden başlayarak adım adım ilerlenir ve her adımın tahmini bir sonraki adımın
girdisine beslenir.

Gerçek elektrik verisinde ardışık dönemler güçlü biçimde ilişkilidir (lag-1 otokorelasyonu
enerji için ~0.84, yenilenebilir pay için ~0.91). Lag modeli bu bilgiyi kullanır ve kısa
vadede belirgin biçimde daha isabetlidir. Uzak horizonda lag değerleri tahminden
beslendiği için model doğal olarak mevsimsel ortalamaya yakınsar.

---

## Modeller

Aday havuzundaki modellerin kısa açıklamaları:

- **SeasonalTrend** — Mevsimsel medyan tablosu (gün/hafta/ay bazında, yumuşatılmış) ile
  lineer trendin toplamı. Aşırı öğrenmez, geleceğe tutarlı ekstrapole eder; sağlam bir
  taban çizgisi sağlar.
- **Ridge_Fourier** — Standartlaştırma ardından Ridge regresyon. Takvim ve Fourier
  mevsimsellik özellikleri üzerinde L2-düzenlileştirilmiş lineer model.
- **GradientBoosting / RandomForest / XGBoost** — Ağaç tabanlı topluluk modelleri;
  lineer olmayan ilişkileri yakalar. Yalnızca yeterli verinin bulunduğu günlük seviyede
  aday havuzuna dahil edilir.
- **Log-uzay varyantları** — Yukarıdaki regresyon modellerinin, hedef değişkeni logaritma
  dönüşümüyle eğitilen sürümleri. Çarpımsal ve gürültülü serilerde MAPE'yi düşürür.
- **Lag modeli (RidgeCV)** — Lag, hareketli ortalama ve takvim özellikleri üzerinde
  RidgeCV regresyonu; düzenlileştirme katsayısı (alpha) çapraz doğrulamayla seçilir.
  Lineer yapısı sayesinde özyinelemeli tahminde kararlıdır.

---

## Kurulum

Python 3.9 veya üzeri gerekir.

```bash
pip install -r requirements.txt
```

Bağımlılıklar: `pandas`, `numpy`, `scikit-learn`, `joblib`, `xgboost`, `requests`.

---

## Kullanım

### 1. Veri çekme

Kimlik bilgileri tanımlandıktan sonra (bkz. *Veri çekme* bölümü):

```bash
python fetch_real_data.py
```

Bu adım `energy_data_real.csv` dosyasını üretir.

### 2. Model eğitimi

Takvim tabanlı modeller:

```bash
python train_models.py
```

Lag tabanlı modeller:

```bash
python lag_forecast.py --train --csv energy_data_real.csv
```

Eğitilen modeller `models/` klasörüne, metrikler ise `models/model_metrics.csv` ve
`models/lag_model_metrics.csv` dosyalarına kaydedilir.

### 3. Tahmin

**Takvim tabanlı model — belirli bir tarih veya dönem:**

```python
from energy_agent_model import EnergyForecastAgent

agent = EnergyForecastAgent(model_dir="models")

# Tek bir gün
gunluk = agent.predict_daily("2025-03-12")

# Haftalık / aylık dönem
haftalik = agent.predict_period("2025-03-01", "2025-04-30", level="weekly")
aylik = agent.predict_period("2025-03-01", "2025-12-31", level="monthly")
```

Hazır örnek betikler:

```bash
python predict.py      # sabit bir tarih için örnek tahmin
python ask_agent.py    # kullanıcıdan tarih alır
```

**Lag tabanlı model — son veriden sonraki dönemler:**

```python
from lag_forecast import LagForecaster

forecaster = LagForecaster(model_dir="models")

gunluk = forecaster.forecast("daily", n_periods=7)     # sonraki 7 gün
haftalik = forecaster.forecast("weekly", n_periods=4)  # sonraki 4 hafta
aylik = forecaster.forecast("monthly", n_periods=3)    # sonraki 3 ay
```

```bash
python lag_forecast.py    # demo: sonraki dönemleri tahmin eder
```

---

## Sonuçlar

Değerlendirme ölçütleri: **MAPE** (Ortalama Mutlak Yüzde Hata, düşük = iyi) ve **R²**
(belirleme katsayısı, yüksek = iyi). Tüm sonuçlar gerçek EPİAŞ verisiyle, 2024 holdout
yılı üzerinde hesaplanmıştır.

### Takvim tabanlı model (`models/model_metrics.csv`)

| Seviye | Hedef | Seçilen model | MAPE | R² |
|---|---|---|---|---|
| günlük | energy_consumption | XGBoost_log | %5.82 | 0.54 |
| günlük | renewable_share | Ridge_Fourier_log | %14.22 | 0.21 |
| haftalık | energy_consumption | SeasonalTrend | %3.59 | 0.71 |
| haftalık | renewable_share | SeasonalTrend | %12.29 | 0.19 |
| aylık | energy_consumption | SeasonalTrend | %3.41 | 0.80 |
| aylık | renewable_share | SeasonalTrend | %10.20 | 0.32 |

### Lag tabanlı model (`models/lag_model_metrics.csv`)

Lag modelinin hatası tahmin **horizonuna** bağlıdır. *1-adım*, bir dönem öncesinden
yapılan tahmindir (en iyi durum). *Özyinelemeli*, tüm yıl boyunca yalnızca model
tahminleriyle ilerlemedir (en kötü durum). Gerçek kullanım bu ikisinin arasındadır.

| Seviye | Hedef | 1-adım MAPE | Özyinelemeli MAPE |
|---|---|---|---|
| günlük | energy_consumption | %1.72 | %4.54 |
| günlük | renewable_share | %7.62 | %12.82 |
| haftalık | energy_consumption | %4.21 | %5.49 |
| haftalık | renewable_share | %8.91 | %12.13 |
| aylık | energy_consumption | %6.58 | %6.23 |
| aylık | renewable_share | %9.65 | %11.39 |

### Hangi modeli seçmeli?

| Tahmin türü | Önerilen model |
|---|---|
| Günlük tüketim / yenilenebilir pay | Lag modeli |
| Haftalık ve aylık yenilenebilir pay | Lag modeli |
| Haftalık ve aylık tüketim | Takvim modeli (SeasonalTrend) |
| Uzak gelecekteki herhangi bir tarih | Takvim modeli |

Haftalık ve aylık tüketim serileri neredeyse saf mevsimsel olduğundan, bu seviyelerde
takvim modeli (SeasonalTrend) lag modelinden daha isabetlidir.

---

## Sınırlamalar ve Notlar

- **Elektrik tüketimi** güçlü ve düzenli bir mevsimselliğe sahip olduğundan yüksek
  doğrulukla tahmin edilebilir (haftalık/aylık MAPE ~%3.5; günlük 1-adım ~%1.7).
- **Yenilenebilir enerji payının** tahmini doğası gereği daha zordur (MAPE ~%8–14).
  Yenilenebilir üretim büyük ölçüde rüzgâr ve yağış koşullarına bağlıdır; bu koşullar
  takvim veya geçmiş değer özellikleriyle tam olarak öngörülemez.
- Modeller, yalnızca tarih ve geçmiş tüketim/üretim verisini kullanır. Hava durumu
  tahmini gibi dış (egzojen) değişkenler bu sürüme dâhil değildir; bunların eklenmesi,
  özellikle yenilenebilir pay tahmininde, ileride iyileştirme sağlayabilir.
- Holdout değerlendirmesi tek bir yıl (2024) üzerinedir; metrikler bu yılın koşullarını
  yansıtır.
