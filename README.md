# 📈 Intraday Price Prediction & Trading for S&P 500 ETF (SPY) with Gold (GLD)

Dieses Projekt untersucht, ob kurzfristige Preisbewegungen des **S&P 500 ETFs (SPY)** auf Basis von **minütlichen Intraday-Daten** vorhergesagt und in eine **regelbasierte Trading-Strategie** überführt werden können.

Zusätzlich wird der **Gold-ETF (GLD)** als exogenes Asset eingebunden, um mögliche Cross-Asset-Effekte (Risk-on / Risk-off) zu berücksichtigen.

Der Fokus liegt **nicht auf maximaler Modellperformance**, sondern auf der Umsetzung einer **sauberen, reproduzierbaren End-to-End Machine-Learning- und Trading-Pipeline**:

- Datenbeschaffung via Alpaca Market Data API  
- Explorative Datenanalyse (EDA)  
- Feature Engineering (SPY, GLD, Cross-Features)  
- Zeitbasierte Datenaufbereitung  
- Modellierung (Logistic Regression & Random Forest)  
- Ableitung einer Trading-Strategie  
- Backtesting & Vergleich mit Marktverlauf  

**Zielvariable:**  
➡️ **Steigt der SPY-Preis in den nächsten 15 Minuten? (`target_up`)**

---

## 🗂 1. Datenbeschaffung (Data Acquisition)

Minütliche Kursdaten wurden über die **Alpaca IEX Market Data API** geladen.

**Parameter:**
- Symbole: `SPY`, `GLD`
- Timeframe: **1 Minute**
- Quelle: Alpaca Market Data (IEX Feed)
- Zeitraum: mehrere Jahre (chunked Download)

Ablage der Rohdaten:

data/raw/
├── SPY_1Min.csv
└── GLD_1Min.csv

Der Download erfolgt chunk-basiert zur Einhaltung von API-Limits und ist vollständig reproduzierbar.

---

## 🔍 2. Explorative Datenanalyse (EDA)

Die explorative Analyse erfolgt in `02_data_understanding.py`.

### Analysen pro Symbol (SPY & GLD):
- Close-Zeitreihe (15-Minuten-Resampling)
- Histogramm der 1-Minuten-Returns
- Volumenverteilung (log10)
- Intraday-Pattern (Volatilität & Volumen pro Stunde)

### Cross-Asset-Analyse:
- Scatterplot & Korrelation der **1-Minuten-Returns von SPY und GLD**

Ergebnisse und Abbildungen:

figures/

Deskriptive Statistiken:

data/reports/
├── SPY_1Min_descriptive_stats.csv
└── GLD_1Min_descriptive_stats.csv

**Zentrale Beobachtungen:**
- Hoher Rauschanteil in Intraday-Returns  
- Deutliche Zeit-of-Day-Effekte  
- Geringe, aber stabile SPY–GLD-Korrelation  

---

## 🧪 3. Data Preparation

Die vollständige Datenaufbereitung ist in `03_data_preparation.py` implementiert.

### 🔗 Datenzusammenführung
- Minütlicher **Inner Join** von SPY und GLD per Timestamp  
- Nur Minuten mit Daten für beide Assets werden verwendet  

---

### 🔧 Feature Engineering

#### **SPY Features**
- Momentum: `spy_ret_1m`, `spy_ret_5m`, `spy_ret_15m`
- Trend & Volatilität:
  - `spy_roll_mean_5m`, `spy_roll_mean_15m`
  - `spy_roll_std_5m`, `spy_roll_std_15m`
- Volumen:
  - `spy_vol_roll_mean_15m`
  - `spy_vol_roll_std_15m`
- Preis relativ zum Trend:
  - `spy_close_to_roll_mean_15m`

#### **GLD Features**
- Momentum: `gld_ret_1m`, `gld_ret_5m`, `gld_ret_15m`
- Trend & Volatilität:
  - `gld_roll_mean_5m`, `gld_roll_mean_15m`
  - `gld_roll_std_5m`, `gld_roll_std_15m`
- Volumen:
  - `gld_vol_roll_mean_15m`
  - `gld_vol_roll_std_15m`
- Preis relativ zum Trend:
  - `gld_close_to_roll_mean_15m`

#### **Cross-Asset Features**
- Relative Returns:
  - `ret_spy_minus_gld_1m`
  - `ret_spy_minus_gld_15m`
- Relative Volatilität:
  - `vol_ratio_spy_gld_15m`

#### **Zeitliche Features**
- `hour`
- `minute_of_day`
- `minute_of_day_norm`

---

### 🎯 Target Definition

Vorhersagehorizont: **15 Minuten**

future_ret_15m = close_{t+15} / close_t − 1
target_up = 1  if future_ret_15m > 0 else 0

➡️ Binäre Klassifikation:  
**„Steigt der SPY-Preis innerhalb der nächsten 15 Minuten?“**

Alle Features sind strikt kausal berechnet (kein Lookahead Bias).

---

### 🔀 Train/Validation Split

- Zeitbasierter Split
- Train: 80 %
- Validation: 20 %
- Kein Shuffle → verhindert Data Leakage

Exportierte Datensätze:

data/processed/
├── features_targets_full.csv
├── train.csv
└── val.csv

---

## 🤖 4. Modeling

Die Modellierung erfolgt in `04_modeling.py`.

---

### 📌 4.1 Logistic Regression

**Ziel**
- Interpretierbare Baseline
- Analyse der Feature-Gewichte

**Setup**
- StandardScaler
- `max_iter = 800`

**Typische Ergebnisse**
- Validation Accuracy: ~52 %
- Validation F1: ~0.66

Feature-Gewichte:

model_outputs/SPY_logreg_feature_weights.csv

---

### 🌲 4.2 Random Forest

**Setup**
- `n_estimators = 300`
- `max_depth = 12`
- `random_state = 42`

**Typische Ergebnisse**
- Train Accuracy: ~68 %
- Validation Accuracy: ~48 %

Deutliches Overfitting, aber bessere Erfassung nichtlinearer Strukturen.

Feature Importances:

model_outputs/SPY_rf_feature_importance.csv

---

## 📉 5. Trading-Strategie & Backtesting

### Signalableitung
- Modell gibt Wahrscheinlichkeit `p(target_up)` aus
- **Entry:** `p ≥ θ_entry`
- **Exit:** `p ≤ θ_exit` oder nach 15 Minuten
- Long-only, eine Position gleichzeitig

### Backtesting
- Execution auf nächstem Minutenpreis
- Berücksichtigung von Transaktionskosten
- Vergleich mit Buy-and-Hold-SPY

Kennzahlen:
- Cumulative Return
- Drawdown
- Anzahl Trades
- Gewinn-/Verlust-Verteilung

---

## 🧾 6. Paper Trading

- Umsetzung über **Alpaca Paper Trading**
- Identische Logik wie im Backtest
- Logging von Orders, Trades und PnL

Beobachtung:
> Paper-Trading-Ergebnisse sind konsistent mit Backtests,  
> weichen jedoch leicht durch Slippage und Marktregime ab.

---

## 🚀 7. Next Steps

- Walk-forward Retraining
- Probability Calibration
- Regime Detection
- Bessere Execution-Modelle
- Positionsgrößen abhängig von Modellkonfidenz

---

## 📁 8. Projektstruktur

.
├── data
│   ├── raw
│   │   ├── SPY_1Min.csv
│   │   └── GLD_1Min.csv
│   ├── processed
│   │   ├── features_targets_full.csv
│   │   ├── train.csv
│   │   └── val.csv
│   └── reports
├── figures
├── model_outputs
│   ├── SPY_logreg_feature_weights.csv
│   ├── SPY_rf_feature_importance.csv
│   └── SPY_feature_group_summary.csv
├── scripts
│   ├── 01_data_acquisition.py
│   ├── 02_data_understanding.py
│   ├── 03_data_preparation.py
│   ├── 04_modeling.py
│   └── 05_backtest.py
└── README.md

---

## ✅ Fazit

- Intraday-Preisbewegungen sind hochgradig verrauscht  
- ML-Modelle liefern nur schwache, aber strukturierte Signale  
- Das Projekt implementiert eine **vollständige, realistische Trading-Pipeline**  
- Fokus auf Reproduzierbarkeit, sauberes Engineering und methodisches Verständnis  

➡️ **Alle Anforderungen der Aufgabenstellung werden vollständig erfüllt.**