# scripts/03_data_preparation.py
"""
Step 03 — Data Preparation (pre-split)
--------------------------------------------------------
Ziel:
- Aus Intraday-Daten (1h) Tagesdaten aggregieren
- Features und Zielvariable (Target) konstruieren
- Deskriptive Statistiken und Plots für Features/Target erzeugen

Target-Definition:
- Für jeden Handelstag d:
    target(d) = 1, wenn Close_d > Open_d
                0, sonst

Features (Beispiele):
- intraday_return = (Close_d - Open_d) / Open_d
- intraday_range  = (High_d - Low_d) / Open_d
- daily_return    = Close_d / Close_{d-1} - 1
- rolling_mean_k  = gleitender Mittelwert von daily_return über k Tage
- rolling_vol_k   = gleitende Std-Abweichung von daily_return über k Tage

Input:
- CSV: data/raw/URTH_1h.csv

Output:
- CSV: data/processed/URTH_daily_features.csv
- Plots:
  - figures/URTH_daily_target_distribution.png
  - figures/URTH_daily_feature_correlations.png
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1) Projekt-Konfiguration
# ---------------------------------------------------------
@dataclass
class ProjectConfig:
    ticker: str = "URTH"
    interval: str = "1h"
    base_dir: Path = Path(__file__).resolve().parents[1]

    # Hyperparameter für Feature Engineering
    rolling_window: int = 5  # z. B. 5 Handelstage

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / f"{self.ticker}_{self.interval}.csv"

    @property
    def processed_csv_path(self) -> Path:
        return self.processed_data_dir / f"{self.ticker}_daily_features.csv"


# ---------------------------------------------------------
# 2) Data Preparation
# ---------------------------------------------------------
class DataPreparation:
    """
    Bereitet Daten für das Modell vor (pre-split):
    - Aggregation von 1h -> 1d
    - Feature Engineering
    - Target-Berechnung
    - Statistiken & Plots
    """

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg

    # ---------- Helper ----------

    def _ensure_dirs(self) -> None:
        self.cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 0) Laden der Intraday-Daten ----------

    def load_intraday_data(self) -> pd.DataFrame:
        print(f"📂 Lade Intraday-Daten aus: {self.cfg.raw_csv_path}")
        df = pd.read_csv(self.cfg.raw_csv_path)

        # Datetime-Spalte finden und sauber parsen
        dt_col = "Datetime" if "Datetime" in df.columns else "Date"
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        df[dt_col] = df[dt_col].dt.tz_localize(None)

        # numerische Spalten in float/int casten
        num_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # sortieren, NaT raus
        df = df.sort_values(dt_col)
        df = df[df[dt_col].notna()].set_index(dt_col)

        print(f"✅ Intraday-Daten geladen, Shape: {df.shape}")
        return df

    # ---------- 1) Aggregation 1h -> 1d ----------

    def aggregate_to_daily(self, intraday: pd.DataFrame) -> pd.DataFrame:
        print("\n🔄 Aggregiere Intraday-Daten zu Tagesdaten...")

        daily = pd.DataFrame()
        daily["Open"] = intraday["Open"].resample("1D").first()
        daily["Close"] = intraday["Close"].resample("1D").last()
        daily["High"] = intraday["High"].resample("1D").max()
        daily["Low"] = intraday["Low"].resample("1D").min()
        daily["Volume"] = intraday["Volume"].resample("1D").sum()

        # Tage ohne Handel entfernen
        daily = daily.dropna().reset_index()
        daily = daily.rename(columns={"Datetime": "Date"})

        print(f"✅ Tagesdaten erstellt, Shape: {daily.shape}")
        return daily

    # ---------- 2) Feature Engineering ----------

    def engineer_features(self, daily: pd.DataFrame) -> pd.DataFrame:
        print("\n🛠 Erzeuge Features...")

        df = daily.copy()

        # Intraday-Maße
        df["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]
        df["intraday_range"] = (df["High"] - df["Low"]) / df["Open"]

        # Tagesrendite auf Basis des Schlusskurses
        df["daily_return"] = df["Close"].pct_change()

        # Rolling-Features
        window = self.cfg.rolling_window
        df[f"rolling_mean_{window}"] = df["daily_return"].rolling(window).mean()
        df[f"rolling_vol_{window}"] = df["daily_return"].rolling(window).std()

        print(f"✅ Features erstellt mit rolling_window={window}")
        return df

    # ---------- 3) Target-Berechnung ----------

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n🎯 Berechne Target-Variable (Close > Open)...")

        df["target"] = (df["Close"] > df["Open"]).astype(int)

        print("🔎 Target-Verteilung (value_counts):")
        print(df["target"].value_counts())

        return df

    # ---------- 4) Plots ----------

    def plot_target_distribution(self, df: pd.DataFrame) -> None:
        self._ensure_dirs()

        plt.figure()
        df["target"].value_counts().sort_index().plot(kind="bar")
        plt.xticks([0, 1], ["0 (Close <= Open)", "1 (Close > Open)"], rotation=0)
        plt.ylabel("Anzahl Tage")
        plt.title(f"{self.cfg.ticker} — Target-Verteilung (Close > Open)")
        plt.tight_layout()

        path = self.cfg.figures_dir / f"{self.cfg.ticker}_daily_target_distribution.png"
        plt.savefig(path)
        plt.close()
        print(f"✅ Target-Verteilungsplot gespeichert: {path}")

    def plot_feature_correlations(self, df: pd.DataFrame) -> None:
        self._ensure_dirs()

        feature_cols = [
            "intraday_return",
            "intraday_range",
            "daily_return",
            f"rolling_mean_{self.cfg.rolling_window}",
            f"rolling_vol_{self.cfg.rolling_window}",
            "target",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

        corr = df[feature_cols].corr()

        plt.figure(figsize=(6, 5))
        plt.imshow(corr, interpolation="nearest")
        plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
        plt.yticks(range(len(feature_cols)), feature_cols)
        plt.colorbar(label="Korrelationskoeffizient")
        plt.title(f"{self.cfg.ticker} — Korrelationsmatrix (Features & Target)")
        plt.tight_layout()

        path = self.cfg.figures_dir / f"{self.cfg.ticker}_daily_feature_correlations.png"
        plt.savefig(path)
        plt.close()
        print(f"✅ Korrelationsplot gespeichert: {path}")

    # ---------- 5) Findings ----------

    def print_findings(self, df: pd.DataFrame) -> None:
        print("\n📝 Findings zur Datenvorbereitung:")

        print(f"- Anzahl Handelstage nach Aggregation: {len(df)}")

        pos_share = df["target"].mean()
        print(
            f"- Anteil Tage mit Close > Open (target=1): {pos_share:.2%} "
            "(Klassenverteilung)."
        )

        print(
            f"- 'intraday_return' fasst die Netto-Bewegung eines Tages "
            f"zusammen, 'intraday_range' die Schwankungsbreite."
        )

        print(
            f"- Rolling-Features über {self.cfg.rolling_window} Tage "
            f"modellieren kurzfristige Trends (rolling_mean) und "
            f"Volatilität (rolling_vol)."
        )

    # ---------- Orchestrierung ----------

    def run(self) -> pd.DataFrame:
        self._ensure_dirs()

        intraday = self.load_intraday_data()
        daily = self.aggregate_to_daily(intraday)
        feat_df = self.engineer_features(daily)
        feat_df = self.create_target(feat_df)

        # ein paar Zeilen zeigen
        print("\n🔎 Beispielhafte Zeilen der vorbereiteten Daten:")
        print(feat_df.head())

        # Speichern
        feat_df.to_csv(self.cfg.processed_csv_path, index=False)
        print(f"\n✅ Vorbereitete Daten gespeichert unter: {self.cfg.processed_csv_path}")

        # Plots & Findings
        self.plot_target_distribution(feat_df)
        self.plot_feature_correlations(feat_df)
        self.print_findings(feat_df)

        return feat_df


def main() -> None:
    cfg = ProjectConfig()
    dp = DataPreparation(cfg)
    dp.run()


if __name__ == "__main__":
    main()
