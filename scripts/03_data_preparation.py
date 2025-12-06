# scripts/03_data_preparation.py
"""
Step 03 — Data Preparation (Intraday, post-split)
-------------------------------------------------
Ziele:
- 1-Minuten-Rohdaten aus data/raw/URTH_1Min.csv laden
- Feature-Engineering:
  - Returns über verschiedene Horizonte
  - Rolling Means & Volatilität
  - Intraday-Position (Stunde, Minute des Tages)
- Target definieren:
  - Binäre Klassifikation: Steigt der Preis in den nächsten 15 Minuten? (1 = ja, 0 = nein)
- Zeitbasierten Train/Validation-Split durchführen
- Ergebnis als CSVs in data/processed speichern
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1) Projekt-Konfiguration
# ---------------------------------------------------------
@dataclass
class ProjectConfig:
    symbol: str = "URTH"
    interval: str = "1Min"
    prediction_horizon_min: int = 15  # wie viele Minuten in die Zukunft wir vorhersagen

    base_dir: Path = Path(__file__).resolve().parents[1]

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
    def raw_csv_path(self) -> Path:
        filename = f"{self.symbol}_{self.interval}.csv"
        return self.raw_data_dir / filename


# ---------------------------------------------------------
# 2) Data Preparation
# ---------------------------------------------------------
class MSCIWorldDataPreparation:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.df: pd.DataFrame | None = None

    # ---------- Load ----------

    def load_raw_data(self) -> pd.DataFrame:
        """
        Lädt die 1-Minuten-Rohdaten und setzt timestamp als Index.
        """
        if not self.cfg.raw_csv_path.exists():
            raise FileNotFoundError(
                f"Rohdaten nicht gefunden: {self.cfg.raw_csv_path}. "
                f"Führe zuerst 01_data_acquisition.py aus."
            )

        df = pd.read_csv(self.cfg.raw_csv_path)
        if "timestamp" not in df.columns:
            raise ValueError("Spalte 'timestamp' fehlt in der CSV.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        self.df = df
        print(f"✅ Rohdaten geladen für Data Preparation: {df.shape[0]} Zeilen.")
        return df

    # ---------- Feature Engineering ----------

    def engineer_features(self) -> pd.DataFrame:
        """
        Erstellt Features basierend auf 1-Minuten-Daten.
        """
        assert self.df is not None, "DataFrame ist leer. load_raw_data() zuerst aufrufen."
        df = self.df.copy()

        # 1) Returns (Vergangenheit)
        df["ret_1m"] = df["close"].pct_change(1)
        df["ret_5m"] = df["close"].pct_change(5)
        df["ret_15m"] = df["close"].pct_change(15)

        # 2) Rolling Means und Volatilität (Std)
        df["roll_mean_5m"] = df["close"].rolling(5).mean()
        df["roll_mean_15m"] = df["close"].rolling(15).mean()

        df["roll_std_5m"] = df["close"].rolling(5).std()
        df["roll_std_15m"] = df["close"].rolling(15).std()

        # 3) Volumen-Features
        df["vol_roll_mean_15m"] = df["volume"].rolling(15).mean()
        df["vol_roll_std_15m"] = df["volume"].rolling(15).std()

        # 4) Intraday-Position
        df["hour"] = df.index.hour
        df["minute_of_day"] = df["hour"] * 60 + df.index.minute
        df["minute_of_day_norm"] = df["minute_of_day"] / (24 * 60)

        # 5) Optional: Verhältnis aktueller Close zum 15-Minuten-MA (Momentum / Trend)
        df["close_to_roll_mean_15m"] = df["close"] / df["roll_mean_15m"] - 1

        self.df = df
        print(f"✅ Features erstellt. Aktuelle Spaltenanzahl: {df.shape[1]}")
        return df

    # ---------- Target Engineering ----------

    def engineer_target(self) -> pd.DataFrame:
        """
        Definiert das Vorhersageziel:
        - future_return_{horizon}min = (close_{t+h} / close_t - 1)
        - target_up = 1, wenn future_return > 0, sonst 0
        """
        assert self.df is not None
        df = self.df.copy()

        horizon = self.cfg.prediction_horizon_min
        col_name = f"future_ret_{horizon}m"

        df[col_name] = df["close"].shift(-horizon) / df["close"] - 1
        df["target_up"] = (df[col_name] > 0).astype(int)

        self.df = df
        print(f"✅ Target erstellt: {col_name} & target_up")
        return df

    # ---------- Cleaning & NaN Handling ----------

    def clean_and_drop_na(self) -> pd.DataFrame:
        """
        Entfernt Zeilen mit NaNs, die durch Rolling-Fenster und Shifts
        entstehen (Anfang & Ende der Zeitreihe).
        """
        assert self.df is not None
        df = self.df.copy()

        before = df.shape[0]
        df = df.dropna()
        after = df.shape[0]

        print(f"🧹 NaN-Bereinigung: {before} → {after} Zeilen (entfernt: {before - after})")

        self.df = df
        return df

    # ---------- Train/Validation-Split ----------

    def train_validation_split(self, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Zeitbasierter Train/Validation-Split.
        Nutzt die chronologische Reihenfolge — keine zufällige Durchmischung.
        """
        assert self.df is not None
        df = self.df.copy()

        n = df.shape[0]
        split_idx = int(n * train_ratio)

        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        print(f"🔀 Train/Val-Split bei Index {split_idx}: Train={train_df.shape[0]}, Val={val_df.shape[0]}")
        return train_df, val_df

    # ---------- Save ----------

    def save_processed(self, full_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """
        Speichert vollständigen Feature+Target-DataFrame und die Split-Subsets.
        """
        self.cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)

        full_path = self.cfg.processed_data_dir / "features_targets_full.csv"
        train_path = self.cfg.processed_data_dir / "train.csv"
        val_path = self.cfg.processed_data_dir / "val.csv"

        full_df.to_csv(full_path)
        train_df.to_csv(train_path)
        val_df.to_csv(val_path)

        print(f"💾 Vollständiger Datensatz gespeichert unter: {full_path}")
        print(f"💾 Train-Set gespeichert unter: {train_path}")
        print(f"💾 Validation-Set gespeichert unter: {val_path}")

    # ---------- Orchestrierung ----------

    def run(self) -> None:
        self.load_raw_data()
        self.engineer_features()
        self.engineer_target()
        df_clean = self.clean_and_drop_na()
        train_df, val_df = self.train_validation_split(train_ratio=0.8)
        self.save_processed(df_clean, train_df, val_df)


# ---------------------------------------------------------
# 3) Skript-Einstiegspunkt
# ---------------------------------------------------------
def main() -> None:
    cfg = ProjectConfig()
    prep = MSCIWorldDataPreparation(cfg)
    prep.run()


if __name__ == "__main__":
    main()
