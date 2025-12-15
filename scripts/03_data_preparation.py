# scripts/03_data_preparation.py
"""
Step 03 — Data Preparation (SPY 1Min + GLD 1Min)
------------------------------------------------
Ziele:
- SPY 1-Minuten-Rohdaten + GLD 1-Minuten-Rohdaten laden
- Zusammenführen per Timestamp (inner join)
- Features erstellen (SPY + GLD + Cross)
- Target definieren (auf SPY):
  Binär: Steigt SPY in den nächsten 15 Minuten? (1/0)
- Zeitbasierten Train/Validation-Split
- Speichern in data/processed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class ProjectConfig:
    primary_symbol: str = "SPY"
    gold_symbol: str = "GLD"

    primary_interval: str = "1Min"
    gold_interval: str = "1Min"

    prediction_horizon_min: int = 15
    train_ratio: float = 0.8

    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def raw_data_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    def raw_csv_path(self, symbol: str, interval: str) -> Path:
        return self.raw_data_dir / f"{symbol}_{interval}.csv"


class DataPreparation:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.df: pd.DataFrame | None = None

    # ---------- loading ----------
    def load_symbol(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self.cfg.raw_csv_path(symbol, interval)
        if not path.exists():
            raise FileNotFoundError(
                f"Rohdaten nicht gefunden: {path}. Bitte zuerst scripts/01_data_acquisition.py ausführen."
            )

        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{symbol}: Spalte 'timestamp' fehlt.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        if interval != "1Min":
            raise ValueError(f"{symbol}: Expected 1Min, got {interval} (Script 03 ist für 1Min+1Min).")

        return df

    # ---------- merge ----------
    def merge_spy_gold(self) -> pd.DataFrame:
        spy = self.load_symbol(self.cfg.primary_symbol, self.cfg.primary_interval).add_prefix("spy_")
        gld = self.load_symbol(self.cfg.gold_symbol, self.cfg.gold_interval).add_prefix("gld_")

        # Inner join: nur Minuten, in denen beide Symbole Bars haben
        df = spy.join(gld, how="inner").sort_index()

        # Doppelte Timestamps entfernen (safety)
        df = df[~df.index.duplicated(keep="last")]

        self.df = df
        print(f"✅ Merged DF: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")
        print(f"   Zeitraum: {df.index.min()} → {df.index.max()}")
        return df

    # ---------- features ----------
    def engineer_features(self) -> pd.DataFrame:
        assert self.df is not None
        df = self.df.copy()

        # --- SPY Features ---
        df["spy_ret_1m"] = df["spy_close"].pct_change(1)
        df["spy_ret_5m"] = df["spy_close"].pct_change(5)
        df["spy_ret_15m"] = df["spy_close"].pct_change(15)

        df["spy_roll_mean_5m"] = df["spy_close"].rolling(5).mean()
        df["spy_roll_mean_15m"] = df["spy_close"].rolling(15).mean()
        df["spy_roll_std_5m"] = df["spy_close"].rolling(5).std()
        df["spy_roll_std_15m"] = df["spy_close"].rolling(15).std()

        df["spy_vol_roll_mean_15m"] = df["spy_volume"].rolling(15).mean()
        df["spy_vol_roll_std_15m"] = df["spy_volume"].rolling(15).std()

        # Momentum relativ zur MA
        df["spy_close_to_roll_mean_15m"] = df["spy_close"] / df["spy_roll_mean_15m"] - 1

        # --- GLD Features (intraday, analog) ---
        df["gld_ret_1m"] = df["gld_close"].pct_change(1)
        df["gld_ret_5m"] = df["gld_close"].pct_change(5)
        df["gld_ret_15m"] = df["gld_close"].pct_change(15)

        df["gld_roll_mean_5m"] = df["gld_close"].rolling(5).mean()
        df["gld_roll_mean_15m"] = df["gld_close"].rolling(15).mean()
        df["gld_roll_std_5m"] = df["gld_close"].rolling(5).std()
        df["gld_roll_std_15m"] = df["gld_close"].rolling(15).std()

        df["gld_vol_roll_mean_15m"] = df["gld_volume"].rolling(15).mean()
        df["gld_vol_roll_std_15m"] = df["gld_volume"].rolling(15).std()

        df["gld_close_to_roll_mean_15m"] = df["gld_close"] / df["gld_roll_mean_15m"] - 1

        # --- Cross Features ---
        df["ret_spy_minus_gld_1m"] = df["spy_ret_1m"] - df["gld_ret_1m"]
        df["ret_spy_minus_gld_15m"] = df["spy_ret_15m"] - df["gld_ret_15m"]

        df["vol_ratio_spy_gld_15m"] = df["spy_roll_std_15m"] / (df["gld_roll_std_15m"] + 1e-12)

        # Optional: Spread/Ratio (manchmal stabiler als Differenz)
        df["price_ratio_spy_gld"] = df["spy_close"] / (df["gld_close"] + 1e-12)

        # --- Intraday Position ---
        df["hour"] = df.index.hour
        df["minute_of_day"] = df["hour"] * 60 + df.index.minute
        df["minute_of_day_norm"] = df["minute_of_day"] / (24 * 60)

        self.df = df
        print(f"✅ Features erstellt. Spalten: {df.shape[1]}")
        return df

    # ---------- target ----------
    def engineer_target(self) -> pd.DataFrame:
        assert self.df is not None
        df = self.df.copy()

        h = self.cfg.prediction_horizon_min
        col = f"future_ret_{h}m"

        df[col] = df["spy_close"].shift(-h) / df["spy_close"] - 1
        df["target_up"] = (df[col] > 0).astype(int)

        self.df = df
        print(f"✅ Target erstellt: {col} + target_up (auf SPY)")
        return df

    # ---------- cleaning ----------
    def clean_drop_na(self) -> pd.DataFrame:
        assert self.df is not None
        df = self.df.copy()
        before = df.shape[0]
        df = df.dropna()
        after = df.shape[0]
        print(f"🧹 NaN-Bereinigung: {before} → {after} (entfernt {before - after})")
        self.df = df
        return df

    # ---------- split ----------
    def train_val_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert self.df is not None
        df = self.df.copy()
        n = len(df)
        split = int(n * self.cfg.train_ratio)
        train_df = df.iloc[:split].copy()
        val_df = df.iloc[split:].copy()
        print(f"🔀 Split: Train={len(train_df)}, Val={len(val_df)}")
        return train_df, val_df

    # ---------- save ----------
    def save(self, full_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)

        full_path = self.cfg.processed_data_dir / "features_targets_full.csv"
        train_path = self.cfg.processed_data_dir / "train.csv"
        val_path = self.cfg.processed_data_dir / "val.csv"

        full_df.to_csv(full_path)
        train_df.to_csv(train_path)
        val_df.to_csv(val_path)

        print(f"💾 Gespeichert: {full_path}")
        print(f"💾 Gespeichert: {train_path}")
        print(f"💾 Gespeichert: {val_path}")

    # ---------- run ----------
    def run(self) -> None:
        self.merge_spy_gold()
        self.engineer_features()
        self.engineer_target()
        df_clean = self.clean_drop_na()
        train_df, val_df = self.train_val_split()
        self.save(df_clean, train_df, val_df)


def main() -> None:
    cfg = ProjectConfig()
    DataPreparation(cfg).run()


if __name__ == "__main__":
    main()