# scripts/02_data_understanding.py
"""
Step 02 — Data Understanding (Intraday, Alpaca-Minutenbars)
-----------------------------------------------------------
Ziele:
- Minutendaten aus data/raw/URTH_1Min.csv laden
- Grundlegende Statistiken berechnen und als CSV speichern
- Wichtige Plots erstellen:
  - Zeitreihe der (aggregierten) Close-Preise
  - Histogramm der 1-Minuten-Returns
  - Histogramm des Volumens
  - Intraday-Pattern (durchschnittliche Volatilität & Volumen pro Stunde)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1) Projekt-Konfiguration
# ---------------------------------------------------------
@dataclass
class ProjectConfig:
    symbol: str = "URTH"
    interval: str = "1Min"

    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @property
    def raw_csv_path(self) -> Path:
        filename = f"{self.symbol}_{self.interval}.csv"
        return self.raw_data_dir / filename


# ---------------------------------------------------------
# 2) Data Understanding
# ---------------------------------------------------------
class MSCIWorldDataUnderstanding:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.df: pd.DataFrame | None = None

    # ---------- Load & basic processing ----------

    def load_raw_data(self) -> pd.DataFrame:
        """
        Lädt die Rohdaten aus data/raw/URTH_1Min.csv
        und setzt den Timestamp als DatetimeIndex.
        """
        if not self.cfg.raw_csv_path.exists():
            raise FileNotFoundError(
                f"Rohdaten nicht gefunden: {self.cfg.raw_csv_path}. "
                f"Führe zuerst 01_data_acquisition.py aus."
            )

        df = pd.read_csv(self.cfg.raw_csv_path)

        # Erwartete Spalten von Alpaca: timestamp, open, high, low, close, volume, trade_count, vwap
        if "timestamp" not in df.columns:
            raise ValueError("Spalte 'timestamp' fehlt in der CSV.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Zusätzliche Felder für spätere Analysen
        df["return_1min"] = df["close"].pct_change()
        df["abs_return_1min"] = df["return_1min"].abs()
        df["hour"] = df.index.hour
        df["minute_of_day"] = df["hour"] * 60 + df.index.minute

        self.df = df
        print(f"✅ Rohdaten geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")
        print(f"   Zeitraum: {df.index.min()}  →  {df.index.max()}")
        return df

    # ---------- Reports ----------

    def save_descriptive_stats(self) -> None:
        """
        Speichert deskriptive Statistiken der wichtigsten numerischen Spalten.
        """
        assert self.df is not None, "DataFrame ist leer. load_raw_data() zuerst aufrufen."

        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap", "return_1min"]
        existing_cols = [c for c in numeric_cols if c in self.df.columns]

        desc = self.df[existing_cols].describe().T
        output_path = self.cfg.reports_dir / "intraday_descriptive_stats.csv"
        desc.to_csv(output_path)

        print(f"💾 Deskriptive Statistiken gespeichert unter: {output_path}")

    # ---------- Plot helpers ----------

    def _ensure_figures_dir(self) -> None:
        self.cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_close_timeseries(self) -> None:
        """
        Zeitreihe der Close-Preise. Zur besseren Lesbarkeit wird auf 15-Minuten
        aggregiert (Mean).
        """
        assert self.df is not None
        self._ensure_figures_dir()

        # 15-Minuten-Resampling
        df_resampled = self.df["close"].resample("15Min").mean()

        plt.figure(figsize=(12, 5))
        plt.plot(df_resampled.index, df_resampled.values)
        plt.title("URTH — 15-Min Close Price (Alpaca 1Min resampled)")
        plt.xlabel("Zeit")
        plt.ylabel("Close-Preis (USD)")
        plt.tight_layout()

        path = self.cfg.figures_dir / "01_close_timeseries_15min.png"
        plt.savefig(path, dpi=150)
        plt.close()

        print(f"📊 Plot gespeichert: {path}")

    def plot_return_histogram(self) -> None:
        """
        Histogramm der 1-Minuten-Returns.
        """
        assert self.df is not None
        self._ensure_figures_dir()

        returns = self.df["return_1min"].dropna()
        # Ausreißer kappen, damit das Histogramm sinnvoll aussieht
        returns = returns.clip(lower=returns.quantile(0.01), upper=returns.quantile(0.99))

        plt.figure(figsize=(8, 5))
        plt.hist(returns, bins=50)
        plt.title("URTH — Histogramm der 1-Minuten-Returns")
        plt.xlabel("Return (close_t / close_{t-1} - 1)")
        plt.ylabel("Häufigkeit")
        plt.tight_layout()

        path = self.cfg.figures_dir / "02_return_histogram_1min.png"
        plt.savefig(path, dpi=150)
        plt.close()

        print(f"📊 Plot gespeichert: {path}")

    def plot_volume_histogram(self) -> None:
        """
        Histogramm des Volumens (log-skaliert).
        """
        assert self.df is not None
        self._ensure_figures_dir()

        volume = self.df["volume"].replace(0, np.nan).dropna()
        log_vol = np.log10(volume)

        plt.figure(figsize=(8, 5))
        plt.hist(log_vol, bins=50)
        plt.title("URTH — Histogramm des Volumens (log10)")
        plt.xlabel("log10(Volume)")
        plt.ylabel("Häufigkeit")
        plt.tight_layout()

        path = self.cfg.figures_dir / "03_volume_histogram_log.png"
        plt.savefig(path, dpi=150)
        plt.close()

        print(f"📊 Plot gespeichert: {path}")

    def plot_intraday_patterns(self) -> None:
        """
        Intraday-Pattern:
        - durchschnittliche absolute 1-Minuten-Returns pro Stunde
        - durchschnittliches Volumen pro Stunde
        """
        assert self.df is not None
        self._ensure_figures_dir()

        grouped = self.df.groupby("hour").agg(
            mean_abs_return=("abs_return_1min", "mean"),
            mean_volume=("volume", "mean"),
        )

        fig, ax1 = plt.subplots(figsize=(10, 5))

        hours = grouped.index

        ax1.plot(hours, grouped["mean_abs_return"], marker="o", label="Ø |Return| (1Min)")
        ax1.set_xlabel("Stunde des Tages (UTC)")
        ax1.set_ylabel("Ø absolute 1-Minuten-Returns")
        ax1.grid(True, axis="y")

        ax2 = ax1.twinx()
        ax2.plot(hours, grouped["mean_volume"], marker="s", color="orange", label="Ø Volume")
        ax2.set_ylabel("Ø Volume")

        plt.title("URTH — Intraday-Pattern: Volatilität & Volumen pro Stunde")
        fig.tight_layout()

        path = self.cfg.figures_dir / "04_intraday_pattern_hourly.png"
        plt.savefig(path, dpi=150)
        plt.close()

        print(f"📊 Plot gespeichert: {path}")

    # ---------- Orchestrierung ----------

    def run(self) -> None:
        self.load_raw_data()
        self.save_descriptive_stats()
        self.plot_close_timeseries()
        self.plot_return_histogram()
        self.plot_volume_histogram()
        self.plot_intraday_patterns()


# ---------------------------------------------------------
# 3) Skript-Einstiegspunkt
# ---------------------------------------------------------
def main() -> None:
    cfg = ProjectConfig()
    understanding = MSCIWorldDataUnderstanding(cfg)
    understanding.run()


if __name__ == "__main__":
    main()
