# scripts/02_data_understanding.py
"""
Step 02 — Data Understanding (SPY 1Min + GLD 1Min)
--------------------------------------------------
Ziele:
- Rohdaten für SPY (1Min) und GLD (1Min) laden
- Deskriptive Statistiken speichern
- Plots je Symbol (intraday):
  - 15-Min Close Zeitreihe
  - Return-Histogramm (1Min)
  - Volume-Histogramm (log10)
  - Intraday-Pattern pro Stunde
- Zusätzlich:
  - Korrelation der 1-Min Returns (SPY vs GLD)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1) Projekt-Konfiguration
# ---------------------------------------------------------
@dataclass
class ProjectConfig:
    primary_symbol: str = "SPY"
    gold_symbol: str = "GLD"

    primary_interval: str = "1Min"
    gold_interval: str = "1Min"

    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def symbols(self) -> list[tuple[str, str]]:
        return [
            (self.primary_symbol, self.primary_interval),
            (self.gold_symbol, self.gold_interval),
        ]

    @property
    def raw_data_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def reports_dir(self) -> Path:
        return self.base_dir / "data" / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    def raw_csv_path(self, symbol: str, interval: str) -> Path:
        return self.raw_data_dir / f"{symbol}_{interval}.csv"


# ---------------------------------------------------------
# 2) Data Understanding Pipeline
# ---------------------------------------------------------
class DataUnderstanding:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.data: dict[str, pd.DataFrame] = {}

    # ---------- helpers ----------
    def _ensure_dirs(self) -> None:
        self.cfg.figures_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    # ---------- loading ----------
    def load_symbol(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self.cfg.raw_csv_path(symbol, interval)
        if not path.exists():
            raise FileNotFoundError(
                f"Rohdaten nicht gefunden: {path}. "
                f"Bitte zuerst scripts/01_data_acquisition.py ausführen."
            )

        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{symbol}: Spalte 'timestamp' fehlt.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        if interval != "1Min":
            raise ValueError(
                f"{symbol}: Expected 1Min data, got {interval}. "
                f"Dieses Skript ist ausschließlich für 1Min + 1Min gedacht."
            )

        # Basis-Features
        df["return_1min"] = df["close"].pct_change()
        df["abs_return_1min"] = df["return_1min"].abs()
        df["hour"] = df.index.hour

        key = f"{symbol}_{interval}"
        self.data[key] = df

        print(
            f"✅ {symbol} ({interval}) geladen: "
            f"{df.shape[0]} Zeilen, {df.shape[1]} Spalten | "
            f"{df.index.min()} → {df.index.max()}"
        )
        return df

    # ---------- stats ----------
    def save_descriptive_stats(self, symbol: str, interval: str) -> None:
        key = f"{symbol}_{interval}"
        df = self.data[key]

        self._ensure_dirs()

        numeric_cols = [
            "open", "high", "low", "close",
            "volume", "trade_count", "vwap",
            "return_1min"
        ]
        existing = [c for c in numeric_cols if c in df.columns]

        desc = df[existing].describe().T
        out = self.cfg.reports_dir / f"{symbol}_{interval}_descriptive_stats.csv"
        desc.to_csv(out)

        print(f"💾 Stats gespeichert: {out}")

    # ---------- intraday plots ----------
    def plot_intraday_bundle(self, symbol: str, interval: str) -> None:
        key = f"{symbol}_{interval}"
        df = self.data[key]
        self._ensure_dirs()

        # --- 15-Min Close ---
        series = df["close"].resample("15Min").mean()
        plt.figure(figsize=(12, 5))
        plt.plot(series.index, series.values)
        plt.title(f"{symbol} — 15-Min Close (resampled from 1Min)")
        plt.xlabel("Zeit")
        plt.ylabel("Close (USD)")
        plt.tight_layout()
        plt.savefig(self.cfg.figures_dir / f"{symbol}_01_close_timeseries_15min.png", dpi=150)
        plt.close()

        # --- Return Histogram ---
        rets = df["return_1min"].dropna()
        rets = rets.clip(rets.quantile(0.01), rets.quantile(0.99))
        plt.figure(figsize=(8, 5))
        plt.hist(rets, bins=50)
        plt.title(f"{symbol} — Histogramm der 1-Min Returns")
        plt.xlabel("Return (1Min)")
        plt.ylabel("Häufigkeit")
        plt.tight_layout()
        plt.savefig(self.cfg.figures_dir / f"{symbol}_02_return_histogram_1min.png", dpi=150)
        plt.close()

        # --- Volume Histogram ---
        vol = df["volume"].replace(0, np.nan).dropna()
        log_vol = np.log10(vol)
        plt.figure(figsize=(8, 5))
        plt.hist(log_vol, bins=50)
        plt.title(f"{symbol} — Histogramm Volume (log10)")
        plt.xlabel("log10(Volume)")
        plt.ylabel("Häufigkeit")
        plt.tight_layout()
        plt.savefig(self.cfg.figures_dir / f"{symbol}_03_volume_histogram_log.png", dpi=150)
        plt.close()

        # --- Intraday Pattern ---
        grouped = df.groupby("hour").agg(
            mean_abs_return=("abs_return_1min", "mean"),
            mean_volume=("volume", "mean"),
        )

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(grouped.index, grouped["mean_abs_return"], marker="o")
        ax1.set_xlabel("Stunde (UTC)")
        ax1.set_ylabel("Ø |Return| (1Min)")
        ax1.grid(True, axis="y")

        ax2 = ax1.twinx()
        ax2.plot(grouped.index, grouped["mean_volume"], marker="s")
        ax2.set_ylabel("Ø Volume")

        plt.title(f"{symbol} — Intraday Pattern (Volatilität & Volume)")
        fig.tight_layout()
        plt.savefig(self.cfg.figures_dir / f"{symbol}_04_intraday_pattern_hourly.png", dpi=150)
        plt.close()

    # ---------- SPY vs GLD correlation ----------
    def plot_spy_gld_return_correlation_1min(self) -> None:
        self._ensure_dirs()

        spy_key = f"{self.cfg.primary_symbol}_{self.cfg.primary_interval}"
        gld_key = f"{self.cfg.gold_symbol}_{self.cfg.gold_interval}"

        spy = self.data[spy_key][["return_1min"]].rename(columns={"return_1min": "SPY_ret"})
        gld = self.data[gld_key][["return_1min"]].rename(columns={"return_1min": "GLD_ret"})

        merged = spy.join(gld, how="inner").dropna()
        corr = float(merged.corr().iloc[0, 1]) if not merged.empty else np.nan

        plt.figure(figsize=(6, 4))
        plt.scatter(merged["GLD_ret"], merged["SPY_ret"], s=2)
        plt.title(f"SPY vs GLD — 1-Min Return Scatter (corr={corr:.4f})")
        plt.xlabel("GLD return_1min")
        plt.ylabel("SPY return_1min")
        plt.tight_layout()
        plt.savefig(self.cfg.figures_dir / "SPY_GLD_return_scatter_1min.png", dpi=150)
        plt.close()

        out = self.cfg.reports_dir / "SPY_GLD_return_correlation_1min.txt"
        out.write_text(
            f"Correlation (SPY 1-min returns vs GLD 1-min returns): {corr:.6f}\n",
            encoding="utf-8",
        )

        print(f"📊 SPY–GLD 1Min Korrelation: {corr:.4f}")

    # ---------- run ----------
    def run(self) -> None:
        for symbol, interval in self.cfg.symbols:
            self.load_symbol(symbol, interval)
            self.save_descriptive_stats(symbol, interval)
            self.plot_intraday_bundle(symbol, interval)

        self.plot_spy_gld_return_correlation_1min()


# ---------------------------------------------------------
# 3) Main
# ---------------------------------------------------------
def main() -> None:
    cfg = ProjectConfig()
    DataUnderstanding(cfg).run()


if __name__ == "__main__":
    main()