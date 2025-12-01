# scripts/02_data_understanding.py
"""
Step 02 — Data Understanding
--------------------------------------------------------
Ziel:
- Rohdaten (Intraday, z. B. 1h) einlesen
- Relevante Spalten erklären
- Deskriptive Statistiken berechnen
- Relevante Plots erstellen
- Erste Findings ausgeben

Input:
- CSV: data/raw/URTH_1h.csv (aus 01_data_acquisition.py)
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
        return self.raw_data_dir / f"{self.ticker}_{self.interval}.csv"

    @property
    def stats_csv_path(self) -> Path:
        return self.reports_dir / f"{self.ticker}_{self.interval}_descriptive_stats.csv"


# ---------------------------------------------------------
# 2) Data Understanding
# ---------------------------------------------------------
class DataUnderstanding:
    """
    Führt den Data-Understanding-Step aus:
    - Daten laden
    - Spalten erklären
    - Statistiken berechnen
    - Plots erzeugen
    - Findings loggen
    """

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg

    # ---------- Helper ----------

    def _ensure_dirs(self) -> None:
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        print(f"📂 Lade Rohdaten aus: {self.cfg.raw_csv_path}")
        df = pd.read_csv(self.cfg.raw_csv_path)

        # Zeitspalte nach Datetime (mit UTC) parsen und Zeitzone entfernen
        dt_col = "Datetime" if "Datetime" in df.columns else "Date"
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        df[dt_col] = df[dt_col].dt.tz_localize(None)

        # sortieren und NaT-Zeilen droppen
        df = df.sort_values(dt_col)
        df = df[df[dt_col].notna()].reset_index(drop=True)

        print(f"✅ Daten geladen, Shape: {df.shape}")
        return df

    # ---------- 1) Spalten erklären ----------

    def explain_columns(self, df: pd.DataFrame) -> None:
        print("\n📘 Relevante Daten-Spalten:")

        explanations = {
            "Datetime": "Zeitstempel der Intraday-Periode (z. B. 1h-Bar).",
            "Date": "Handelstag (nur bei Daily-Daten).",
            "Open": "Kurs zu Beginn des jeweiligen Intervalls.",
            "High": "Höchster Kurs innerhalb des Intervalls.",
            "Low": "Tiefster Kurs innerhalb des Intervalls.",
            "Close": "Kurs am Ende des Intervalls.",
            "Adj Close": "Um Dividenden/Splits bereinigter Schlusskurs.",
            "Volume": "Gehandeltes Volumen im Intervall.",
        }

        for col in df.columns:
            desc = explanations.get(col, "(Keine spezielle Beschreibung hinterlegt.)")
            print(f"- {col}: {desc}")

    # ---------- 2) Deskriptive Statistiken ----------

    def compute_descriptive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n📊 Berechne deskriptive Statistiken...")

        num_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        num_cols = [c for c in num_cols if c in df.columns]

        # sicherstellen, dass die Spalten wirklich numerisch sind
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        stats = df[num_cols].describe().T  # count, mean, std, min, max, etc.

        self._ensure_dirs()
        stats.to_csv(self.cfg.stats_csv_path)
        print(f"✅ Statistiken gespeichert unter: {self.cfg.stats_csv_path}")

        print("\n🔎 Auszug aus den Statistiken:")
        print(stats)

        return stats

    # ---------- 3) Plots ----------

    def create_plots(self, df: pd.DataFrame) -> None:
        print("\n📈 Erzeuge Plots...")

        self._ensure_dirs()
        dt_col = "Datetime" if "Datetime" in df.columns else "Date"

        # Maske ohne NaT
        mask = df[dt_col].notna()

        # Time Series des Close-Preises
        plt.figure()
        plt.plot(df.loc[mask, dt_col], df.loc[mask, "Close"])
        plt.xlabel("Zeit")
        plt.ylabel("Close")
        plt.title(f"{self.cfg.ticker} Close Price ({self.cfg.interval})")
        plt.tight_layout()
        ts_path = self.cfg.figures_dir / f"{self.cfg.ticker}_{self.cfg.interval}_close_timeseries.png"
        plt.savefig(ts_path)
        plt.close()
        print(f"✅ Time-Series-Plot gespeichert: {ts_path}")

        # Intraday Returns
        df["return"] = df["Close"].pct_change()

        plt.figure()
        df["return"].hist(bins=100)
        plt.xlabel("Return")
        plt.ylabel("Häufigkeit")
        plt.title(f"{self.cfg.ticker} Intraday Returns ({self.cfg.interval})")
        plt.tight_layout()
        ret_path = self.cfg.figures_dir / f"{self.cfg.ticker}_{self.cfg.interval}_return_hist.png"
        plt.savefig(ret_path)
        plt.close()
        print(f"✅ Return-Histogramm gespeichert: {ret_path}")

        # Volumen-Verteilung
        plt.figure()
        df["Volume"].hist(bins=100)
        plt.xlabel("Volume")
        plt.ylabel("Häufigkeit")
        plt.title(f"{self.cfg.ticker} Volumen-Verteilung ({self.cfg.interval})")
        plt.tight_layout()
        vol_path = self.cfg.figures_dir / f"{self.cfg.ticker}_{self.cfg.interval}_volume_hist.png"
        plt.savefig(vol_path)
        plt.close()
        print(f"✅ Volumen-Histogramm gespeichert: {vol_path}")

    # ---------- 4) Findings ----------

    def print_findings(self, stats: pd.DataFrame) -> None:
        print("\n📝 Erste Findings (kurz & knackig):")

        close_stats = stats.loc["Close"]
        vol_stats = stats.loc["Volume"]

        print(
            f"- Close: Mittelwert ≈ {close_stats['mean']:.2f}, "
            f"Min = {close_stats['min']:.2f}, Max = {close_stats['max']:.2f}"
        )
        print(
            f"- Volume: Median ≈ {vol_stats['50%']:.0f}, "
            f"starke Unterschiede zwischen Min ({vol_stats['min']:.0f}) "
            f"und Max ({vol_stats['max']:.0f})."
        )
        print(
            "- Die Return-Verteilung ist (wie typisch für Finanzdaten) "
            "rechtsschief mit vielen kleinen Bewegungen und einigen Ausreißern."
        )

    # ---------- Orchestrierung ----------

    def run(self) -> pd.DataFrame:
        df = self.load_data()
        self.explain_columns(df)
        stats = self.compute_descriptive_stats(df)
        self.create_plots(df)
        self.print_findings(stats)
        return df


def main() -> None:
    cfg = ProjectConfig()
    du = DataUnderstanding(cfg)
    du.run()


if __name__ == "__main__":
    main()
