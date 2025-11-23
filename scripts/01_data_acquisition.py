# scripts/01_data_acquisition.py
"""
Step 01 — Data Acquisition (MSCI World ETF)
--------------------------------------------------------
Ziel:
- End-of-Day-Daten (1d) für den MSCI World ETF laden
- Quelle: Yahoo Finance (via yfinance)
- Ticker: URTH (iShares MSCI World ETF, US-Listing)

Outputs:
- CSV-Datei: data/raw/URTH_1d.csv

Klassen:
- ProjectConfig: hält zentrale Konfiguration und Pfade
- YahooFinanceClient: lädt OHLCV-Daten von Yahoo Finance
- MSCIWorldDataAcquisition: orchestriert Download + Speichern
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd


# ---------------------------------------------------------
# 1) Projekt-Konfiguration
# ---------------------------------------------------------
@dataclass
class ProjectConfig:
    """
    Hält alle wichtigen Parameter für das Projekt.

    - ticker: Instrument, das wir analysieren (hier: MSCI World ETF)
    - start_date: Startdatum für historische Daten
    - end_date: Enddatum (None = bis heute)
    - interval: Zeitauflösung (1d = End-of-Day)
    - base_dir: Basisverzeichnis des Projekts
    """
    ticker: str = "URTH"
    start_date: str = "2012-01-01"  # URTH gibt es erst ab 2012
    end_date: Optional[str] = None
    interval: str = "1d"
    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / f"{self.ticker}_1d.csv"


# ---------------------------------------------------------
# 2) Client für Yahoo Finance
# ---------------------------------------------------------
class YahooFinanceClient:
    """
    Kapselt die Logik zum Laden von Kursdaten über yfinance.
    """

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg

    def download_ohlcv(self) -> pd.DataFrame:
        """
        Lädt OHLCV-Daten für den konfigurierten Ticker
        und gibt ein Pandas DataFrame zurück.
        """
        print(
            f"📥 Lade Daten von Yahoo Finance: "
            f"Ticker={self.cfg.ticker}, "
            f"Start={self.cfg.start_date}, "
            f"End={self.cfg.end_date}, "
            f"Interval={self.cfg.interval}"
        )
        df = yf.download(
            self.cfg.ticker,
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            interval=self.cfg.interval,
            auto_adjust=False,  # Adj Close behalten wir separat
            progress=True,
        )

        if df.empty:
            raise RuntimeError(
                "❌ Es wurden keine Daten geladen. "
                "Bitte Ticker/Zeitraum/Internetverbindung prüfen."
            )

        # Index (Datum) als Spalte speichern
        df = df.reset_index()
        return df


# ---------------------------------------------------------
# 3) Data Acquisition Workflow
# ---------------------------------------------------------
class MSCIWorldDataAcquisition:
    """
    Orchestriert den kompletten Ablauf:
    - sicherstellen, dass Ordner existieren
    - Daten mit YahooFinanceClient laden
    - CSV-Datei speichern
    """

    def __init__(self, cfg: ProjectConfig, client: YahooFinanceClient) -> None:
        self.cfg = cfg
        self.client = client

    def run(self) -> pd.DataFrame:
        """
        Führt den Data-Acquisition-Workflow aus und gibt die Daten zurück.
        """
        # Ordner anlegen
        self.cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Daten laden
        df = self.client.download_ohlcv()

        # CSV speichern
        df.to_csv(self.cfg.raw_csv_path, index=False)
        print(f"✅ Daten gespeichert unter: {self.cfg.raw_csv_path}")

        # kleine Vorschau zur Kontrolle / Präsentation
        print("\n🔎 Vorschau auf die Rohdaten:")
        print(df.head())

        return df


# ---------------------------------------------------------
# 4) Skript-Einstiegspunkt
# ---------------------------------------------------------
def main() -> None:
    cfg = ProjectConfig()
    client = YahooFinanceClient(cfg)
    acquisition = MSCIWorldDataAcquisition(cfg, client)
    acquisition.run()


if __name__ == "__main__":
    main()