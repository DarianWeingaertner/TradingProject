# scripts/01_data_acquisition.py
"""
Step 01 — Data Acquisition (MSCI World ETF, Intraday)
--------------------------------------------------------
Ziel:
- Intraday-Kursdaten (z. B. stündlich) für den MSCI World ETF laden
- Quelle: Yahoo Finance (via yfinance)
- Ticker: URTH (iShares MSCI World ETF, US-Listing)

Hinweis:
- Yahoo Finance stellt Intraday-Daten (z. B. 1h) nur für einen
  begrenzten Zeitraum (ca. 730 Tage) bereit.
  -> Deshalb verwenden wir für Intraday-Intervalle 'period' statt
     einer festen Start-/End-Zeitspanne.

Outputs:
- CSV-Datei: data/raw/URTH_<interval>.csv

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
    - start_date / end_date: werden primär für Daily-Daten verwendet
    - period: wird für Intraday-Daten verwendet (z. B. "730d")
    - interval: Zeitauflösung (z. B. "1h", "15m", "1d")
    - base_dir: Basisverzeichnis des Projekts
    """
    ticker: str = "URTH"
    start_date: str = "2012-01-01"
    end_date: Optional[str] = None
    interval: str = "1h"          # z. B. "1h", "15m", "1d"
    period: Optional[str] = "730d"  # für Intraday sinnvoll, für Daily ignorierbar
    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interval_label(self) -> str:
        return self.interval

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / f"{self.ticker}_{self.interval_label}.csv"

    @property
    def is_intraday(self) -> bool:
        """
        True, wenn es sich um ein Intraday-Intervall handelt,
        bei dem Yahoo die Historie begrenzt (z. B. "1m", "5m", "15m", "30m", "1h").
        """
        return self.interval.endswith("m") or self.interval.endswith("h")


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
        if self.cfg.is_intraday:
            # Intraday: Yahoo erlaubt nur begrenzten Zeitraum -> period verwenden
            print(
                f"📥 Lade Intraday-Daten von Yahoo Finance: "
                f"Ticker={self.cfg.ticker}, "
                f"Period={self.cfg.period}, "
                f"Interval={self.cfg.interval}"
            )
            df = yf.download(
                self.cfg.ticker,
                period=self.cfg.period,
                interval=self.cfg.interval,
                auto_adjust=False,
                progress=True,
            )
        else:
            # Daily: normal mit Start-/Enddatum
            print(
                f"📥 Lade Daily-Daten von Yahoo Finance: "
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
                auto_adjust=False,
                progress=True,
            )

        if df.empty:
            raise RuntimeError(
                "❌ Es wurden keine Daten geladen. "
                "Bitte Ticker/Zeitraum/Interval/Internetverbindung prüfen."
            )

        # Index (Zeitstempel) als Spalte speichern
        df = df.reset_index()

        # kleine Info zur Kontrolle
        print(f"✅ Anzahl geladener Zeilen: {len(df)}")
        print(f"🔁 Zeitbereich: {df.iloc[0, 0]} bis {df.iloc[-1, 0]}")

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
    cfg = ProjectConfig()  # hier ggf. interval/period anpassen
    client = YahooFinanceClient(cfg)
    acquisition = MSCIWorldDataAcquisition(cfg, client)
    acquisition.run()


if __name__ == "__main__":
    main()
