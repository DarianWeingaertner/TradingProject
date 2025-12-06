# scripts/01_data_acquisition.py
"""
Step 01 — Data Acquisition (MSCI World ETF, Intraday via Alpaca)
-----------------------------------------------------------------
Ziel:
- Minutendaten (z. B. 1Min) für den MSCI World ETF laden
- Quelle: Alpaca Market Data API (historische Daten)
- Ticker: URTH (iShares MSCI World ETF, US-Listing)

Hinweis:
- Alpaca liefert historische Minutendaten kostenlos (Paper-Account reicht).
- Wir ziehen hier die letzten `days_back` Tage als Minutendaten.
- Output landet als CSV unter: data/raw/{symbol}_{interval}.csv
  z. B.: data/raw/URTH_1Min.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import os

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ---------------------------------------------------------
# 1) Projekt-Konfiguration
# ---------------------------------------------------------
@dataclass
class ProjectConfig:
    """
    Hält alle wichtigen Parameter für den Datenzugriff.

    - symbol: Instrument, das wir analysieren (hier: MSCI World ETF = URTH)
    - interval: Zeitauflösung (Alpaca-Namenskonvention, z. B. "1Min", "5Min")
    - days_back: wie viele Tage in die Vergangenheit wir Minutendaten holen
    - base_dir: Basisverzeichnis des Projekts
    """

    symbol: str = "URTH"
    interval: str = "1Min"  # Minutendaten
    days_back: int = 30     # wie viele Tage zurück

    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def raw_csv_path(self) -> Path:
        """
        Zielpfad für die Rohdaten-CSV, z. B. data/raw/URTH_1Min.csv
        """
        filename = f"{self.symbol}_{self.interval}.csv"
        return self.raw_data_dir / filename

    # Alpaca-Credentials (werden aus Umgebungsvariablen gelesen)
    @property
    def alpaca_api_key(self) -> str:
        return os.environ.get("APCA_API_KEY_ID", "")

    @property
    def alpaca_secret_key(self) -> str:
        return os.environ.get("APCA_API_SECRET_KEY", "")


# ---------------------------------------------------------
# 2) Alpaca Market Data Client
# ---------------------------------------------------------
class AlpacaMarketDataClientWrapper:
    """
    Kapselt die Logik zum Laden von Kursdaten über Alpaca.
    """

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg

        if not self.cfg.alpaca_api_key or not self.cfg.alpaca_secret_key:
            raise RuntimeError(
                "Alpaca API Keys nicht gesetzt. "
                "Bitte Umgebungsvariablen APCA_API_KEY_ID und "
                "APCA_API_SECRET_KEY setzen."
            )

        # Historischer Daten-Client
        self.client = StockHistoricalDataClient(
            api_key=self.cfg.alpaca_api_key,
            secret_key=self.cfg.alpaca_secret_key,
        )

    def _to_timeframe(self) -> TimeFrame:
        """
        Mappt den Text in cfg.interval auf ein TimeFrame-Objekt.
        Aktuell unterstützen wir nur Minutendaten (1Min).
        """
        if self.cfg.interval == "1Min":
            return TimeFrame.Minute
        # falls du später z. B. "5Min" o. Ä. nutzt, könntest du hier erweitern
        raise ValueError(f"Unsupported interval for Alpaca: {self.cfg.interval}")

    def download_intraday_minutes(self) -> pd.DataFrame:
        """
        Lädt Minutendaten für das konfigurierte Symbol und gibt
        ein Pandas DataFrame mit OHLCV zurück.
        """

        # Ende = jetzt (UTC, glatt auf volle Minute)
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = end - timedelta(days=self.cfg.days_back)

        timeframe = self._to_timeframe()

        print(
            f"📥 Lade Minutendaten von Alpaca: "
            f"Symbol={self.cfg.symbol}, "
            f"Interval={self.cfg.interval}, "
            f"Start={start.isoformat()}, "
            f"End={end.isoformat()}"
        )

        request = StockBarsRequest(
            symbol_or_symbols=self.cfg.symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment="raw",
            feed="iex",

        )

        bars = self.client.get_stock_bars(request)
        df = bars.df

        if df is None or df.empty:
            raise RuntimeError(
                "❌ Es wurden keine Daten von Alpaca zurückgeliefert. "
                "Prüfe Symbol, Zeitraum, API-Keys und ob das Asset unterstützt wird."
            )

        # Wenn mehrere Symbole abgefragt werden, ist der Index MultiIndex.
        # Hier nutzen wir nur ein Symbol -> ggf. herausfiltern.
        if isinstance(df.index, pd.MultiIndex):
            # MultiIndex-Level heißt normalerweise "symbol"
            df = df.xs(self.cfg.symbol, level="symbol")

        # Index schöner benennen
        df = df.sort_index()
        df.index.name = "timestamp"

        # Spaltennamen dokumentieren (typischerweise: open, high, low, close, volume, vwap, trade_count)
        print(f"✅ {len(df)} Zeilen geladen. Spalten: {list(df.columns)}")

        return df


# ---------------------------------------------------------
# 3) Data Acquisition Workflow
# ---------------------------------------------------------
class MSCIWorldDataAcquisitionAlpaca:
    """
    Orchestriert den kompletten Ablauf:
    - sicherstellen, dass Ordner existieren
    - Minutendaten mit AlpacaMarketDataClientWrapper laden
    - CSV-Datei speichern
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        client: AlpacaMarketDataClientWrapper,
    ) -> None:
        self.cfg = cfg
        self.client = client

    def ensure_directories(self) -> None:
        """
        Stellt sicher, dass data/raw existiert.
        """
        self.cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """
        Führt den kompletten ETL-Schritt aus:
        - Ordner anlegen
        - Daten laden
        - CSV schreiben
        """
        self.ensure_directories()

        df = self.client.download_intraday_minutes()

        # CSV speichern
        self.cfg.raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cfg.raw_csv_path)

        print(
            f"💾 Rohdaten gespeichert unter: {self.cfg.raw_csv_path} "
            f"({len(df)} Zeilen)."
        )


# ---------------------------------------------------------
# 4) Skript-Einstiegspunkt
# ---------------------------------------------------------
def main() -> None:
    cfg = ProjectConfig()  # ggf. symbol/interval/days_back anpassen
    client = AlpacaMarketDataClientWrapper(cfg)
    acquisition = MSCIWorldDataAcquisitionAlpaca(cfg, client)
    acquisition.run()


if __name__ == "__main__":
    main()
