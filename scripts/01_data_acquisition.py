"""
Step 01 — Data Acquisition (Alpaca)
-----------------------------------
Ziel:
- SPY als 1Min (intraday)
- GLD als 1Day (daily)
- mindestens 5 Jahre zurück

Output:
- data/raw/SPY_1Min.csv
- data/raw/GLD_1Min.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import time

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()


@dataclass
class ProjectConfig:
    primary_symbol: str = "SPY"
    gold_symbol: str = "GLD"

    # separate intervals per symbol
    primary_interval: str = "1Min"
    gold_interval: str = "1Min"

    years_back: int = 5

    # Chunking nur für Intraday sinnvoll
    chunk_download: bool = True
    chunk_days_intraday: int = 30

    feed: str = "iex"
    adjustment: str = "raw"

    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def symbol_configs(self) -> list[tuple[str, str]]:
        return [
            (self.primary_symbol, self.primary_interval),
            (self.gold_symbol, self.gold_interval),
        ]

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    def raw_csv_path(self, symbol: str, interval: str) -> Path:
        return self.raw_data_dir / f"{symbol}_{interval}.csv"

    @property
    def alpaca_api_key(self) -> str:
        return os.environ.get("APCA_API_KEY_ID", "")

    @property
    def alpaca_secret_key(self) -> str:
        return os.environ.get("APCA_API_SECRET_KEY", "")


class AlpacaMarketDataClientWrapper:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg

        if not self.cfg.alpaca_api_key or not self.cfg.alpaca_secret_key:
            raise RuntimeError(
                "Alpaca API Keys nicht gesetzt.\n"
                "Bitte APCA_API_KEY_ID und APCA_API_SECRET_KEY setzen (z.B. via .env)."
            )

        self.client = StockHistoricalDataClient(
            api_key=self.cfg.alpaca_api_key,
            secret_key=self.cfg.alpaca_secret_key,
        )

    @staticmethod
    def _to_timeframe(interval: str) -> TimeFrame:
        if interval == "1Min":
            return TimeFrame.Minute
        if interval == "1Day":
            return TimeFrame.Day
        raise ValueError(f"Unsupported interval: {interval}")

    def _fetch(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._to_timeframe(interval),
            start=start,
            end=end,
            adjustment=self.cfg.adjustment,
            feed=self.cfg.feed,
        )

        bars = self.client.get_stock_bars(request)
        df = bars.df

        if df is None or df.empty:
            return pd.DataFrame()

        # Bei einem Symbol kann trotzdem MultiIndex kommen
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df = df.sort_index()
        df.index.name = "timestamp"
        return df

    def download_bars(self, symbol: str, interval: str) -> pd.DataFrame:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

        # Für Daily ist "bis heute (00:00 UTC)" sauberer als "bis jetzt"
        if interval == "1Day":
            end = now.replace(hour=0, minute=0)
        else:
            end = now

        start = end - timedelta(days=int(365 * self.cfg.years_back))

        print(
            f"📥 Lade Daten: Symbol={symbol}, Interval={interval}, "
            f"Start={start.isoformat()}, End={end.isoformat()}, Feed={self.cfg.feed}"
        )

        # Daily: kein Chunking nötig
        if interval == "1Day":
            df = self._fetch(symbol, interval, start, end)
            if df.empty:
                raise RuntimeError(f"❌ Keine Daten erhalten für {symbol} ({interval}).")
            df = df[~df.index.duplicated(keep="last")]
            print(f"✅ Geladen ({symbol}, {interval}): {len(df)} Zeilen.")
            return df

        # Intraday: chunked Download empfohlen
        if not self.cfg.chunk_download:
            df = self._fetch(symbol, interval, start, end)
            if df.empty:
                raise RuntimeError(f"❌ Keine Daten erhalten für {symbol} ({interval}).")
            df = df[~df.index.duplicated(keep="last")]
            print(f"✅ Geladen ({symbol}, {interval}): {len(df)} Zeilen.")
            return df

        all_parts: list[pd.DataFrame] = []
        cur = start
        step = timedelta(days=self.cfg.chunk_days_intraday)

        while cur < end:
            nxt = min(cur + step, end)
            print(f"  → {symbol} Chunk: {cur.date()} bis {nxt.date()} ...")

            part = self._fetch(symbol, interval, cur, nxt)
            if not part.empty:
                all_parts.append(part)

            # kleine Pause gegen Rate-Limits
            time.sleep(0.1)
            cur = nxt

        if not all_parts:
            raise RuntimeError(f"❌ Keine Daten erhalten für {symbol} ({interval}).")

        df = pd.concat(all_parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        print(f"✅ Gesamt geladen ({symbol}, {interval}): {len(df)} Zeilen.")
        return df


class DataAcquisition:
    def __init__(self, cfg: ProjectConfig, client: AlpacaMarketDataClientWrapper) -> None:
        self.cfg = cfg
        self.client = client

    def run(self) -> None:
        self.cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)

        for symbol, interval in self.cfg.symbol_configs:
            df = self.client.download_bars(symbol, interval)
            out = self.cfg.raw_csv_path(symbol, interval)
            df.to_csv(out)
            print(f"💾 Gespeichert: {out} ({len(df)} Zeilen)")


def main() -> None:
    cfg = ProjectConfig()
    client = AlpacaMarketDataClientWrapper(cfg)
    DataAcquisition(cfg, client).run()


if __name__ == "__main__":
    main()